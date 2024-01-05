import torch
import math
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from copy import copy, deepcopy
from torch.nn.parameter import Parameter
from einops import rearrange
from utils import clones, is_list_or_tuple, np2var, ts2var


class constant_init():
    def __init__(self, feature_size, _k, **kwargs):
        super(constant_init, self).__init__(**kwargs)
        # (y x) -> (H W)
        # [9,8] represents the index of the first skeleton points in SMPL model
        self.Pose_position = [[9, 8], [10, 10], [10, 6], [7, 8],
                              [11, 11], [11, 5], [5, 8], [12, 12],
                              [12, 4], [4, 8], [13, 13], [13, 3],
                              [1, 8], [3, 9], [3, 7], [0, 8],
                              [2, 14], [2, 2], [6, 15], [6, 1],
                              [8, 16], [8, 0]]
        # 17=width(sum of number) of all horizontal skeleton joints 
        # 14=height(sum of number) of all vertical skeleton joints 
        self.HW_hop = (14, 17)

        w, h = feature_size[0], feature_size[1]
        query = torch.zeros([w*h, 2])
        for i in range(w):
            for j in range(h):
                index = i*h+j
                query[index][0] = i #w_index
                query[index][1] = j #h_index
        input_ = self.update_pose_position(h, w)
        k_index = self.k_nearest_neighbor(query, input_, _k)
        # print(k_index)
        res = torch.zeros([w*h, 22])
        res = res.scatter(1, k_index, 1/_k)
        res = res.permute(1, 0)
        # print("k:{}".format(_k))
        # print("res:{}".format(res))
        self.res = res


    def update_pose_position(self, h, w):
        Re_Pose_position = torch.tensor([[item[0] / self.HW_hop[0] * h, item[1] / self.HW_hop[1] * w]
                                for item in self.Pose_position])
        return Re_Pose_position

    def squared_distance(self, xyz1, xyz2):
        """
        Calculate the Euclidean squared distance between every two points.
        :param xyz1: the 1st set of points, [n_points_1, 3]
        :param xyz2: the 2nd set of points, [n_points_2, 3]
        :return: squared distance between every two points, [n_points_1, n_points_2]
        """
        assert xyz1.shape[-1] == xyz2.shape[-1] and xyz1.shape[-1] <= 3  # assert channel_last
        n_points1, n_points2 = xyz1.shape[0], xyz2.shape[0]
        dist = -2 * torch.matmul(xyz1, xyz2.permute(1, 0))
        dist += torch.sum(xyz1 ** 2, -1).view(n_points1, 1)
        dist += torch.sum(xyz2 ** 2, -1).view(1, n_points2)
        return dist
    
    def k_nearest_neighbor(self, _query_xyz, _input_xyz, k):
        dists = self.squared_distance(_query_xyz, _input_xyz)
        # print("dists:{}".format(dists))
        # print("dists.topk:{}".format(dists.topk(k, dim=1, largest=False)))
        return dists.topk(k, dim=1, largest=False).indices.to(torch.long)



class Time_Conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super(Time_Conv, self).__init__()
        self.conv_t = nn.Conv1d(in_channels, out_channels, kernel_size, bias=False)

    def forward(self, x):
        x = self.conv_t(x)
        return x


class HorizontalPoolingPyramid():
    """
        Horizontal Pyramid Matching for Person Re-identification
        Arxiv: https://arxiv.org/abs/1804.05275
        Github: https://github.com/SHI-Labs/Horizontal-Pyramid-Matching
    """

    def __init__(self, bin_num=None):
        if bin_num is None:
            bin_num = [16, 8, 4, 2, 1]
        self.bin_num = bin_num

    def __call__(self, x):
        """
            x  : [n, c, h, w]
            ret: [n, c, p] 
        """
        n, c = x.size()[:2]
        features = []
        for b in self.bin_num:
            z = x.view(n, c, b, -1)
            z = z.mean(-1) + z.max(-1)[0]
            features.append(z)
        return torch.cat(features, -1)


class SetBlockWrapper(nn.Module):
    def __init__(self, forward_block):
        super(SetBlockWrapper, self).__init__()
        self.forward_block = forward_block

    def forward(self, x, *args, **kwargs):
        """
            In  x: [n, s, c, h, w]
            Out x: [n, s, ...]
        """
        n, s, c, h, w = x.size()
        x = self.forward_block(x.contiguous().view(-1, c, h, w), *args, **kwargs)
        input_size = x.size()
        output_size = [n, s] + [*input_size[1:]]
        return x.contiguous().view(*output_size)


class SequenceBlockWrapper(nn.Module):
    def __init__(self, forward_block):
        super(SequenceBlockWrapper, self).__init__()
        self.forward_block = forward_block

    def forward(self, x, *args, **kwargs):
        """
            In  x: [n, s, c, h, w]
            Out x: [n, s, ...]
        """
        # n, s, c, h, w = x.size()
        x = rearrange(x, 'n s c h w -> n c s h w')
        x = self.forward_block(x, *args, **kwargs)
        x = rearrange(x, 'n c s h w -> n s c h w')
        return x


class PackSequenceWrapper(nn.Module):
    def __init__(self, pooling_func):
        super(PackSequenceWrapper, self).__init__()
        self.pooling_func = pooling_func

    def forward(self, seqs, seqL, seq_dim=1, **kwargs):
        """
            In  seqs: [n, s, ...]
            Out rets: [n, ...]
        """
        if seqL is None:
            return self.pooling_func(seqs, **kwargs)
        seqL = seqL[0].data.cpu().numpy().tolist()
        start = [0] + np.cumsum(seqL).tolist()[:-1]

        rets = []
        for curr_start, curr_seqL in zip(start, seqL):
            narrowed_seq = seqs.narrow(seq_dim, curr_start, curr_seqL)
            # save the memory
            # splited_narrowed_seq = torch.split(narrowed_seq, 256, dim=1)
            # ret = []
            # for seq_to_pooling in splited_narrowed_seq:
            #     ret.append(self.pooling_func(seq_to_pooling, keepdim=True, **kwargs)
            #                [0] if self.is_tuple_result else self.pooling_func(seq_to_pooling, **kwargs))
            rets.append(self.pooling_func(narrowed_seq, **kwargs))
        if len(rets) > 0 and is_list_or_tuple(rets[0]):
            return [torch.cat([ret[j] for ret in rets])
                    for j in range(len(rets[0]))]
        return torch.cat(rets)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x


class SeparateFCs(nn.Module):
    def __init__(self, parts_num, in_channels, out_channels, norm=False):
        super(SeparateFCs, self).__init__()
        self.p = parts_num
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, out_channels)))
        self.norm = norm

    def forward(self, x):
        """
            x: [p, n, c]
        """
        if self.norm:
            out = x.matmul(F.normalize(self.fc_bin, dim=1))
        else:
            out = x.matmul(self.fc_bin)
        return out



class GetRes():
    def __init__(self):
        super(GetRes, self).__init__()

    def __call__(self, x):
        """
            x:  [n, s, c, h, w]
            out:[n, s, c, h, w]
        """
        n, s, c, h, w = x.shape
        out = torch.cat((x[:,1:,:,:,:], x[:,-1,:,:,:].unsqueeze(1)), dim=1) - x
        return out


class TOIMG():
    def __init__(self):
        super(TOIMG, self).__init__()
    
    def __call__(self, x):
        """
            x:  [n, s, j, d]
            out:[n, s, d, h, w]
        """
        n, s, j, d = x.shape
        out = torch.zeros((n, s, 16, 16, d)).cuda()
        for i in range(16):
            for j in range(16):
                if int(i/4)==0 and int(j/4)==0:
                    out[:,:,i,j,:] = x[:,:,17,:]
                if int(i/4)==0 and int(j/4)==1:
                    out[:,:,i,j,:] = (x[:,:,15,:]+x[:,:,12,:]+x[:,:,17,:])/3
                if int(i/4)==0 and int(j/4)==2:
                    out[:,:,i,j,:] = (x[:,:,15,:]+x[:,:,12,:]+x[:,:,16,:])/3
                if int(i/4)==0 and int(j/4)==3:
                    out[:,:,i,j,:] = x[:,:,16,:]
                if int(i/4)==1 and int(j/4)==0:
                    out[:,:,i,j,:] = (x[:,:,19,:]+x[:,:,21,:])/2
                if int(i/4)==1 and int(j/4)==1:
                    out[:,:,i,j,:] = (x[:,:,14,:]+x[:,:,9,:]+x[:,:,6,:])/3
                if int(i/4)==1 and int(j/4)==2:
                    out[:,:,i,j,:] = (x[:,:,13,:]+x[:,:,9,:]+x[:,:,6,:])/3
                if int(i/4)==1 and int(j/4)==3:
                    out[:,:,i,j,:] = (x[:,:,18,:]+x[:,:,20,:])/2
                if int(i/4)==2 and int(j/4)==0:
                    out[:,:,i,j,:] = (x[:,:,2,:]+x[:,:,5,:])/2
                if int(i/4)==2 and int(j/4)==1:
                    out[:,:,i,j,:] = (x[:,:,3,:]+x[:,:,0,:]+x[:,:,2,:]+x[:,:,5,:])/4
                if int(i/4)==2 and int(j/4)==2:
                    out[:,:,i,j,:] = (x[:,:,3,:]+x[:,:,0,:]+x[:,:,1,:]+x[:,:,4,:])/4
                if int(i/4)==2 and int(j/4)==3:
                    out[:,:,i,j,:] = (x[:,:,1,:]+x[:,:,4,:])/2
                if int(i/4)==3 and int(j/4)==0:
                    out[:,:,i,j,:] = (x[:,:,8,:]+x[:,:,11,:])/2
                if int(i/4)==3 and int(j/4)==1:
                    out[:,:,i,j,:] = (x[:,:,5,:]+x[:,:,8,:]+x[:,:,11,:])/3
                if int(i/4)==3 and int(j/4)==2:
                    out[:,:,i,j,:] = (x[:,:,4,:]+x[:,:,7,:]+x[:,:,10,:])/3
                if int(i/4)==3 and int(j/4)==3:
                    out[:,:,i,j,:] = (x[:,:,7,:]+x[:,:,10,:])/2
        out = rearrange(out, 'n s h w d -> n s d h w')
        return out


class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()

        # Spatial transformer localization-network
        # 28*28 -> 10*10 -> 
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 12 * 7, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))


    def forward(self, x):
        
        # print("x:{}".format(x.shape))
        xs = self.localization(x)
        # print("xs:{}".format(xs.shape))
        xs = xs.view(-1, 10 * 12 * 7)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        # print("x:{}".format(x))
        # print("grid:{}".format(grid))
        if x.dtype != grid.dtype:
            grid = grid.type(x.dtype)
        # print("x:{}".format(x))
        # print("grid:{}".format(grid))
        x = F.grid_sample(x, grid)
        # print("x:{}".format(x.shape))
        # print("x:{}".format(x))

        return x

class SeparateBNNecks(nn.Module):
    """
        GaitSet: Bag of Tricks and a Strong Baseline for Deep Person Re-Identification
        CVPR Workshop:  https://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf
        Github: https://github.com/michuanhaohao/reid-strong-baseline
    """

    def __init__(self, parts_num, in_channels, class_num, norm=True, parallel_BN1d=True):
        super(SeparateBNNecks, self).__init__()
        self.p = parts_num
        self.class_num = class_num
        self.norm = norm
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.zeros(parts_num, in_channels, class_num)))
        if parallel_BN1d:
            self.bn1d = nn.BatchNorm1d(in_channels * parts_num)
        else:
            self.bn1d = clones(nn.BatchNorm1d(in_channels), parts_num)
        self.parallel_BN1d = parallel_BN1d

    def forward(self, x):
        """
            x: [p, n, c]
        """
        if self.parallel_BN1d:
            p, n, c = x.size()
            x = x.transpose(0, 1).contiguous().view(n, -1)  # [n, p*c]
            x = self.bn1d(x)
            x = x.view(n, p, c).permute(1, 0, 2).contiguous()
        else:
            x = torch.cat([bn(_.squeeze(0)).unsqueeze(0)
                           for _, bn in zip(x.split(1, 0), self.bn1d)], 0)  # [p, n, c]
        if self.norm:
            feature = F.normalize(x, dim=-1)  # [p, n, c]
            logits = feature.matmul(F.normalize(
                self.fc_bin, dim=1))  # [p, n, c]
        else:
            feature = x
            logits = feature.matmul(self.fc_bin)
        return feature, logits


class FocalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, halving, **kwargs):
        super(FocalConv2d, self).__init__()
        self.halving = halving
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, bias=False, padding=1, **kwargs)

    def forward(self, x):
        if self.halving == 0:
            z = self.conv(x)
        else:
            h = x.size(2)
            split_size = int(h // 2**self.halving)
            # print(split_size)
            z = x.split(split_size, 2)
            z = torch.cat([self.conv(_) for _ in z], 2)
        return z


class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias, **kwargs)

    def forward(self, ipts):
        '''
            ipts: [n, c, s, h, w]
            outs: [n, c, s, h, w]
        '''
        outs = self.conv3d(ipts)
        return outs


def RmBN2dAffine(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.requires_grad = False
            m.bias.requires_grad = False


class Bottleneck3D(nn.Module):
    def __init__(self, bottleneck2d, block, inflate_time=False, temperature=4, contrastive_att=True):
        super().__init__()
        self.conv1 = inflate_conv(bottleneck2d.conv1, time_dim=1)
        self.bn1 = inflate_batch_norm(bottleneck2d.bn1)
        if inflate_time == True:
            self.conv2 = block(bottleneck2d.conv2, temperature=temperature, contrastive_att=contrastive_att)
        else:
            self.conv2 = inflate_conv(bottleneck2d.conv2, time_dim=1)
        self.bn2 = inflate_batch_norm(bottleneck2d.bn2)
        self.conv3 = inflate_conv(bottleneck2d.conv3, time_dim=1)
        self.bn3 = inflate_batch_norm(bottleneck2d.bn3)
        self.relu = nn.ReLU(inplace=True)

        if bottleneck2d.downsample is not None:
            self.downsample = _inflate_downsample(bottleneck2d.downsample)
        else:
            self.downsample = None
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            # max_pool = inflate.MaxPool2dFor3dInput
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d

        self.g = conv_nd(self.in_channels, self.inter_channels,
                         kernel_size=1, stride=1, padding=0, bias=True)
        self.theta = conv_nd(self.in_channels, self.inter_channels,
                             kernel_size=1, stride=1, padding=0, bias=True)
        self.phi = conv_nd(self.in_channels, self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        # if sub_sample:
        #     self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
        #     self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))
        if sub_sample:
            if dimension == 3:
                self.g = nn.Sequential(self.g, max_pool((1, 2, 2)))
                self.phi = nn.Sequential(self.phi, max_pool((1, 2, 2)))
            else:
                self.g = nn.Sequential(self.g, max_pool(kernel_size=2))
                self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(self.inter_channels, self.in_channels,
                        kernel_size=1, stride=1, padding=0, bias=True),
                bn(self.in_channels)
            )
        else:
            self.W = conv_nd(self.inter_channels, self.in_channels,
                             kernel_size=1, stride=1, padding=0, bias=True)
        
        # init
        for m in self.modules():
            if isinstance(m, conv_nd):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, bn):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if bn_layer:
            nn.init.constant_(self.W[1].weight.data, 0.0)
            nn.init.constant_(self.W[1].bias.data, 0.0)
        else:
            nn.init.constant_(self.W.weight.data, 0.0)
            nn.init.constant_(self.W.bias.data, 0.0)


    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f = F.softmax(f, dim=-1)

        y = torch.matmul(f, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        y = self.W(y)
        z = y + x

        return z


class NonLocalBlock1D(NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NonLocalBlock2D(NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NonLocalBlock3D(NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)




class LocaltemporalAG(nn.Module):
    def __init__(self, inplanes, planes, dilation=1, bias=False, **kwargs):
        super(LocaltemporalAG, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1), stride=(3,1,1), bias=bias,padding=(0, 0, 0))

    def forward(self, x):
        out1 = self.conv1(x)
        out = F.leaky_relu(out1, inplace=True)
        return out

class BasicConv3d_gl(nn.Module):
    def __init__(self, inplanes, planes, kernel=3, bias=False, **kwargs):
        super(BasicConv3d_gl, self).__init__()
        self.ratio = 0.7
        self.convdl = nn.Conv3d(inplanes, planes, kernel_size=(kernel, kernel, kernel), bias=bias, padding=((kernel-1)//2, (kernel-1)//2, (kernel-1)//2))
        self.convdg = nn.Conv3d(inplanes, planes, kernel_size=(kernel, kernel, kernel), bias=bias, padding=((kernel-1)//2, (kernel-1)//2, (kernel-1)//2))
    def forward(self, x):
        n, c, t, h, w = x.shape
        mask_q = torch.zeros_like(x)
        mask_p = torch.zeros_like(x)
        dim_mask = int(h*0.7)
        indices = list(range(h))
        indices = np.random.choice(
                        indices, dim_mask, replace=False)
        for i in range(h):
            if i in indices:
                mask_p[:,:,:,i,:] = 1
            else:
                mask_q[:,:,:,i,:] = 1
        feature_p = self.convdl(torch.mul(x, mask_p))
        feature_p = F.leaky_relu(feature_p, inplace=True)
        feature_q = self.convdl(torch.mul(x, mask_q))
        feature_q = F.leaky_relu(feature_q, inplace=True)
        feature_p = feature_p + feature_q

        del mask_p
        del mask_q

        outg = self.convdg(x)
        outg = F.leaky_relu(outg, inplace=True)

        out = torch.cat((outg, feature_p), dim=3)
        return out


class BasicConv3d_p(nn.Module):
    def __init__(self, inplanes, planes, kernel=3, bias=False, p=2, FM=False, **kwargs):
        super(BasicConv3d_p, self).__init__()
        self.p = p
        self.fm = FM
        self.convdl = nn.Conv3d(inplanes, planes, kernel_size=(kernel, kernel, kernel), bias=bias, padding=((kernel-1)//2, (kernel-1)//2, (kernel-1)//2))
        self.convdg = nn.Conv3d(inplanes, planes, kernel_size=(kernel, kernel, kernel), bias=bias, padding=((kernel-1)//2, (kernel-1)//2, (kernel-1)//2))
    def forward(self, x):
        n, c, t, h, w = x.size()
        scale = h//self.p
        # print('p-',x.shape,n, c, t, h, w,'scale-',scale)
        feature = list()
        for i in range(self.p):
            temp = self.convdl(x[:,:,:,i*scale:(i+1)*scale,:])
            # print(temp.shape,i*scale,(i+1)*scale)
            feature.append(temp)

        outl = torch.cat(feature, 3)
        # print('outl-',outl.shape)
        outl = F.leaky_relu(outl, inplace=True)

        outg = self.convdg(x)
        outg = F.leaky_relu(outg, inplace=True)
        # print('outg-',outg.shape)
        if not self.fm:
            # print('1-1')
            out = outg + outl
        else:
            # print('1-2')
            out = torch.cat((outg, outl), dim=3)
        return out

class BasicConv3d_leaky(nn.Module):
    def __init__(self, inplanes, planes, dilation=1, bias=False, **kwargs):
        super(BasicConv3d_leaky, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 3, 3), bias=bias, dilation=(dilation, 1, 1), padding=(dilation, 1, 1))

    def forward(self, x):
        out = self.conv1(x)
        out = F.leaky_relu(out, inplace=True)
        return out

class Temporal(nn.Module):
    def __init__(self, inplanes, planes, bias=False, **kwargs):
        super(Temporal, self).__init__()

    def forward(self, x):
        
        out = torch.max(x, 2)[0]
        return out

def gem(x, p=6.5, eps=1e-6):
    # print('x-',x.shape)
    # print('xpow-',x.clamp(min=eps).pow(p).shape)
    # print(F.avg_pool2d(x.clamp(min=eps).pow(p), (1, x.size(-1))).shape)
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (1, x.size(-1))).pow(1./p)

class GeM(nn.Module):

    def __init__(self, p=6.5, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        # print('p-',self.p)
        return gem(x, p=self.p, eps=self.eps)
        
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'