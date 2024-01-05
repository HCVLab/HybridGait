# -*- coding: utf-8 -*-
import random
import numpy as np
import torch
import sys
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
from torch.autograd import Variable
from einops import rearrange

import cv2

from PIL import Image
from ..visualization import array_to_cam, def_visualization, deform_visualization
from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks, BasicConv2d, constant_init


class HybridGait(BaseModel):
    def __init__(self, cfgs, is_training):
        super().__init__(cfgs, is_training)

    def build_network(self, model_cfg):

        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])

        # width, height
        self.filter = constant_init([16, 16], 4).res

        in_c = model_cfg['backbone_cfg']['in_channels']
        self.set_block1 = nn.Sequential(BasicConv2d(in_c[0], in_c[1], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        BasicConv2d(in_c[1], in_c[1], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        self.set_block2 = nn.Sequential(BasicConv2d(in_c[1], in_c[2], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        BasicConv2d(in_c[2], in_c[2], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        self.set_block3 = nn.Sequential(BasicConv2d(in_c[2], in_c[3], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        BasicConv2d(in_c[3], in_c[3], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True))

        self.set_block1_smpl = nn.Sequential(BasicConv2d(in_c[0], in_c[1], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        BasicConv2d(in_c[1], in_c[1], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))
        self.set_block2_smpl = nn.Sequential(BasicConv2d(in_c[1], in_c[2], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        BasicConv2d(in_c[2], in_c[2], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))

        self.set_block_tmp = nn.Sequential(BasicConv2d(in_c[2], in_c[3], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        BasicConv2d(in_c[3], in_c[3], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True))
        self.set_block_tmp = SetBlockWrapper(self.set_block_tmp)


        self.set_block1 = SetBlockWrapper(self.set_block1)
        self.set_block2 = SetBlockWrapper(self.set_block2)
        self.set_block3 = SetBlockWrapper(self.set_block3)

        self.set_block1_smpl = SetBlockWrapper(self.set_block1_smpl)
        self.set_block2_smpl = SetBlockWrapper(self.set_block2_smpl)


        self.conv_offset = nn.Conv2d(256, 18, kernel_size=3, stride=1, padding=1)
        nn.init.constant_(self.conv_offset.weight, 0)
        self.conv_offset = SetBlockWrapper(self.conv_offset)

        self.conv_mask = nn.Conv2d(256, 9, kernel_size=3, stride=1, padding=1)
        nn.init.constant_(self.conv_mask.weight, 0)
        self.conv_mask = SetBlockWrapper(self.conv_mask)
        self.groups = 1
        self.regular_conv = nn.Conv2d(in_channels=128,
                                      out_channels=256,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1)

        self.token_fc = nn.Linear(3, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.sp_encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4)
        self.sp_transformer = nn.TransformerEncoder(self.sp_encoder_layer, num_layers=3)
        self.tp_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8)
        self.tp_transformer = nn.TransformerEncoder(self.tp_encoder_layer, num_layers=3)


    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        sils = ipts[0][0]    # [n, s, h, w]
        smpl_sil = ipts[1][0]
        tn, ts, _ = ipts[2][0].shape
        if ts > 30:
            pose = torch.zeros((1,30,66))
            seqL_t = seqL[0].data.cpu()
            seqL_t.numpy().tolist()
            start = [0] + np.cumsum(seqL_t).tolist()[:-1]
            i = 0
            for curr_start, curr_seqL in zip(start, seqL_t):
                curr_start = int(curr_start)
                curr_seqL = int(curr_seqL)
                if curr_seqL >= 30:
                    print(curr_seqL)
                    startidx = curr_start
                    pose[i] = ipts[2][0][0][startidx:startidx+30,3:69]  # [n, s, d]
                else:
                    print(curr_seqL)
                    iteranum = int(30/curr_seqL)
                    iteranum_rest = 30%curr_seqL
                    for j in range (iteranum):
                        starti = 0 + j * curr_seqL
                        endi = starti + curr_seqL
                        pose[i,starti:endi,:] = ipts[2][0][0][0:curr_seqL,3:69]
                    starti = iteranum * curr_seqL
                    endi = 31
                    pose[i,starti:endi,:] = ipts[2][0][0][0:iteranum_rest,3:69]
                i = i + 1
            pose = pose.cuda()
        elif ts == 30:
            pose = ipts[2][0][:,:,3:69]
        else:
            iteranum = int(30/ts)
            iteranum_rest = 30%ts
            pose = torch.zeros((tn,30,66))
            for i in range (iteranum):
                starti = 0 + i * ts
                endi = starti + ts
                pose[:,starti:endi,:] = ipts[2][0][:,:,3:69]
            starti = 0 + iteranum * ts
            endi = 31
            pose[:,starti:endi,:] = ipts[2][0][:,0:iteranum_rest,3:69]
            pose = pose.cuda()


        if len(sils.size()) == 4:
            sils = sils.unsqueeze(2)
        if len(smpl_sil.size()) == 4:
            smpl_sil = smpl_sil.unsqueeze(2)

        del ipts

        pn, ps, pd = pose.shape
        pose = rearrange(pose, 'n s (j d) -> (n s j) d', d=3)
        em_pose = F.relu(self.bn1(self.token_fc(pose)))
        em_pose = rearrange(em_pose, '(n s j) d -> j (n s) d', n=pn, s=ps, j=int(pd/3))
        em_pose = self.sp_transformer(em_pose)

        em_pose = rearrange(em_pose, 'j (n s) d -> (n s) d j', n=pn, s=ps)

        weight_filter = self.filter.unsqueeze(0).repeat(pn*ps, 1, 1).cuda()

        em_pose = torch.bmm(em_pose, weight_filter)
        em_pose = rearrange(em_pose, '(n s) c (h w) -> n s c h w', n=pn, s=ps, h=16, w=16)
        em_pose = self.set_block_tmp(em_pose)
        em_pose = torch.mean(em_pose, dim=2)
        em_pose = rearrange(em_pose, 'n s h w -> s n (h w)')

        em_pose = self.tp_transformer(em_pose)
        em_pose = rearrange(em_pose, 's n (h w) -> n s h w',h=16,w=16)
        outs_gt = torch.mean(em_pose, dim=1) + torch.max(em_pose, dim=1)[0]
        

        outs = self.set_block1(sils)
        out_ss = self.set_block1_smpl(smpl_sil)

        outs = self.set_block2(outs)
        out_ss = self.set_block2_smpl(out_ss)

        outs_n, outs_s, outs_c, outs_h, outs_w = outs.shape
        
        
        offmask = torch.cat((out_ss, outs), dim=2)
        offset = self.conv_offset(offmask)
        mask = torch.sigmoid(self.conv_mask(offmask))
        offset_vis = offset
        mask_vis = mask

        offset = rearrange(offset, 'n s c h w -> (n s) c h w')
        mask = rearrange(mask, 'n s c h w -> (n s) c h w')
        outs = rearrange(outs, 'n s c h w -> (n s) c h w')
        out_ss = rearrange(out_ss, 'n s c h w -> (n s) c h w')

        out_deform = torch.relu(torchvision.ops.deform_conv2d(input=out_ss, offset=offset, 
                                                   weight=self.regular_conv.weight, 
                                                   bias=self.regular_conv.bias,
                                                   mask=mask, padding=(1, 1)))

        outs = rearrange(outs, '(n s) c h w -> n s c h w', n=outs_n, s=outs_s)
        out_deform = rearrange(out_deform, '(n s) c h w -> n s c h w', n=outs_n, s=outs_s)



        outs_n, outs_s, outs_c, outs_h, outs_w = outs.shape
        zero_tensor = Variable(torch.zeros((outs_n, outs_s, outs_c, outs_h, outs_h-outs_w))).cuda()
        outs = torch.cat([outs, zero_tensor], -1)
        outs = rearrange(outs, 'n s c h w -> (n s c) h w')
        outs_gt_2 = outs_gt.unsqueeze(1).unsqueeze(1).repeat(1, outs_s, outs_c, 1, 1)
        outs_gt_2 = rearrange(outs_gt_2, 'n s c h w -> (n s c) h w')
        outs = torch.bmm(outs, outs_gt_2)
        outs = rearrange(outs, '(n s c) h w -> n s c h w', n=outs_n, s=outs_s, c=outs_c)

        outs = self.set_block3(outs)


        # Temporal Pooling, TP
        outs = self.TP(outs, seqL, dim=1)[0]
        outs_n, outs_c, outs_h, outs_w = outs.size()
        outs = rearrange(outs, 'n c h w -> (n c) h w')

        outs_gt = outs_gt.unsqueeze(1).repeat(1, outs_c, 1, 1)
        outs_gt = rearrange(outs_gt, 'n c h w -> (n c) h w')
        outs = torch.bmm(outs, outs_gt) # [n c d] n 256 256
        outs = rearrange(outs, '(n c) h w -> n c h w', n=outs_n, c=outs_c)

        out_deform = self.TP(out_deform, seqL, dim=1)[0]
        outs_n, outs_c, outs_h, outs_w = out_deform.shape
        zero_tensor = Variable(torch.zeros((outs_n, outs_c, outs_h, outs_h-outs_w))).cuda()
        out_deform = torch.cat([out_deform, zero_tensor], -1)
        outs = outs + out_deform

        feat = self.HPP(outs)  # [n, c, p] c=256
        feat = feat.permute(2, 0, 1).contiguous()  # [p, n, c]
        embed_1 = self.FCs(feat)  # [p, n, c]

    
        _, logits = self.BNNecks(embed_1)  # [p+1, n, c]

        embed_1 = embed_1.permute(1, 0, 2).contiguous()  # [n, p+1, c]
        logits = logits.permute(1, 0, 2).contiguous()  # [n, p+1, c] 


        n, s, _, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils.view(n*s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': embed_1
            }
        }
        return retval
