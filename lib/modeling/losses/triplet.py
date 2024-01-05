import torch
import torch.nn.functional as F

from .base import BaseLoss, gather_and_scale_wrapper


class TripletLoss(BaseLoss):
    def __init__(self, margin, loss_term_weight=1.0):
        super(TripletLoss, self).__init__(loss_term_weight)
        self.margin = margin

    @gather_and_scale_wrapper
    def forward(self, embeddings, labels):
        # embeddings: [n, p, c], label: [n]
        embeddings = embeddings.permute(
            1, 0, 2).contiguous()  # [n, p, c] -> [p, n, c]
        embeddings = embeddings.float()

        ref_embed, ref_label = embeddings, labels
        dist = self.ComputeDistance(embeddings, ref_embed)  # [p, n1, n2]
        mean_dist = dist.mean(1).mean(1)
        ap_dist, an_dist = self.Convert2Triplets(labels, ref_label, dist)
        dist_diff = ap_dist - an_dist
        loss = F.relu(dist_diff + self.margin)

        hard_loss = torch.max(loss, -1)[0]
        loss_avg, loss_num = self.AvgNonZeroReducer(loss)

        self.info.update({
            'loss': loss_avg.detach().clone(),
            'hard_loss': hard_loss.detach().clone(),
            'loss_num': loss_num.detach().clone(),
            'mean_dist': mean_dist.detach().clone()})

        return loss_avg, self.info

    def AvgNonZeroReducer(self, loss):
        eps = 1.0e-9
        loss_sum = loss.sum(-1)
        loss_num = (loss != 0).sum(-1).float()

        loss_avg = loss_sum / (loss_num + eps)
        loss_avg[loss_num == 0] = 0
        return loss_avg, loss_num

    def ComputeDistance(self, x, y):
        """
            x: [p, n_x, c]
            y: [p, n_y, c]
        """
        x2 = torch.sum(x ** 2, -1).unsqueeze(2)  # [p, n_x, 1]
        y2 = torch.sum(y ** 2, -1).unsqueeze(1)  # [p, 1, n_y]
        inner = x.matmul(y.transpose(-1, -2))  # [p, n_x, n_y]
        dist = x2 + y2 - 2 * inner
        dist = torch.sqrt(F.relu(dist))  # [p, n_x, n_y]
        return dist

    def Convert2Triplets(self, row_labels, clo_label, dist):
        """
            row_labels: tensor with size [n_r]
            clo_label : tensor with size [n_c]
        """
        # torch.set_printoptions(profile="full")
        matches = (row_labels.unsqueeze(1) ==
                   clo_label.unsqueeze(0)).byte()  # [n_r, n_c]
        diffenc = matches ^ 1  # [n_r, n_c]
        mask = matches.unsqueeze(2) * diffenc.unsqueeze(1)
        # print("mask.shape:{}".format(mask.shape))
        # print("mask:{}".format(mask))
        a_idx, p_idx, n_idx = torch.where(mask)

        # print("a_idx:{}".format(a_idx))
        # print("a_idx.shape:{}".format(a_idx.shape))
        # print("p_idx:{}".format(p_idx))
        # print("n_idx:{}".format(n_idx))

        ap_dist = dist[:, a_idx, p_idx]
        an_dist = dist[:, a_idx, n_idx]
        # print("ap_dist.shape:{}".format(ap_dist.shape))
        return ap_dist, an_dist
