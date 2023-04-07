import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class TotalLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.num_classes = opt.max_time
        self.delta_t = opt.delta_t
        self.coeff0 = opt.coeff0
        self.coeff1 = opt.coeff1
        self.coeff2 = opt.coeff2
        self.coeff3 = opt.coeff3
        self.coeff4 = opt.coeff4

    def forward(self, is_observed, output, target, output_b, target_b):
        ce = self.coeff4 * self.CELoss(is_observed, output, target, self.delta_t)

        output = F.softmax(output.squeeze(-1), dim=-1)
        output_b = F.softmax(output_b.squeeze(-1), dim=-1)

        meanvar, mean, var = self.MeanVarianceLoss(is_observed, output, target, self.delta_t, self.coeff0, self.coeff1, self.coeff3)

        disc = self.coeff2 * self.DiscLoss(is_observed, output, target, output_b, target_b)
        MAE = self.MAELoss(is_observed, output, target)
        TotalLoss = meanvar + disc + ce
        return TotalLoss, MAE, meanvar, mean, var, disc, ce

    def MeanVarianceLoss(self, is_observed, output, target, delta_t, coeff0, coeff1, coeff3):
        rank = torch.arange(0, self.num_classes, dtype=output.dtype, device=output.device)
        mean = torch.squeeze((output * rank).sum(1, keepdim=True), dim=1)
        is_censored = 1 - is_observed
        target_new = target + delta_t * is_censored

        mse = (mean - target_new) ** 2
        coeff = coeff0 * is_observed + coeff3 * is_censored
        mean_loss = (coeff * mse).mean() / 2.0
        b = (rank[None, :] - mean[:, None]) ** 2
        variance_loss = coeff1 * (output * b).sum(1, keepdim=True).mean()

        MVLoss = mean_loss + variance_loss
        return MVLoss, mean_loss, variance_loss

    def CELoss(self, is_observed, output, target, delta_t):
        criterion1 = nn.CrossEntropyLoss(reduction='mean').to(device=target.device)
        is_censored = 1 - is_observed
        target_new = target + delta_t * is_censored
        CELoss = criterion1(output, target_new.unsqueeze(1))
        return CELoss

    def DiscLoss(self, is_observed, output, target, output_b, target_b):
        cond_a = is_observed & (target < target_b)
        if torch.sum(cond_a) > 0:
            surv_probs_a = 1 - torch.cumsum(output, dim=1)
            mean_lifetimes_a = torch.sum(surv_probs_a, dim=1)

            surv_probs_b = 1 - torch.cumsum(output_b, dim=1)
            mean_lifetimes_b = torch.sum(surv_probs_b, dim=1)

            diff = mean_lifetimes_a[cond_a.bool()] - mean_lifetimes_b[cond_a.bool()]
            true_diff = target_b[cond_a.bool()] - target[cond_a.bool()]
            DiscLoss = torch.mean(nn.ReLU()(true_diff + diff))
        else:
            DiscLoss = torch.tensor(0.0, device=output.device, requires_grad=True)
        return DiscLoss


if __name__ == '__main__':
    x = torch.rand(3, 4)
    print(x)
    indices = torch.tensor([[0, 0], [2, 0]])
    print(torch.index_select(x, 0, indices))
    # ranks = torch.arange(0, 10).repeat(5, 1)
    # print(ranks)
    # i = torch.tensor([0, 1, 1, 0, 1])
    # i_repeated = i.unsqueeze(1).repeat(1, 10)
    # t = torch.tensor([3, 5, 4, 5, 0])
    # print(i)
    # print(i_repeated)
    # print(t)
    # for k in range(t.size(0)):
    #     if i[k] == 1:
    #         print(t[k])
    #         i_repeated[k][t[k] + 1:] = 0
    # print(i_repeated)
    # print(i_repeated * ranks)
