#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb
import numpy as np
from utils import accuracy


class LossFunction(nn.Module):

    def __init__(self, init_w=10.0, init_b=-5.0, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True

        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion = torch.nn.CrossEntropyLoss()

        print('Initialised the Proposed AngleProto for Mixup')

    def forward(self, x, label=None, permuted_label=None, lam=1):
        assert x.size()[1] >= 2

        out_anchor = torch.mean(x[:, 1:, :], 1)
        out_positive = x[:, 0, :]
        stepsize = out_anchor.size()[0]

        cos_sim_matrix = F.cosine_similarity(out_positive.unsqueeze(-1), out_anchor.unsqueeze(-1).transpose(0, 2))
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b

        #Get the max value in each row. Deduct the max value in each row to avoid stack overflow when taking exponential operation
        row_max, _ = torch.max(cos_sim_matrix, dim = 1)
        row_max = row_max.unsqueeze(-1)
        cos_sim_matrix = cos_sim_matrix - row_max

        if permuted_label is None:
            permuted_label = torch.from_numpy(np.asarray(range(0, stepsize))).cuda()
        if label is None:
            label = torch.from_numpy(np.asarray(range(0, stepsize))).cuda()

        permuted_cos_sim_matrix = cos_sim_matrix[:, permuted_label]
        similarity_j_j = torch.exp(cos_sim_matrix)
        similarity_j_j_diag = torch.diag(similarity_j_j).unsqueeze(-1)

        similarity_j_j_permuted = torch.exp(permuted_cos_sim_matrix)
        similarity_j_j_diag_permuted = torch.diag(similarity_j_j_permuted).unsqueeze(-1)

        numerator = lam * similarity_j_j_diag + (1 - lam) * similarity_j_j_diag_permuted
        denominator = torch.exp(cos_sim_matrix).sum(dim = 1).unsqueeze(-1)
        log_val = torch.log(numerator / denominator)
        nloss = -(1 / stepsize) * torch.sum(log_val)

        prec1_part1 = accuracy(cos_sim_matrix.detach(), label.detach(), topk=(1,))[0]
        prec1_part2 = accuracy(permuted_cos_sim_matrix.detach(), permuted_label.detach(), topk=(1,))[0]

        prec1 = lam * prec1_part1 + (1- lam) * prec1_part2

        return nloss, prec1