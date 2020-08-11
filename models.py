import torch.nn as nn
import torch
import torch.nn.functional as F

from module_RP import RP_layers
from module_G import G_layers
from torch.autograd import Variable

class MatchNet(nn.Module):
    def __init__(self, depth_T, N, M):
        super(MatchNet, self).__init__()
        self.M = M
        self.T = depth_T
        
        self.RP = RP_layers(width_RP = 7, depth_RP = 3, M = M)
        self.G = G_layers(width_G = 9, depth_G = 3, M = M)       
        self.score_norm = nn.ModuleList([nn.BatchNorm1d(1) for _ in range(depth_T)])

    def forward(self, score, edges, label_idx): 
        

        N = score.size()[0]
        # iteation = # of applying RP's during 1st stage
        iteration = self.T
        
        # score = torch.exp(score)
        for depth in range(iteration):
            score = self.RP(score, edges)
            
        pred_logit = self.G(score, label_idx)
        pred = torch.sigmoid(pred_logit)

        return pred



