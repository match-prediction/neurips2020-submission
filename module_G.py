import numpy as np
import scipy.sparse as sp


from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F


class G_layers(nn.Module):
    
    def __init__(self, width_G, depth_G, M):
        super(G_layers, self).__init__()
        self.M = M
        
        width_G = 9
        depth_G = 3

        self.G_start = nn.Linear(2*M, width_G*M)
        self.G_hid = nn.ModuleList([nn.Linear(width_G*M, width_G*M) for _ in range(depth_G)])
        self.G_end = nn.Linear(width_G*M, 1)

        
    def forward(self, score, label_idx):
        M = self.M
        N = score.size()[0]
        for m in range(2*M):
            if m < 2*M:
                feat = score[label_idx.long()[:, m], :]
                if m == 0: 
                    node_input = feat
                else:
                    node_input = torch.cat([node_input, feat], dim = 1)
        
        pred_logit = F.relu(self.G_start(node_input))
        for G in self.G_hid:
            pred_logit = F.relu(G(pred_logit))
        pred_logit = self.G_end(pred_logit)

        return pred_logit