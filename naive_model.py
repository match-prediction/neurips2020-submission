import torch.nn as nn
import torch
import torch.nn.functional as F

from module_RP import RP_layers
from module_G import G_layers
# from module_RP_v2 import RP_layers
# from module_G_v2 import G_layers

from torch.autograd import Variable

class singlenet(nn.Module):
    def __init__(self, depth, N, M):
        super(singlenet, self).__init__()
        self.T = depth
        self.N = N
        self.M = M
        self.layer = nn.Linear(N,1)
        

    def forward(self, edges): 
        edges = edges.long()
        # edges = torch.cat([edges, edges], dim=0)
        idx = torch.arange(edges.size()[0]).view(-1, 1).repeat(1, edges.size()[1])
        # team A index
        idx_A = idx[:, :self.M].contiguous().view(-1)
        edges_A = edges[:, :self.M].contiguous().view(-1)
        # team B index
        idx_B = idx[:, self.M:].contiguous().view(-1)
        edges_B = edges[:, self.M:].contiguous().view(-1)

        # print(edges.size()[0], self.N)
        single_input = torch.zeros([edges.size()[0], self.N])
        single_input[idx_A, edges_A] = 1
        single_input[idx_B, edges_B] = -1
        pred_logit = self.layer(single_input)
        pred = torch.sigmoid(pred_logit)

        return pred



