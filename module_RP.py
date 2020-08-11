import numpy as np
import scipy.sparse as sp


from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F




class RP_layers(nn.Module):
    
    def __init__(self, width_RP, depth_RP, M):
        super(RP_layers, self).__init__()
        
        self.M = M
        self.norm_factor = 2
        # self.norm_factor = 1
        
        width_RP = 7
        depth_RP = 3
        
        self.R_start = nn.Linear(2*M, width_RP*M)
        self.R_hid = nn.ModuleList([nn.Linear(width_RP*M, width_RP*M) for _ in range(depth_RP)])
        self.R_end = nn.Linear(width_RP*M, 2*M)
        
        self.P_start = nn.Linear(2*M, width_RP*M)
        self.P_hid = nn.ModuleList([nn.Linear(width_RP*M, width_RP*M) for _ in range(depth_RP)])
        self.P_end = nn.Linear(width_RP*M, 2*M)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, score, edges):
        
        M = self.M
        N = score.size()[0]
        edge_idx = edges[:, :-1].long()

        for m in range(2*M):
            if m < 2*M:
                node_feat = score[edges.long()[:, m], :]
                node_feat_rev = score[edges.long()[:, 2*M - m - 1], :]
                if m == 0: 
                    node_score = node_feat
                    node_score_rev = node_feat_rev
                else:
                    node_score = torch.cat([node_score, node_feat], dim = 1)
                    node_score_rev = torch.cat([node_score_rev, node_feat_rev], dim = 1)
        
        

        reward = self.relu(self.R_start(node_score))
        for R in self.R_hid:
            reward = self.relu(R(reward))
        reward = self.R_end(reward)
        
        penalty = self.relu(self.P_start(node_score))
        for P in self.P_hid:
            penalty = self.relu(P(penalty))
        penalty = self.P_end(penalty)

        reward_rev = self.relu(self.R_start(node_score_rev))
        for R in self.R_hid:
            reward_rev = self.relu(R(reward_rev))
        reward_rev = self.R_end(reward_rev)
        
        penalty_rev = self.relu(self.P_start(node_score_rev))
        for P in self.P_hid:
            penalty_rev = self.relu(P(penalty_rev))
        penalty_rev = self.P_end(penalty_rev)


        # Aggregation
        edges_0to1 = ((edges[:, -1] + 1) / 2)
        contribute_r = edges_0to1.view(-1, 1).repeat(1, reward.size()[1])*torch.sigmoid(reward)
        contribute_p = (1 - edges_0to1).view(-1, 1).repeat(1, penalty.size()[1])*torch.sigmoid(penalty)
        contribute = contribute_r - contribute_p


        contribute_r_rev = (1 - edges_0to1).view(-1, 1).repeat(1, reward_rev.size()[1])*torch.sigmoid(reward_rev)
        contribute_p_rev = edges_0to1.view(-1, 1).repeat(1, penalty_rev.size()[1])*torch.sigmoid(penalty_rev)
        contribute_rev = contribute_r_rev - contribute_p_rev

                 
        RP_aggregation = torch.zeros([N, 1])
        degree = torch.zeros([N, 1])
        contribute = contribute.t().contiguous().view(1, -1).t().contiguous()
        contribute_rev = contribute_rev.t().contiguous().view(1, -1).t().contiguous()
        for m in range(M):
            RP_aggregation.index_add_(0, edge_idx[:, m], contribute[int(m/(2*M)*len(contribute)):int((m+1)/(2*M)*len(contribute)), :])
            RP_aggregation.index_add_(0, edge_idx[:, M + m], contribute_rev[int(m/(2*M)*len(contribute_rev)):int((m+1)/(2*M)*len(contribute_rev)), :])
            degree.index_add_(0, edge_idx[:, m], torch.ones_like(contribute[int(m/(2*M)*len(contribute)):int((m+1)/(2*M)*len(contribute)), :]))
            degree.index_add_(0, edge_idx[:, M + m], torch.ones_like(contribute_rev[int(m/(2*M)*len(contribute_rev)):int((m+1)/(2*M)*len(contribute_rev)), :]))
        
        RP_aggregation = RP_aggregation.div(degree.max().div(self.norm_factor))
        
        # Normalization
        score = score + RP_aggregation
        score = score - score.mean()
        score = score.div(score.norm(dim = 0))
        
        return score