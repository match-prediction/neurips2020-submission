import os
import numpy as np
import scipy.sparse as sp
import torch
from torch.autograd import Variable
from random import shuffle
# import trueskill

# from trueskill import Rating, quality, rate 
import torch.nn.functional as F
import itertools
import math
import time
from scipy.stats import norm
from sklearn.metrics import roc_auc_score




####### MOVIE RANKING #######
path = os.path.join('data/movie/')
dataset = 'movie'
gt_test = np.genfromtxt("{}{}.GT_test".format(path, dataset), dtype=np.float32)

# Performance metric
##############################
def accuracy(output, labels):
    N = labels.size()
    output = output.view(N)
    labels = labels

    th = 0.5*torch.ones_like(output)
    win = torch.ones_like(output)
    loss = 0*torch.ones_like(output)

    # ##### MOVIE RANKING ######
    if len(output) == 5000:
        score = []
        for i in range(50):
            idx_i = range(100*i, 100*(i+1), 2)
            score.append(torch.sum(output[idx_i]).cpu().detach().numpy())
        
        score_sort_idx = np.argsort(score)
        gt_sort_idx = np.argsort(gt_test)
        dcg_3 = 0
        idcg_3 = 0
        dcg_5 = 0
        idcg_5 = 0
        dcg_7 = 0
        idcg_7 = 0
        dcg_10 = 0
        idcg_10 = 0
        dcg_25 = 0
        idcg_25 = 0
        dcg_50 = 0
        idcg_50 = 0
        for i in range(50):
            score_i = gt_test[score_sort_idx[len(score) - i - 1]]
            gt_i = gt_test[gt_sort_idx[len(score) - i - 1]]
            # print(score_i, gt_i)
            if i < 3:
                dcg_3 += (2**(score_i) - 1) / np.log2(i + 2)
                idcg_3 += (2**(gt_i) - 1) / np.log2(i + 2)
            if i < 5:
                dcg_5 += (2**(score_i) - 1) / np.log2(i + 2)
                idcg_5 += (2**(gt_i) - 1) / np.log2(i + 2)
            if i < 7:
                dcg_7 += (2**(score_i) - 1) / np.log2(i + 2)
                idcg_7 += (2**(gt_i) - 1) / np.log2(i + 2)
            if i < 10:
                dcg_10 += (2**(score_i) - 1) / np.log2(i + 2)
                idcg_10 += (2**(gt_i) - 1) / np.log2(i + 2)
            if i < 25:
                dcg_25 += (2**(score_i) - 1) / np.log2(i + 2)
                idcg_25 += (2**(gt_i) - 1) / np.log2(i + 2)
            
            dcg_50 += (2**(score_i) - 1) / np.log2(i + 2)
            idcg_50 += (2**(gt_i) - 1) / np.log2(i + 2)
        
        ndcg_3 = dcg_3 / idcg_3
        ndcg_5 = dcg_5 / idcg_5
        ndcg_7 = dcg_7 / idcg_7
        ndcg_10 = dcg_10 / idcg_10
        ndcg_25 = dcg_25 / idcg_25
        ndcg_50 = dcg_50 / idcg_50
        print("NDCG@3: {:0.4f}, NDCG@5: {:0.4f}, NDCG@7: {:0.4f}, NDCG@10: {:0.4f}, NDCG@25: {:0.4f}, NDCG@50: {:0.4f}".format(ndcg_3, ndcg_5, ndcg_7, ndcg_10, ndcg_25, ndcg_50))

        kt_dist = 0
        ct = 0
        for i in range(50):
            for j in range(i+1, 50):
                if (score[i] > score[j]) != (gt_test[i] > gt_test[j]):
                    kt_dist += 1
                ct += 1
        print("KT distance: {}".format(kt_dist / float(ct)))
    # ########################

    bin_output = torch.where(output > th, win, loss)
    bin_labels = torch.where(labels > th, win, loss)

    output_clone = output.clone()
    bin_labels_clone = bin_labels.clone()
    auc = roc_auc_score(bin_labels_clone.cpu().detach().numpy().astype(int), output_clone.cpu().detach().numpy())
    
    acc = torch.where(bin_output == bin_labels, win, loss)
    correct = acc.sum().cpu().numpy() / N
    correct = correct[0]
    expected_acc = torch.sum(bin_output*labels + (1-bin_output)*(1-labels)).cpu().detach().numpy() / N
    expected_acc = expected_acc[0]
    Hinge_loss = torch.sum(1 - output*labels - (1-output)*(1-labels)).cpu().detach().numpy() / N
    Hinge_loss = Hinge_loss[0]
    CE_loss = -torch.sum(labels*torch.log(output) + (1 - labels)*torch.log(1  - output)).cpu().detach().numpy() / N
    CE_loss = CE_loss[0]
    MSE_loss = torch.sum(torch.pow(labels - output, 2)).cpu().detach().numpy() / N
    MSE_loss = MSE_loss[0]
    
    return correct, expected_acc, auc, Hinge_loss, CE_loss, MSE_loss




    # BASELINE ALGORITHMS
def RankCentrality(train, dev, test, N): 
    edge_train_idx = train[:, :-1].long()
    edge_dev_idx = dev[:, :-1].long()
    edge_test_idx = test[:, :-1].long()
    comp_train = train[:, -1]
    comp_dev = dev[:, -1]
    comp_test = test[:, -1]

    # Generalized Rank Centrality
    M = int((train.shape[1] - 1/2)/2)
    n_iteration = 10000
    features = torch.ones(N, 1) / N
    diffuse_mat = torch.zeros(N*N)

    # # index = torch.range(1, N).long().view(-1, 1).repeat(1, N)
    for m1 in range(M):
        for m2 in range(M):
            diffuse_mat.index_add_(0, N*edge_train_idx[:, m1] + edge_train_idx[:, M + m2], comp_train)
            diffuse_mat.index_add_(0, N*edge_train_idx[:, M + m2] + edge_train_idx[:, m1], 1 - comp_train)
            diffuse_mat.index_add_(0, N*edge_train_idx[:, m1] + edge_train_idx[:, m2], -(1 - comp_train))
            diffuse_mat.index_add_(0, N*edge_train_idx[:, M + m2] + edge_train_idx[:, M + m1], -comp_train)

    diffuse_mat = diffuse_mat.view(N, N)
    diffuse_mat = diffuse_mat.div(-2*diffuse_mat.min())
    diffuse_mat += torch.eye(N)
    features = torch.ones([N, 1]).div(N)
    for iter_idx in range(n_iteration):
        features = torch.matmul(diffuse_mat, features)
    # features = torch,features + features.min()
    features = features.div(torch.max(features))


    # Match prediction
    group_A_score_train = features[edge_train_idx[:, :M]].sum(dim = 1)
    group_B_score_train = features[edge_train_idx[:, M:]].sum(dim = 1)
    group_A_score_dev = features[edge_dev_idx[:, :M]].sum(dim = 1)
    group_B_score_dev = features[edge_dev_idx[:, M:]].sum(dim = 1)
    group_A_score_test = features[edge_test_idx[:, :M]].sum(dim = 1)
    group_B_score_test = features[edge_test_idx[:, M:]].sum(dim = 1)
    
    win_train = torch.ones_like(group_A_score_train)
    loss_train = 1 - torch.ones_like(group_A_score_train)
    win_dev = torch.ones_like(group_A_score_dev)
    loss_dev = 1 - torch.ones_like(group_A_score_dev)
    win_test = torch.ones_like(group_A_score_test)
    loss_test = 1 - torch.ones_like(group_A_score_test)
    
    output_train = torch.where(group_A_score_train > group_B_score_train, win_train, loss_train)
    output_dev = torch.where(group_A_score_dev > group_B_score_dev, win_dev, loss_dev)
    prob_test = group_A_score_test.div(group_A_score_test + group_B_score_test)
    
    output_test = torch.where(prob_test > 0, prob_test, 1e-6*torch.ones_like(prob_test))
    output_test = torch.where(prob_test < 1, output_test, (1 - 1e-6)*torch.ones_like(prob_test))

    acc_test, exacc_test, auc, Hinge_loss, CE_loss, MSE_loss = accuracy(output_test, comp_test)

    performance = [acc_test, exacc_test, auc, Hinge_loss, CE_loss, MSE_loss]

    return features, performance
def Counting(train, dev, test, N):
    edge_train_idx = train[:, :-1].long()
    edge_dev_idx = dev[:, :-1].long()
    edge_test_idx = test[:, :-1].long()
    comp_train = train[:, -1]
    comp_dev = dev[:, -1]
    comp_test = test[:, -1]

    # Generalized Rank Centrality
    M = int((train.shape[1] - 1/2)/2)
    features = torch.zeros([N, 1])
    count = torch.zeros([N, 1]) 
    for m in range(M):
        features.index_add_(0, edge_train_idx[:, m], comp_train.view(-1, 1))
        features.index_add_(0, edge_train_idx[:, M + m], 1 - comp_train.view(-1, 1))

    features = features.div(torch.max(features))

    # Match prediction
    group_A_score_train = features[edge_train_idx[:, :M]].sum(dim = 1)
    group_B_score_train = features[edge_train_idx[:, M:]].sum(dim = 1)
    group_A_score_dev = features[edge_dev_idx[:, :M]].sum(dim = 1)
    group_B_score_dev = features[edge_dev_idx[:, M:]].sum(dim = 1)
    group_A_score_test = features[edge_test_idx[:, :M]].sum(dim = 1)
    group_B_score_test = features[edge_test_idx[:, M:]].sum(dim = 1)
    
    win_train = torch.ones_like(group_A_score_train)
    loss_train = 1 - torch.ones_like(group_A_score_train)
    win_dev = torch.ones_like(group_A_score_dev)
    loss_dev = 1 - torch.ones_like(group_A_score_dev)
    win_test = torch.ones_like(group_A_score_test)
    loss_test = 1 - torch.ones_like(group_A_score_test)
    
    output_train = torch.where(group_A_score_train > group_B_score_train, win_train, loss_train)
    output_dev = torch.where(group_A_score_dev > group_B_score_dev, win_dev, loss_dev)
    prob_test = group_A_score_test.div(group_A_score_test + group_B_score_test)
    output_test = torch.where(prob_test > 0, prob_test, 1e-4*torch.ones_like(prob_test))
    output_test = torch.where(prob_test < 1, output_test, (1 - 1e-4)*torch.ones_like(prob_test))

    acc_test, exacc_test, auc, Hinge_loss, CE_loss, MSE_loss = accuracy(output_test, comp_test)

    performance = [acc_test, exacc_test, auc, Hinge_loss, CE_loss, MSE_loss]

    return features, performance
def MM_sum(train, dev, test, N):
    edge_train_idx = train[:, :-1].long()
    edge_dev_idx = dev[:, :-1].long()
    edge_test_idx = test[:, :-1].long()
    comp_train = train[:, -1]
    comp_dev = dev[:, -1]
    comp_test = test[:, -1]
    
    # Generalized Rank Centrality
    M = int((train.shape[1] - 1/2)/2)
    n_iteration = 1000
    features = torch.ones(N).div(N)
    for iter_idx in range(n_iteration):
        prev_features = features
        update_feat_nume = torch.zeros([N, 1])
        update_feat_demo = torch.zeros([N, 1])
        for m in range(M):
            update_feat_nume.index_add_(0, edge_train_idx[:, m], comp_train.view(-1, 1).div(features[edge_train_idx[:, :M]].sum(dim = 1).view(-1, 1)))
            update_feat_nume.index_add_(0, edge_train_idx[:, M + m], (1 - comp_train.view(-1, 1)).div(features[edge_train_idx[:, M:2*M]].sum(dim = 1).view(-1, 1)))
            update_feat_demo.index_add_(0, edge_train_idx[:, m], torch.ones_like(comp_train.view(-1, 1)).div(features[edge_train_idx].sum(dim = 1).view(-1, 1)))
            update_feat_demo.index_add_(0, edge_train_idx[:, M + m], torch.ones_like(comp_train.view(-1, 1)).div(features[edge_train_idx].sum(dim = 1).view(-1, 1)))

        features = features.view(-1, 1) 
        features = features*(update_feat_nume+1e-8*torch.ones_like(update_feat_nume)).div(update_feat_demo+1e-8*torch.ones_like(update_feat_demo)) 
        features = features + 1e-17*torch.ones_like(features)
        features = features.div(features.sum())
        features = features.squeeze()
    features = features.div(torch.max(features)).view(features.shape[0], 1)
    
    # Match prediction
    group_A_score_train = features[edge_train_idx[:, :M]].sum(dim = 1)
    group_B_score_train = features[edge_train_idx[:, M:]].sum(dim = 1)
    group_A_score_dev = features[edge_dev_idx[:, :M]].sum(dim = 1)
    group_B_score_dev = features[edge_dev_idx[:, M:]].sum(dim = 1)
    group_A_score_test = features[edge_test_idx[:, :M]].sum(dim = 1)
    group_B_score_test = features[edge_test_idx[:, M:]].sum(dim = 1)
    
    win_train = torch.ones_like(group_A_score_train)
    loss_train = 1 - torch.ones_like(group_A_score_train)
    win_dev = torch.ones_like(group_A_score_dev)
    loss_dev = 1 - torch.ones_like(group_A_score_dev)
    win_test = torch.ones_like(group_A_score_test)
    loss_test = 1 - torch.ones_like(group_A_score_test)
    
    output_train = torch.where(group_A_score_train > group_B_score_train, win_train, loss_train)
    output_dev = torch.where(group_A_score_dev > group_B_score_dev, win_dev, loss_dev)
    prob_test = group_A_score_test.div(group_A_score_test + group_B_score_test)
    output_test = torch.where(prob_test > 0, prob_test, 1e-6*torch.ones_like(prob_test))
    output_test = torch.where(prob_test < 1, output_test, (1 - 1e-6)*torch.ones_like(prob_test))
    
    acc_test, exacc_test, auc, Hinge_loss, CE_loss, MSE_loss = accuracy(output_test, comp_test)

    performance = [acc_test, exacc_test, auc, Hinge_loss, CE_loss, MSE_loss]

    return features, performance
def MM_prod(train, dev, test, N):
    edge_train_idx = train[:, :-1].long()
    edge_dev_idx = dev[:, :-1].long()
    edge_test_idx = test[:, :-1].long()
    comp_train = train[:, -1]
    comp_dev = dev[:, -1]
    comp_test = test[:, -1]
    
    # Generalized Rank Centrality
    M = int((train.shape[1] - 1/2)/2)
    n_iteration = 1000
    features = torch.ones(N).div(N)

    
    for iter_idx in range(n_iteration): 
        update_feat_nume = torch.zeros([N, 1])
        update_feat_demo = torch.zeros([N, 1])
        for m in range(M):
            update_feat_nume.index_add_(0, edge_train_idx[:, m], comp_train.view(-1, 1))
            update_feat_nume.index_add_(0, edge_train_idx[:, M + m], (1 - comp_train).view(-1, 1))
            update_feat_demo.index_add_(0, edge_train_idx[:, m], features[edge_train_idx[:, :M]].prod(dim = 1).div(features[edge_train_idx[:, :M]].prod(dim = 1) + features[edge_train_idx[:, M:2*M]].prod(dim = 1)).view(-1, 1))
            update_feat_demo.index_add_(0, edge_train_idx[:, M + m], features[edge_train_idx[:, M:2*M]].prod(dim = 1).div(features[edge_train_idx[:, :M]].prod(dim = 1) + features[edge_train_idx[:, M:2*M]].prod(dim = 1)).view(-1, 1))

        features = features.view(-1, 1)*(update_feat_nume+1e-8*torch.ones_like(update_feat_nume)).div((update_feat_demo+1e-8*torch.ones_like(update_feat_demo)))
    features = features.div(torch.max(features)).view(features.shape[0], 1)
    # features = features.div(torch.min(features)).view(features.shape[0], 1)
    # features = features / 1e31

    # Match prediction
    group_A_score_train = features[edge_train_idx[:, :M]].prod(dim = 1)
    group_B_score_train = features[edge_train_idx[:, M:]].prod(dim = 1)
    group_A_score_dev = features[edge_dev_idx[:, :M]].prod(dim = 1)
    group_B_score_dev = features[edge_dev_idx[:, M:]].prod(dim = 1)
    group_A_score_test = features[edge_test_idx[:, :M]].prod(dim = 1)
    group_B_score_test = features[edge_test_idx[:, M:]].prod(dim = 1)
    
    win_train = torch.ones_like(group_A_score_train)
    loss_train = 1 - torch.ones_like(group_A_score_train)
    win_dev = torch.ones_like(group_A_score_dev)
    loss_dev = 1 - torch.ones_like(group_A_score_dev)
    win_test = torch.ones_like(group_A_score_test)
    loss_test = 1 - torch.ones_like(group_A_score_test)
    
    output_train = torch.where(group_A_score_train > group_B_score_train, win_train, loss_train)
    output_dev = torch.where(group_A_score_dev > group_B_score_dev, win_dev, loss_dev)
    # group_A_score_test = group_A_score_test + 1e-6
    # group_B_score_test = group_B_score_test + 1e-6
    prob_test = group_A_score_test.div(group_A_score_test + group_B_score_test)
    output_test = torch.where(prob_test > 0, prob_test, 1e-6*torch.ones_like(prob_test))
    output_test = torch.where(prob_test < 1, output_test, (1 - 1e-6)*torch.ones_like(prob_test))
    acc_test, exacc_test, auc, Hinge_loss, CE_loss, MSE_loss = accuracy(output_test, comp_test)

    performance = [acc_test, exacc_test, auc, Hinge_loss, CE_loss, MSE_loss]

    return features, performance
def SGD_HOI(train, dev, test, N):
    edge_train_idx = train[:, :-1].long()
    edge_dev_idx = dev[:, :-1].long()
    edge_test_idx = test[:, :-1].long()
    comp_train = train[:, -1]
    comp_dev = dev[:, -1]
    comp_test = test[:, -1]

    performance = {}
    M = int((train.shape[1] - 1/2)/2)
    n_iteration = 300
    n_iteration_sub = 10
    
    # n_iteration = 0
    # n_iteration_sub = 0
    
    reg_rate_score = 0.8
    reg_rate_feat = 0.8 
    
    # Hyperparameter
    feat_dim_range = [1, 7]
    # feat_dim_range = [1, 3, 5, 7, 9]  
    
    degree = torch.zeros([N, 1])
    for m in range(2*M):
        degree.index_add_(0, edge_train_idx[:, m], torch.ones_like(edge_train_idx[:, m]).float())
    
    degree[degree == 0] = degree.mean() 
    degree = degree.div(degree.min())
    
    for feat_dim in feat_dim_range:
        scores = torch.randn([N, 1])/100
        features = torch.randn([N, feat_dim])/100
        features = features - features.sum().div(N*feat_dim)
        scores = scores - scores.sum().div(N*feat_dim)
        prev_features = torch.zeros([N, feat_dim])
        prev_scores = torch.zeros([N, 1])
        for iter_idx in range(n_iteration):
            for sub_iter_idx in range(n_iteration_sub):
                prev_scores[:, :] = scores
                prev_features[:, :] = features

                temp_score_A = prev_scores[edge_train_idx[:, :M]].sum(dim = 1).squeeze()
                temp_score_B = prev_scores[edge_train_idx[:, M:]].sum(dim = 1).squeeze()
                temp_score_A += prev_features[edge_train_idx[:, :M]].matmul(prev_features[edge_train_idx[:, :M]].transpose(1, 2)).sum(dim = 2).sum(dim = 1) 
                temp_score_B += prev_features[edge_train_idx[:, M:]].matmul(prev_features[edge_train_idx[:, M:]].transpose(1, 2)).sum(dim = 2).sum(dim = 1)  
                temp_score_gap = temp_score_A - temp_score_B 
                score_gap = torch.where(temp_score_gap > 100, temp_score_gap, 100*torch.ones_like(temp_score_gap))
                score_gap = torch.where(temp_score_gap < -100, score_gap, -100*torch.ones_like(temp_score_gap))
                A_wins_B_est = torch.sigmoid(score_gap)
                B_wins_A_est = 1 - A_wins_B_est

                feat_sum_A = features[edge_train_idx[:, :M]].sum(dim = 1)
                feat_sum_B = features[edge_train_idx[:, M:]].sum(dim = 1)
                
                update_feat_A = (comp_train*B_wins_A_est - (1 - comp_train)*A_wins_B_est).view(-1, 1).repeat(1, feat_dim)*feat_sum_A
                update_feat_B = -(comp_train*B_wins_A_est - (1 - comp_train)*A_wins_B_est).view(-1, 1).repeat(1, feat_dim)*feat_sum_B  
                update_score_A = (comp_train*B_wins_A_est - (1 - comp_train)*A_wins_B_est).view(-1, 1)
                update_score_B = -(comp_train*B_wins_A_est - (1 - comp_train)*A_wins_B_est).view(-1, 1)

                update_score = torch.zeros([N, 1])
                for m in range(M):
                    update_score.index_add_(0, edge_train_idx[:, m], (update_score_A).div(M*update_score_A.size()[0]/N))
                    update_score.index_add_(0, edge_train_idx[:, M + m], (update_score_B).div(M*update_score_B.size()[0]/N))

                scores = (1 - reg_rate_score)*prev_scores + update_score
                
            prev_scores[:, :] = scores
            prev_features[:, :] = features

            temp_score_A = prev_scores[edge_train_idx[:, :M]].sum(dim = 1).squeeze()
            temp_score_B = prev_scores[edge_train_idx[:, M:]].sum(dim = 1).squeeze()
            temp_score_A += prev_features[edge_train_idx[:, :M]].matmul(prev_features[edge_train_idx[:, :M]].transpose(1, 2)).sum(dim = 2).sum(dim = 1) 
            temp_score_B += prev_features[edge_train_idx[:, M:]].matmul(prev_features[edge_train_idx[:, M:]].transpose(1, 2)).sum(dim = 2).sum(dim = 1)  
            
            temp_score_gap = temp_score_A - temp_score_B 
            score_gap = torch.where(temp_score_gap > 100, temp_score_gap, 100*torch.ones_like(temp_score_gap))
            score_gap = torch.where(temp_score_gap < -100, score_gap, -100*torch.ones_like(temp_score_gap))
            A_wins_B_est = torch.sigmoid(score_gap)
            B_wins_A_est = 1 - A_wins_B_est

            feat_sum_A = features[edge_train_idx[:, :M]].sum(dim = 1)
            feat_sum_B = features[edge_train_idx[:, M:]].sum(dim = 1)
            
            update_feat_A = (comp_train*B_wins_A_est - (1 - comp_train)*A_wins_B_est).view(-1, 1).repeat(1, feat_dim)*feat_sum_A
            update_feat_B = -(comp_train*B_wins_A_est - (1 - comp_train)*A_wins_B_est).view(-1, 1).repeat(1, feat_dim)*feat_sum_B  
            update_score_A = (comp_train*B_wins_A_est - (1 - comp_train)*A_wins_B_est).view(-1, 1)
            update_score_B = -(comp_train*B_wins_A_est - (1 - comp_train)*A_wins_B_est).view(-1, 1)

            update_feat = torch.zeros([N, feat_dim])    
            for m in range(M):
                update_feat.index_add_(0, edge_train_idx[:, m], (update_feat_A).div(M*update_feat_A.size()[0]/N))
                update_feat.index_add_(0, edge_train_idx[:, M + m], (update_feat_B).div(M*update_feat_B.size()[0]/N))
            features = (1 - torch.ones_like(degree).div(degree)*reg_rate_feat)*prev_features + 0.01*update_feat
            
            features[features != features] = 0
            prev_features[prev_features != prev_features] = 0
        
        # # Normalize
        # features = features.div(features.max())
        # scores = scores.div(scores.max())
            
        # Match prediction
        group_A_score_train = scores[edge_train_idx[:, :M]].sum(dim = 1).squeeze() + features[edge_train_idx[:, :M]].matmul(features[edge_train_idx[:, :M]].transpose(1, 2)).sum(dim = 2).sum(dim = 1)
        group_B_score_train = scores[edge_train_idx[:, M:]].sum(dim = 1).squeeze() + features[edge_train_idx[:, M:]].matmul(features[edge_train_idx[:, M:]].transpose(1, 2)).sum(dim = 2).sum(dim = 1)
        group_A_score_dev = scores[edge_dev_idx[:, :M]].sum(dim = 1).squeeze() + features[edge_dev_idx[:, :M]].matmul(features[edge_dev_idx[:, :M]].transpose(1, 2)).sum(dim = 2).sum(dim = 1)
        group_B_score_dev = scores[edge_dev_idx[:, M:]].sum(dim = 1).squeeze() + features[edge_dev_idx[:, M:]].matmul(features[edge_dev_idx[:, M:]].transpose(1, 2)).sum(dim = 2).sum(dim = 1)
        group_A_score_test = scores[edge_test_idx[:, :M]].sum(dim = 1).squeeze() + features[edge_test_idx[:, :M]].matmul(features[edge_test_idx[:, :M]].transpose(1, 2)).sum(dim = 2).sum(dim = 1)
        group_B_score_test = scores[edge_test_idx[:, M:]].sum(dim = 1).squeeze() + features[edge_test_idx[:, M:]].matmul(features[edge_test_idx[:, M:]].transpose(1, 2)).sum(dim = 2).sum(dim = 1)
        
        win_train = torch.ones_like(group_A_score_train)
        loss_train = 1 - torch.ones_like(group_A_score_train)
        win_dev = torch.ones_like(group_A_score_dev)
        loss_dev = 1 - torch.ones_like(group_A_score_dev)
        win_test = torch.ones_like(group_A_score_test)
        loss_test = 1 - torch.ones_like(group_A_score_test)
        
        prob_dev = torch.sigmoid(group_A_score_dev - group_B_score_dev)
        output_dev = torch.where(prob_dev > 1e-6, prob_dev, 1e-6*torch.ones_like(prob_dev))
        output_dev = torch.where(prob_dev < (1 - 1e-6), output_dev, (1 - 1e-6)*torch.ones_like(prob_dev))

        prob_test = torch.sigmoid(group_A_score_test - group_B_score_test)
        output_test = torch.where(prob_test > 1e-6, prob_test, 1e-6*torch.ones_like(prob_test))
        output_test = torch.where(prob_test < (1 - 1e-6), output_test, (1 - 1e-6)*torch.ones_like(prob_test))
        
        acc_dev, exacc_dev, auc_dev, Hinge_loss_dev, CE_loss_dev, MSE_loss_dev = accuracy(output_dev, comp_dev)
        acc_test, exacc_test, auc_test, Hinge_loss, CE_loss, MSE_loss = accuracy(output_test, comp_test)

        performance[feat_dim] = [acc_dev, exacc_dev, auc_dev, Hinge_loss_dev, CE_loss_dev, MSE_loss_dev, acc_test, exacc_test, auc_test, Hinge_loss, CE_loss, MSE_loss]

    
    return features, performance
def TrueSkill(train, dev, test, N):
    edge_train_idx = train[:, :-1].long()
    edge_dev_idx = dev[:, :-1].long()
    edge_test_idx = test[:, :-1].long()
    comp_train = train[:, -1]
    comp_dev = dev[:, -1]
    comp_test = test[:, -1]

    performance = {}
    performance['dev'] = {}
    performance['test'] = {}    
    
    M = int((train.shape[1] - 1/2)/2)
    features_TS = {}
    for i in range(N):
        features_TS[i] = Rating()
    
    # Hyperparameter
    BETA_range = [5]
    n_iteration = 1
    
    for iter_idx in range(n_iteration):
        rand_split = np.random.permutation(comp_train.shape[0])
        edge_train_idx = edge_train_idx[rand_split]
        comp_train = comp_train[rand_split]
        for sample_idx in range(comp_train.size()[0]):
            group_A = []
            group_B = []
            for m in range(M):  
                group_A.append(features_TS[int(edge_train_idx[sample_idx, m].cpu().numpy())])
                group_B.append(features_TS[int(edge_train_idx[sample_idx, M + m].cpu().numpy())])
            (group_A), (group_B) = rate([group_A, group_B], ranks = [1 - comp_train[sample_idx], comp_train[sample_idx]])
            for m in range(M):  
                features_TS[int(edge_train_idx[sample_idx, m].cpu().numpy())] = group_A[m]
                features_TS[int(edge_train_idx[sample_idx, M + m].cpu().numpy())] = group_B[m]

        BETA = 9
        prob_dev = torch.zeros([edge_dev_idx.shape[0], 1])
        for dev_idx in range(edge_dev_idx.shape[0]):
            # print(dev_idx)
            team1 = []
            team2 = []
            for m in range(M):
                team1.append(features_TS[int(edge_dev_idx[dev_idx, m].cpu().numpy())])
                team2.append(features_TS[int(edge_dev_idx[dev_idx, 2*M - m - 1].cpu().numpy())])
            delta_mu = sum(r.mu for r in team1) - sum(r.mu for r in team2)
            sum_sigma = sum(r.sigma ** 2 for r in itertools.chain(team1, team2))
            # print(team1[0].sigma)
            size = len(team1) + len(team2)
            denom = math.sqrt(size * (BETA * BETA) + sum_sigma)
            prob_dev[dev_idx] = norm.cdf(delta_mu / denom)

        prob_dev = prob_dev.view(-1, 1)

        output_dev = torch.where(prob_dev > 0, prob_dev, 1e-6*torch.ones_like(prob_dev))
        output_dev = torch.where(prob_dev < 1, output_dev, (1 - 1e-6)*torch.ones_like(prob_dev))
        acc_dev, exacc_dev, auc_dev, Hinge_loss_dev, CE_loss_dev, MSE_loss_dev = accuracy(output_dev, comp_dev)
        performance['dev'][iter_idx] = [acc_dev, exacc_dev, auc_dev,  Hinge_loss_dev, CE_loss_dev, MSE_loss_dev]


    features= torch.zeros([N, 1])
    for i in range(N):
        features[i] = features_TS[i].mu

    for BETA in BETA_range:
        prob_test = torch.zeros([edge_test_idx.size()[0], 1])
        for test_idx in range(edge_test_idx.size()[0]):
            team1 = []
            team2 = []
            for m in range(M):
                team1.append(features_TS[int(edge_test_idx[test_idx, m].cpu().numpy())])
                team2.append(features_TS[int(edge_test_idx[test_idx, 2*M - m - 1].cpu().numpy())])
            delta_mu = sum(r.mu for r in team1) - sum(r.mu for r in team2)
            sum_sigma = sum(r.sigma ** 2 for r in itertools.chain(team1, team2))
            size = len(team1) + len(team2)
            denom = math.sqrt(size * (BETA * BETA) + sum_sigma)
            prob_test[test_idx] = norm.cdf(delta_mu / denom)
            
        prob_test = prob_test.view(-1, 1)

        output_test = torch.where(prob_test > 0, prob_test, 1e-6*torch.ones_like(prob_test))
        output_test = torch.where(prob_test < 1, output_test, (1 - 1e-6)*torch.ones_like(prob_test))
        acc_test, exacc_test, auc_test, Hinge_loss, CE_loss, MSE_loss = accuracy(output_test, comp_test)

        performance['test'][BETA] = [acc_test, exacc_test, auc_test, Hinge_loss, CE_loss, MSE_loss]

    return features, performance
def Random_estimate(train, dev, test, N):
    edge_train_idx = train[:, :-1].long()
    edge_dev_idx = dev[:, :-1].long()
    edge_test_idx = test[:, :-1].long()
    comp_train = train[:, -1]
    comp_dev = dev[:, -1]
    comp_test = test[:, -1]

    M = int((train.shape[1] - 1/2)/2)
    features = torch.ones(N).div(N)
    # Match prediction
    group_A_score_train = features[edge_train_idx[:, :M]].prod(dim = 1)
    group_B_score_train = features[edge_train_idx[:, M:]].prod(dim = 1)
    group_A_score_dev = features[edge_dev_idx[:, :M]].prod(dim = 1)
    group_B_score_dev = features[edge_dev_idx[:, M:]].prod(dim = 1)
    group_A_score_test = features[edge_test_idx[:, :M]].prod(dim = 1)
    group_B_score_test = features[edge_test_idx[:, M:]].prod(dim = 1)
    
    win_train = torch.ones_like(group_A_score_train)
    loss_train = 1 - torch.ones_like(group_A_score_train)
    win_dev = torch.ones_like(group_A_score_dev)
    loss_dev = 1 - torch.ones_like(group_A_score_dev)
    win_test = torch.ones_like(group_A_score_test)
    loss_test = 1 - torch.ones_like(group_A_score_test)
    
    output_train = torch.where(group_A_score_train > group_B_score_train, win_train, loss_train)
    output_dev = torch.where(group_A_score_dev > group_B_score_dev, win_dev, loss_dev)
    prob_test = group_A_score_test.div(group_A_score_test + group_B_score_test)
    output_test = torch.where(prob_test > 0, prob_test, 1e-6*torch.ones_like(prob_test))
    output_test = torch.where(prob_test < 1, output_test, (1 - 1e-6)*torch.ones_like(prob_test))
    acc_test, exacc_test, auc_test, Hinge_loss, CE_loss, MSE_loss = accuracy(output_test, comp_test)

    performance = [acc_test, exacc_test, auc_test, Hinge_loss, CE_loss, MSE_loss]

    return features, performance


def load_data(path, dataset, args):
    
    idx_features_labels = np.genfromtxt("{}{}.nodes".format(path, dataset), dtype=np.dtype(str))
   
    idx = np.array(idx_features_labels[:, 0], dtype=np.float32)
    N = idx.shape[0]
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.genfromtxt("{}{}.edges".format(path, dataset), dtype=np.float32)


    # D_obs -> D_train:D_dev = train_rate:dev_rate
    if dataset == "BTL_sum":
        dev_rate = 0.1
        # dev_rate = 0.65 - args.train_rate
        split = [args.train_rate, dev_rate] 
    elif dataset == "BTL_prod":
        dev_rate = 0.1
        split = [args.train_rate, dev_rate]
    elif dataset == "BTL_HOI":
        dev_rate = 0.1
        split = [args.train_rate, dev_rate]
    elif dataset == "TrueSkill":
        dev_rate = 0.1
        split = [args.train_rate, dev_rate]
    elif dataset == "GIFGIF":
        dev_rate = 0.1
        split = [args.train_rate, dev_rate]
    elif dataset == "HOTS":
        # dev_rate = 0.1
        # dev_rate = 0.02
        dev_rate = 0
        split = [args.train_rate, dev_rate]
    elif dataset == "DOTA":
        dev_rate = 0.1
        split = [args.train_rate, dev_rate]
    elif dataset == "LoL":
        dev_rate = 0
        split = [args.train_rate, dev_rate]
    elif dataset == "LoL_pro":
        dev_rate = 0
        split = [args.train_rate, dev_rate]
    elif dataset == "movie":
        dev_rate = 0.01
        split = [args.train_rate, dev_rate]

    features = torch.randn([N, 1])
    features = features.div(features.norm(dim = 0))
    

    np.random.seed(args.seed)
    ############## HOTS setting ###############
    if dataset in ["HOTS", "LoL", "LoL_pro"]:
        edges_test_full = np.genfromtxt("{}{}.edges_test".format(path, dataset), dtype=np.float32)
        rand_test_split = np.random.permutation(edges_test_full.shape[0])
        # 3
        edges_dev = edges_test_full[rand_test_split[:int(1*len(edges_test_full)/4)], :]
        edges_test = edges_test_full[rand_test_split[int(1*len(edges_test_full)/4):], :]
        
        edges_train = edges[:int(split[0]*len(edges)), :]
        # edges_dev = edges[int(split[0]*len(edges))+1:int((split[0] + split[1])*len(edges)), :]
        edges_train_full = np.concatenate((edges_train, edges_dev), axis = 0)
    #############################################
    else:
        edges_train_full = edges[:int((split[0] + split[1])*len(edges)), :]
        edges_train = edges_train_full[:int(split[0]*len(edges)), :]
        edges_dev = edges_train_full[int(split[0]*len(edges))+1:, :]
        edges_test = np.genfromtxt("{}{}.edges_test".format(path, dataset), dtype=np.float32)
    

    
    edges_train_full = torch.cuda.FloatTensor(np.array(edges_train_full))
    edges_train = torch.cuda.FloatTensor(np.array(edges_train))
    edges_dev = torch.cuda.FloatTensor(np.array(edges_dev))
    edges_test = torch.cuda.FloatTensor(np.array(edges_test))

    # torch.device("cpu")
    # edges_train_full = torch.FloatTensor(np.array(edges_train_full))
    # edges_train = torch.FloatTensor(np.array(edges_train))
    # edges_dev = torch.FloatTensor(np.array(edges_dev))
    # edges_test = torch.FloatTensor(np.array(edges_test))


    ##########################################
    # Run baseline algorithms
    if args.baseline_flag == 1:
        # edges_train_full = torch.cat([edges_train, edges_dev], dim = 0)   
        
        t_trueskill = time.time()
        # TrueSkill_feature, TrueSkill_performance = TrueSkill(edges_train_full, edges_dev, edges_test, N)
        TrueSkill_feature, TrueSkill_performance = TrueSkill(edges_train, edges_dev, edges_test, N)
        print("Total time elapsed (TrueSkill): {:.5f}s".format(time.time() - t_trueskill))
        
        t_hoi = time.time()
        # HOI_feature, HOI_performance = SGD_HOI(edges_train_full, edges_dev, edges_test, N)
        HOI_feature, HOI_performance = SGD_HOI(edges_train, edges_dev, edges_test, N)
        print("Total time elapsed (HOI): {:.5f}s".format(time.time() - t_hoi))
        # print(HOI_performance)

        t_prod = time.time()       
        # MM_prod_feature, MM_prod_performance = MM_prod(edges_train_full, edges_dev, edges_test, N)
        MM_prod_feature, MM_prod_performance = MM_prod(edges_train, edges_dev, edges_test, N)
        print("Total time elapsed (prod): {:.5f}s".format(time.time() - t_prod))
        # print(MM_prod_performance)

        t_sum = time.time()
        # MM_feature, MM_performance = MM_sum(edges_train_full, edges_dev, edges_test, N)
        MM_feature, MM_performance = MM_sum(edges_train, edges_dev, edges_test, N)
        print("Total time elapsed (sum): {:.5f}s".format(time.time() - t_sum))
        # print(MM_performance)

        t_rc = time.time()
        # RC_feature, RC_performance = RankCentrality(edges_train_full, edges_dev, edges_test, N)
        RC_feature, RC_performance = RankCentrality(edges_train, edges_dev, edges_test, N)
        print("Total time elapsed (RC): {:.5f}s".format(time.time() - t_rc))
        # print(RC_performance)

        CT_feature, CT_performance = Counting(edges_train_full, edges_dev, edges_test, N)
        Random_estimate_feature, Random_estimate_performance = Random_estimate(edges_train_full, edges_dev, edges_test, N)
        acc_test, exacc_test, auc_test, Hinge_loss, CE_loss, MSE_loss = accuracy(edges_test[:, -1],  edges_test[:, -1])
        GT_performance = [acc_test, exacc_test, auc_test, Hinge_loss, CE_loss, MSE_loss]
        
        baseline_test_result = {}

        baseline_test_result["TrueSkill"] = TrueSkill_performance
        baseline_test_result["HOI"] = HOI_performance
        baseline_test_result["MM_prod"] = MM_prod_performance        
        baseline_test_result["MM"] = MM_performance
        baseline_test_result["RC"] = RC_performance
        baseline_test_result["CT"] = CT_performance
        baseline_test_result["Random_estimate"] = Random_estimate_performance
        baseline_test_result["GT"] = GT_performance

        np.save(os.path.join('result', args.dataset + '_expnum=' + str(args.exp_num) + '_seed=' + str(args.seed) + '_train_rate=' + str(args.train_rate) + '_baseline'), baseline_test_result)
        # print(baseline_test_result)
    ##########################################


    return N, edges_train, edges_dev, edges_test, features



