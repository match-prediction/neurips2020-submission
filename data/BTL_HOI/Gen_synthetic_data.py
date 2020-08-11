import numpy as np
import os
import argparse
from random import shuffle
from scipy.special import comb

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
args = parser.parse_args()


def Gen_GroundTruth(n_items, dim):
    GroundTruth_score = np.random.normal(0, 2, (n_items, 1))
    GroundTruth_feat = np.random.normal(0, 1, (n_items, dim))

    return GroundTruth_score, GroundTruth_feat


def Gen_data_under_BTL(score, feat, p_obs, l_rep, M):
    n_items = len(score[:, 0])
    num_dataset = int(comb(n_items, 2 * M) * comb(2 * M, M) * p_obs)
    pairwise_data = []
    pairwise_data_test = []

    print(num_dataset)
    popular_group = np.random.permutation(range(n_items))[:int(n_items / 3)]
    for i in range(num_dataset):
        if i % 1000 == 0:
            print(str(i) + '/' + str(num_dataset))
        group = np.random.choice(range(n_items), 2 * M, replace=False)
        group_A_score = 0
        group_B_score = 0
        for i in range(M):
            for j in range(M):
                group_A_score += np.inner(feat[group[i]], feat[group[j]])
                group_B_score += np.inner(feat[group[M + i]], feat[group[M + j]])
        p_comp = 1 / (1 + np.exp(group_B_score - group_A_score))
        for rep_comp in range(l_rep):
            A_wins_B = int(np.random.binomial(1, p_comp))
            pairwise_datum1 = []
            pairwise_datum2 = []
            for tmp in range(2 * M):
                pairwise_datum1.append(group[tmp])
                pairwise_datum2.append(group[2 * M - tmp - 1])
            pairwise_datum1.append(A_wins_B)
            pairwise_datum2.append(1 - A_wins_B)
            pairwise_data.append(pairwise_datum1)
            pairwise_data.append(pairwise_datum2)

    num_test_dataset = 10000
    for i in range(num_test_dataset):
        group = np.random.choice(range(n_items), 2 * M, replace=False)
        group_A_score = 0
        group_B_score = 0
        for i in range(M):
            for j in range(M):
                group_A_score += np.inner(feat[group[i]], feat[group[j]])
                group_B_score += np.inner(feat[group[M + i]], feat[group[M + j]])
        p_comp = 1 / (1 + np.exp(group_B_score - group_A_score))

        A_wins_B = p_comp
        pairwise_datum1 = []
        pairwise_datum2 = []
        for tmp in range(2 * M):
            pairwise_datum1.append(group[tmp])
            pairwise_datum2.append(group[2 * M - tmp - 1])
        pairwise_datum1.append(A_wins_B)
        pairwise_datum2.append(1 - A_wins_B)
        pairwise_data_test.append(pairwise_datum1)
        pairwise_data_test.append(pairwise_datum2)

    np.random.shuffle(pairwise_data_test)
    return pairwise_data, pairwise_data_test


def Gen_feature(n_items, dataset):
    feature = []
    feature.append(range(n_items))
    feature.append(range(n_items))
    feature.append(range(n_items))
    feature = list(map(list, zip(*feature)))
    return feature


def save_data(pairwise_data):
    pass


def main(n_items, dim, score_range, p_obs, l_rep, M):
    score, feat = Gen_GroundTruth(n_items, dim)
    pairwise_data, pairwise_data_test = Gen_data_under_BTL(score, feat, p_obs, l_rep, M)
    feature = Gen_feature(n_items, pairwise_data)
    # GroundTruth = score
    GroundTruth = feat
    np.savetxt(os.path.join('data/BTL_HOI', 'BTL_HOI.GT'), GroundTruth)
    np.savetxt(os.path.join('data/BTL_HOI', 'BTL_HOI.edges'), pairwise_data)
    np.savetxt(os.path.join('data/BTL_HOI', 'BTL_HOI.edges_test'), pairwise_data_test)
    np.savetxt(os.path.join('data/BTL_HOI', 'BTL_HOI.nodes'), feature)


if __name__ == "__main__":
    n_items = 300
    dim = 7
    M = 5
    score_range = 1e+5
    p_obs = 40 * (np.log(n_items) / (comb(n_items - 1, 2 * M - 1) * comb(2 * M - 1, M - 1)))
    l_rep = 10

    main(n_items, dim, score_range, p_obs, l_rep, M)
