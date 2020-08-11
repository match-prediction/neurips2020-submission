# from __future__ import print_function
# import torch

import numpy as np
import os
import argparse
from random import shuffle
from scipy.special import comb
import trueskill
from trueskill import Rating, quality, rate 
import matplotlib.pyplot as plt
import itertools
import math


parser = argparse.ArgumentParser()
args = parser.parse_args()

# np.random.seed(0)

def win_probability(team1, team2):
    delta_mu = sum(r.mu for r in team1) - sum(r.mu for r in team2)
    sum_sigma = sum(r.sigma ** 2 for r in itertools.chain(team1, team2))
    size = len(team1) + len(team2)
    BETA = 9
    denom = math.sqrt(size * (BETA * BETA) + sum_sigma)
    ts = trueskill.global_env()

    return ts.cdf(delta_mu / denom)


def Gen_GroundTruth(n_items, score_range):
	GroundTruth = {}
	GroundTruth_mu = []
	Highest_score = 40
	Lowest_score = 10
	score = np.zeros(n_items) 
	for i in range(n_items):
		score[i] = np.random.uniform(Lowest_score, Highest_score)
	score = sorted(score, reverse = True)
	for i in range(n_items):
		GroundTruth[i] = Rating(score[i])
	for i in range(n_items):
		GroundTruth_mu.append(GroundTruth[i].mu)

	return GroundTruth, GroundTruth_mu

def Gen_data_under_BTL(GroundTruth, p_obs, l_rep, M):
	n_items = len(GroundTruth)
	num_dataset = int(comb(n_items, 2*M)*comb(2*M, M)*p_obs)
	pairwise_data = []
	pairwise_data_test = []

	print(num_dataset)
	popular_group = np.random.permutation(range(n_items))[:int(n_items/3)]
	for i in range(num_dataset):
		if i % 1000 == 0:
			print(str(i) + '/' + str(num_dataset))

		group = np.random.choice(range(n_items), 2*M, replace = False)
		group_A = []
		group_B = []
		for m in range(M):
			group_A.append(GroundTruth[group[m]])
			group_B.append(GroundTruth[group[M + m]])
		
		p_comp = win_probability(group_A, group_B)
		for rep_comp in range(l_rep):  			
			A_wins_B =  int(np.random.binomial(1, p_comp))
			pairwise_datum1 = []
			pairwise_datum2 = []
			for tmp in range(2*M):
				pairwise_datum1.append(group[tmp])
				pairwise_datum2.append(group[2*M - tmp - 1])
			pairwise_datum1.append(A_wins_B)
			pairwise_datum2.append(1 - A_wins_B)
			pairwise_data.append(pairwise_datum1)
			pairwise_data.append(pairwise_datum2)

	num_test_dataset = 10000
	for i in range(num_test_dataset):
		group = np.random.choice(range(n_items), 2*M, replace = False)
		group_A = []
		group_B = []
		for m in range(M):
			group_A.append(GroundTruth[group[m]])
			group_B.append(GroundTruth[group[M + m]])
		
		p_comp = win_probability(group_A, group_B)
		for rep_comp in range(l_rep):  			
			A_wins_B =  int(np.random.binomial(1, p_comp))
			pairwise_datum1 = []
			pairwise_datum2 = []
			for tmp in range(2*M):
				pairwise_datum1.append(group[tmp])
				pairwise_datum2.append(group[2*M - tmp - 1])
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

def main(n_items, score_range, p_obs, l_rep, M):
	GroundTruth, GroundTruth_mu = Gen_GroundTruth(n_items, score_range)
	pairwise_data, pairwise_data_test = Gen_data_under_BTL(GroundTruth, p_obs, l_rep, M)
	feature = Gen_feature(n_items, pairwise_data)
	np.savetxt(os.path.join('data/TrueSkill', 'TrueSkill.GT'), GroundTruth_mu)
	np.savetxt(os.path.join('data/TrueSkill', 'TrueSkill.edges'), pairwise_data)
	np.savetxt(os.path.join('data/TrueSkill', 'TrueSkill.edges_test'), pairwise_data_test)
	np.savetxt(os.path.join('data/TrueSkill', 'TrueSkill.nodes'), feature)

if __name__ == "__main__":
	n_items = 300
	M = 5
	score_range = 1e+5
	p_obs = 40*(np.log(n_items) / (comb(n_items - 1, 2*M - 1)*comb(2*M - 1, M - 1)) )
	l_rep = 10

	main(n_items, score_range, p_obs, l_rep, M)
 
