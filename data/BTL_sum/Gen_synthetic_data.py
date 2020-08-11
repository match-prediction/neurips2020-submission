import numpy as np
import os
import argparse
from random import shuffle
from scipy.special import comb

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
args = parser.parse_args()


def Gen_GroundTruth(n_items, score_range):
	GroundTruth = np.zeros(n_items)
	Highest_score = 1
	Lowest_score = 1 / score_range 
	for i in range(n_items):
		GroundTruth[i] = np.random.uniform(Lowest_score, Highest_score)
	GroundTruth = sorted(GroundTruth, reverse = True)
	GroundTruth = GroundTruth / np.linalg.norm(GroundTruth)	
	return GroundTruth

def Gen_data_under_BTL(GroundTruth, p_obs, l_rep, M):
	# M = 2
	n_items = len(GroundTruth)
	num_dataset = int(comb(n_items, 2*M)*comb(2*M, M)*p_obs)
	pairwise_data = []
	pairwise_data_test = []

	print(num_dataset)
	for i in range(num_dataset):
		if i % 1000 == 0:
			print(str(i) + '/' + str(num_dataset))
		group = np.random.choice(range(n_items), 2*M, replace = False)
		group_A_score = sum(GroundTruth[group[:M]])
		group_B_score = sum(GroundTruth[group[M:]])
		p_comp = group_A_score / (group_A_score + group_B_score)

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
		group_A_score = sum(GroundTruth[group[:M]])
		group_B_score = sum(GroundTruth[group[M:]])
		p_comp = group_A_score / (group_A_score + group_B_score)
		
		A_wins_B =  p_comp
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
	GroundTruth = Gen_GroundTruth(n_items, score_range)
	pairwise_data, pairwise_data_test = Gen_data_under_BTL(GroundTruth, p_obs, l_rep, M)
	feature = Gen_feature(n_items, pairwise_data)
	np.savetxt(os.path.join('data/BTL_sum', 'BTL_sum.GT'), GroundTruth)
	np.savetxt(os.path.join('data/BTL_sum', 'BTL_sum.edges'), pairwise_data)
	np.savetxt(os.path.join('data/BTL_sum', 'BTL_sum.edges_test'), pairwise_data_test)
	np.savetxt(os.path.join('data/BTL_sum', 'BTL_sum.nodes'), feature)

if __name__ == "__main__":
	n_items = 300
	M = 5
	score_range = 1e+5
	p_obs = 40*(np.log(n_items) / (comb(n_items - 1, 2*M - 1)*comb(2*M - 1, M - 1)) )
	l_rep = 10

	main(n_items, score_range, p_obs, l_rep, M)
 
	