import numpy as np
import os
from random import shuffle
import csv

import matplotlib.pyplot as plt




## Data reference: https://www.kaggle.com/datasnaek/league-of-legends #### 

######################### Edge feature ##########################  
with open('data/HOTS/ReplayCharacters.csv', 'r') as f:
	reader = csv.reader(f)
	raw_dataset = list(reader)
data_length = len(raw_dataset)

with open('data/HOTS/Replays.csv', 'r') as f:
    reader = csv.reader(f)
    raw_mode_dataset = list(reader)
data_length = len(raw_dataset)


M = 5
heroes = range(84)

##################################### Training dataset ##################################################
All_match_data = []
All_match_mmr = []
All_match_data_test = []
All_match_mmr_test = []

for i, data in enumerate(raw_dataset[1:]):
# for i, data in enumerate(raw_dataset[9365000:]):
#     print(data)
    match_idx = int(i/10)
    if i%10 == 0:
        if i != 0:
            match_mmr = match_mmr / (2*M)
            if match_mmr > 2700:    
                match_data_1 = [team_1, team_2, '1']
                match_data_2 = [team_2, team_1, '0']

                flattened_1 = [val for sublist in match_data_1 for val in sublist]
                flattened_1 = [int(x) for x in flattened_1]
                flattened_2 = [val for sublist in match_data_2 for val in sublist]
                flattened_2 = [int(x) for x in flattened_2]
                
                if match_mmr > 3200:
                    in_test = np.random.binomial(1, 0.5, 1)
                    if in_test[0] == 1:
                        if len(flattened_1) == 11:
                            All_match_data_test.append(flattened_1)
                            All_match_data_test.append(flattened_2)
                            All_match_mmr_test.append(match_mmr)
                    else:
                        if len(flattened_1) == 11:
                            All_match_data.append(flattened_1)
                            All_match_data.append(flattened_2)
                            All_match_mmr.append(match_mmr)
                else:
                    if len(flattened_1) == 11:
                        All_match_data = [flattened_1] + All_match_data
                        All_match_data = [flattened_2] + All_match_data
                        All_match_mmr = [match_mmr] + All_match_mmr

        team_1 = []
        team_2 = []
        match_mmr = 0

    if data[5] == '':
        match_mmr += 0
    else:
        match_mmr += int(data[5])
#         print(match_mmr)
    
    
    if data[4] == '1':
        team_1.append(int(data[2]) - 1)
#         match_data_1.insert(0, int(data[2]) - 1)
#         match_data_2.insert(-1, int(data[2]) - 1)
    else:
        team_2.append(int(data[2]) - 1)
#         match_data_1.insert(-1, int(data[2]) - 1)
#         match_data_2.insert(0, int(data[2]) - 1)
            
# np.random.shuffle(All_match_data)
############################################################################################################



# ##################################### Test dataset ##################################################

# All_match_data_test = []
# All_match_mmr_test = []
# for i, data in enumerate(raw_dataset[1:]):
# # for i, data in enumerate(raw_dataset[9365000:]):
# #     print(data)
#     match_idx = int(i/10)
#     if i%10 == 0:
#         if i != 0:
#             match_mmr = match_mmr / (2*M)
#             if match_mmr > 3200:
#                 match_data_1 = [team_1, team_2, '1']
#                 match_data_2 = [team_2, team_1, '0']

#                 flattened_1 = [val for sublist in match_data_1 for val in sublist]
#                 flattened_1 = [int(x) for x in flattened_1]
#                 flattened_2 = [val for sublist in match_data_2 for val in sublist]
#                 flattened_2 = [int(x) for x in flattened_2]
                
#                 if len(flattened_1) == 11:
#                     All_match_data_test.append(flattened_1)
#                     All_match_data_test.append(flattened_2)
#                     All_match_mmr_test.append(match_mmr)
#         team_1 = []
#         team_2 = []
#         match_mmr = 0

#     if data[5] == '':
#         match_mmr += 0
#     else:
#         match_mmr += int(data[5])
# #         print(match_mmr)
    
    
#     if data[4] == '1':
#         team_1.append(int(data[2]) - 1)
# #         match_data_1.insert(0, int(data[2]) - 1)
# #         match_data_2.insert(-1, int(data[2]) - 1)
#     else:
#         team_2.append(int(data[2]) - 1)
# #         match_data_1.insert(-1, int(data[2]) - 1)
# #         match_data_2.insert(0, int(data[2]) - 1)
            
# # np.random.shuffle(All_match_data)
# ############################################################################################################


feature = []
n_items = len(heroes)
feature.append(range(n_items))
feature.append(range(n_items))
feature.append(range(n_items))
feature = list(map(list, zip(*feature)))
GT = range(n_items)

np.savetxt(os.path.join('data/HOTS', 'HOTS.nodes'), feature)
np.savetxt(os.path.join('data/HOTS', 'HOTS.GT'), GT)
np.savetxt(os.path.join('data/HOTS', 'HOTS.edges'), All_match_data)
# np.savetxt(os.path.join('data/HOTS', 'HOTS.edges'), All_match_data[:int(0.9*len(All_match_data))])
# np.savetxt(os.path.join('data/HOTS', 'HOTS.edges_test'), All_match_data[int(0.9*len(All_match_data))+1:])
np.savetxt(os.path.join('data/HOTS', 'HOTS.edges_test'), All_match_data_test)


# fig = plt.figure()
# ax = plt.subplot(111)

# plt.hist(match_mmr, normed=True, bins=30)
# plt.show()


#################### Rank Centrality feature ####################  
# n_iteration = 10000
# n_items = int(np.max(new_edges) + 1)
# feature = []

# score = np.ones(n_items)
# Markov_chain = np.zeros((n_items, n_items))
# adjacency = np.zeros((n_items, n_items))
# for i, datum in enumerate(new_edges):
#     idx1, idx2 = int(datum[0]), int(datum[1])
#     Markov_chain[idx1][idx2] += datum[2]
#     Markov_chain[idx2][idx1] += 1 - datum[2]
#     adjacency[idx1][idx2] += 1
#     adjacency[idx2][idx1] += 1
# degree = np.sum(adjacency, axis = 0)
# dmax = max(degree)
# count = np.sum(Markov_chain, axis = 1)
# score2 = count / degree
# Markov_chain = Markov_chain / dmax  
# for i in range(n_items):
#     Markov_chain[i][i] = 1 - np.sum(Markov_chain[:, i])
# for i in range(n_iteration):
#     # if i % 1000 ==0:
#     # print(str(i) + '/' + str(n_iteration))
#     score = Markov_chain.dot(score)

# dataset = new_edges
# n_iteration = 1000
# epsilon = 1e-10
# score3 = np.ones(n_items) / n_items
# n_win = np.zeros(n_items)
# adjacency = np.zeros((n_items, n_items))
# for i, datum in enumerate(dataset):
#     idx1, idx2, winloss = int(datum[0]), int(datum[1]), int(datum[2])
#     if winloss == 1:
#         n_win[idx1] += 1
#     elif winloss == 0:
#         n_win[idx2] += 1 

#     adjacency[idx1][idx2] += 1
#     adjacency[idx2][idx1] += 1

# adjacency = np.array(adjacency, dtype = np.float)
# for iter_idx in range(n_iteration):
#     score_matrix = np.repeat([score3], n_items, axis = 0)
#     score_matrix = score_matrix + np.transpose(score_matrix)

#     recipro_score = adjacency / score_matrix
#     recipro_score = np.sum(recipro_score, axis = 1)
#     score3 = (n_win + epsilon*np.ones(n_items)) / recipro_score

#     # for i in range(n_items):
#     #     score3[i] = (n_win[i] + epsilon) / recipro_score[i]

# score = score / max(score)
# score2 = score2 / max(score2)
# score3 = score3 / max(score3)

# feature.append(range(n_items))
# feature.append(score)
# feature.append(score2)
# feature.append(score3)
# feature.append(range(n_items))

# feature = list(map(list, zip(*feature)))
# #################################################################
# np.savetxt(os.path.join('data/GIFGIF', 'GIFGIF.nodes'), feature)
