import numpy as np
import os
from random import shuffle
import csv

import matplotlib.pyplot as plt


## Data reference: https://www.kaggle.com/datasnaek/league-of-legends ####

######################### Edge feature ##########################
with open(os.path.join('data/LoL_pro/LeagueofLegends.csv'), 'r') as f:
    reader = csv.reader(f)
    raw_dataset = list(reader)
data_length = len(raw_dataset)

label_list = raw_dataset[0]

M = 5
label_index_list = [[], []]
for team_idx, team in enumerate(['blue', 'red']):
    for m in ['Top', 'Jungle', 'Middle', 'ADC', 'Support']:
        label_index_list[team_idx].append(label_list.index(team + m + 'Champ'))
label_index_list.append(label_list.index('bResult'))
label_index_list[1] = label_index_list[1][::-1]


All_match_data = []
nodes = []
degree = []
for i, data in enumerate(raw_dataset[1:]):
    if i % (10**2) == 0:
        print('Processed: ' + str(i) + '/' + str(data_length))

    team_1_data = []
    team_2_data = []
    for m in range(M):
        node_name1 = data[label_index_list[0][m]]
        node_name2 = data[label_index_list[1][m]]
        if node_name1 in nodes:
            node_idx1 = nodes.index(node_name1)
            degree[node_idx1] += 1
        else:
            nodes.append(node_name1)
            node_idx1 = len(nodes) - 1
            degree.append(1)
        if node_name2 in nodes:
            node_idx2 = nodes.index(node_name2)
            degree[node_idx2] += 1
        else:
            nodes.append(node_name2)
            node_idx2 = len(nodes) - 1
            degree.append(1)

        team_1_data.append(node_idx1)
        team_2_data.append(node_idx2)

    edge_data = data[label_index_list[-1]]

    if edge_data == '0':
        match_data_1 = [team_1_data, team_2_data, '1']
        match_data_2 = [team_2_data, team_1_data, '0']
    elif edge_data == '1':
        match_data_1 = [team_1_data, team_2_data, '0']
        match_data_2 = [team_2_data, team_1_data, '1']
    else:
        print(edge_data)
        print("ERROR")

    flattened_1 = [val for sublist in match_data_1 for val in sublist]
    flattened_1 = [int(x) for x in flattened_1]
    flattened_2 = [val for sublist in match_data_2 for val in sublist]
    flattened_2 = [int(x) for x in flattened_2]

    All_match_data.append(flattened_1)
    All_match_data.append(flattened_2)

np.random.shuffle(All_match_data)
# print(All_match_data)


feature = []
n_items = len(nodes)
feature.append(range(n_items))
feature.append(range(n_items))
feature.append(range(n_items))
feature = list(map(list, zip(*feature)))
GT = range(n_items)

np.savetxt(os.path.join('data/LoL_pro', 'LoL_pro.nodes'), feature)
np.savetxt(os.path.join('data/LoL_pro', 'LoL_pro.GT'), GT)
np.savetxt(os.path.join('data/LoL_pro', 'LoL_pro.edges'), All_match_data[:int(0.8 * len(All_match_data))])
np.savetxt(os.path.join('data/LoL_pro', 'LoL_pro.edges_test'), All_match_data[int(0.8 * len(All_match_data)) + 1:])
