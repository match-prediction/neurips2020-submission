import numpy as np
import os
from random import shuffle
import csv

import matplotlib.pyplot as plt




## Data reference: https://www.kaggle.com/devinanzelmo/dota-2-matches/version/3 #### 

######################### Edge feature ##########################  
with open('data/DOTA/players.csv', 'r') as f:
	reader = csv.reader(f)
	raw_players = list(reader)
with open('data/DOTA/match.csv', 'r') as f:
    reader = csv.reader(f)
    raw_matches = list(reader)

data_length = len(raw_matches)


All_match_data = []
M = 5
heroes = range(113)
for match_idx, data in enumerate(raw_matches[1:]):
    team_1_2 = []
    team_2_1 = []
    if data[9] == 'True':
        team_split_1 = [0,1,2,3,4,5,6,7,8,9]
        team_split_2 = [9,8,7,6,5,4,3,2,1,0]
    elif data[9] == 'False':
        team_split_1 = [9,8,7,6,5,4,3,2,1,0]
        team_split_2 = [0,1,2,3,4,5,6,7,8,9]
    else:
        print(raw_matches[match_idx])
        
    for hero_idx in team_split_1:
        temp_hero_idx = (10*match_idx + hero_idx + 1)
        team_1_2.append(int(raw_players[temp_hero_idx][2]) - 1)
    for hero_idx in team_split_2:
        temp_hero_idx = (10*match_idx + hero_idx + 1)
        team_2_1.append(int(raw_players[temp_hero_idx][2]) - 1)
    
    if not (-1 in team_1_2):
        team_1_2.append(1)
        team_2_1.append(0)
        All_match_data.append(team_1_2)
        All_match_data.append(team_2_1)    



feature = []
n_items = len(heroes)
feature.append(range(n_items))
feature.append(range(n_items))
feature.append(range(n_items))
feature = list(map(list, zip(*feature)))
GT = range(n_items)

np.savetxt(os.path.join('data/DOTA', 'DOTA.nodes'), feature)
np.savetxt(os.path.join('data/DOTA', 'DOTA.GT'), GT)
np.savetxt(os.path.join('data/DOTA', 'DOTA.edges'), All_match_data[:int(0.9*len(All_match_data))])
np.savetxt(os.path.join('data/DOTA', 'DOTA.edges_test'), All_match_data[int(0.9*len(All_match_data))+1:])

