import numpy as np
import os
from random import shuffle
import csv

import matplotlib.pyplot as plt




# ######################## Edge feature ##########################  
with open('data/GIFGIF/raw_data.csv', 'r') as f:
    reader = csv.reader(f)
    raw_dataset = list(reader)
nodes = []
degree = []
edges = []
data_length = len(raw_dataset)
for i, data in enumerate(raw_dataset[1:]):
    if i % (10**4) == 0:
        print('Filter 1: ' + str(i) + '/' + str(data_length))
    if data[0] == 'happiness' and data[3] != 'neither':
        node_name1 = data[1]
        node_name2 = data[2]
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
        if data[3] == 'left':
            edge = [node_idx1, node_idx2, 1]
            edges.append(edge)
            edge = [node_idx2, node_idx1, 0]
            edges.append(edge)
        elif data[3] == 'right':
            edge = [node_idx2, node_idx1, 1]
            edges.append(edge)
            edge = [node_idx1, node_idx2, 0]
            edges.append(edge)
        elif data[3] == 'neither':
            edge = [node_idx1, node_idx2, 1/2]
            edges.append(edge)
            edge = [node_idx2, node_idx1, 1/2]
            edges.append(edge)

# th = 55
th = 25
new_edges = []
new_nodes = []
new_degree = []
edge_length = len(edges)
for i, data in enumerate(edges):
    if i % (10**3) == 0:
        print('Filter 2: ' + str(i) + '/' + str(edge_length))
    node_idx1 = data[0]
    node_idx2 = data[1]
    if degree[node_idx1] > th and degree[node_idx2] > th:
        if node_idx1 in new_nodes:
            node_new_idx1 = new_nodes.index(node_idx1)
            new_degree[node_new_idx1] += 1
        else:
            new_nodes.append(node_idx1)
            node_new_idx1 = len(new_nodes) - 1
            new_degree.append(1)
        if node_idx2 in new_nodes:
            node_new_idx2 = new_nodes.index(node_idx2)
            new_degree[node_new_idx2] += 1
        else:
            new_nodes.append(node_idx2)
            node_new_idx2 = len(new_nodes) - 1
            new_degree.append(1)
        edge = [node_new_idx1, node_new_idx2, data[2]]
        new_edges.append(edge)
print(np.mean(new_degree))
np.random.shuffle(new_edges)
# #################################################################
np.savetxt(os.path.join('data/GIFGIF', 'GIFGIF.edges'), new_edges[:int(0.9*len(new_edges))])
np.savetxt(os.path.join('data/GIFGIF', 'GIFGIF.edges_test'), new_edges[int(0.9*len(new_edges))+1:])

feature = []
n_items = int(np.max(new_edges) + 1)
feature.append(range(n_items))
feature.append(range(n_items))
feature.append(range(n_items))
feature = list(map(list, zip(*feature)))
np.savetxt(os.path.join('data/GIFGIF', 'GIFGIF.nodes'), feature)
GT = range(n_items)
np.savetxt(os.path.join('data/GIFGIF', 'GIFGIF.GT'), GT)
