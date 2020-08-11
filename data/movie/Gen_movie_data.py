import numpy as np
import os
from random import shuffle
import csv
import random

import matplotlib.pyplot as plt



 
## https://www.kaggle.com/carolzhangdc/imdb-5000-movie-dataset

######################### Edge feature ########################## 
with open(os.path.join('data/movie/movie_metadata.csv'), 'r') as f:
	reader = csv.reader(f)
	raw_dataset = list(reader)
data_length = len(raw_dataset)

label_list = raw_dataset[0]
key_idx = label_list.index('plot_keywords')
imdb_idx = label_list.index('imdb_score')

all_movie_rating = []
for i, data in enumerate(raw_dataset):
	keywords = data[key_idx]
	imdb_score = data[imdb_idx]
	if len(keywords) > 0:
		keywords = keywords.split("|")
		if len(keywords) == 5:
			all_movie_rating.append([keywords, imdb_score])

print(len(all_movie_rating))

all_keywords = []
degree = []
for idx, movie_1 in enumerate(all_movie_rating):
	if idx % (10**2) == 0:
		print('Processed: ' + str(idx) + '/' + str(len(all_movie_rating)))
	
	for kwd in movie_1[0]:
		if kwd in all_keywords:
			kwd_idx = all_keywords.index(kwd)
			degree[kwd_idx] += 1
		else:
			all_keywords.append(kwd)
			kwd_idx = len(all_keywords) - 1
			degree.append(1)

print(np.mean(np.array(degree)))
print(len(degree))


all_movie_rating_trn = all_movie_rating[:int(0.8*len(all_movie_rating))]
# all_movie_rating_trn = all_movie_rating[int(0.8*len(all_movie_rating))-25:int(0.8*len(all_movie_rating))]
# all_movie_rating_tst = all_movie_rating[int(0.8*len(all_movie_rating)):]
all_movie_rating_tst = all_movie_rating[int(0.8*len(all_movie_rating)):int(0.8*len(all_movie_rating))+50]
# all_movie_rating_tst = all_movie_rating[:50]
All_match_data_trn = []
All_match_data_tst = []
for idx, movie_1 in enumerate(all_movie_rating_trn[:-1]):
	if idx % (10**2) == 0:
		print('Processed: ' + str(idx) + '/' + str(len(all_movie_rating_trn)))

	movie_2_list = random.sample(range(len(all_movie_rating_trn))[idx+1:], np.min([15, len(all_movie_rating_trn) - idx - 1]))
	# np.random.choice(len(all_movie_rating_trn), 3, replace=False, p=[0.1, 0, 0.3, 0.6, 0])
	# print(movie_2_list)
	# [:20]
	for movie_2_idx in movie_2_list:
		movie_2 = all_movie_rating_trn[movie_2_idx]
		movie_1_data = []
		movie_2_data = []

		# if len(movie_1[0]) ==  and len(movie_2[0]) == 3:
		for kwd in movie_1[0]:
			kwd_idx = all_keywords.index(kwd)
			degree[kwd_idx] += 1
			movie_1_data.append(kwd_idx)

		for kwd in movie_2[0]:
			kwd_idx = all_keywords.index(kwd)
			degree[kwd_idx] += 1
			movie_2_data.append(kwd_idx)
	

		if movie_1[1] > movie_2[1]:
			edge_data = '0' 
		elif movie_1[1] < movie_2[1]:
			edge_data = '1'
		elif movie_1[1] == movie_2[1]:
			edge_data = '1/2'
			
		if edge_data == '0':
			match_data_1 = [movie_1_data, movie_2_data, '1']
			match_data_2 = [movie_2_data, movie_1_data, '0']
		elif edge_data == '1':
			match_data_1 = [movie_1_data, movie_2_data, '0']
			match_data_2 = [movie_2_data, movie_1_data, '1']
		elif edge_data == '1/2':
			bern = np.random.binomial(1, 0.5, 1)
			if bern == 1:
				match_data_1 = [movie_1_data, movie_2_data, '1']
				match_data_2 = [movie_2_data, movie_1_data, '0']
			else:
				match_data_1 = [movie_1_data, movie_2_data, '0']
				match_data_2 = [movie_2_data, movie_1_data, '1']
		else:
			print(edge_data)
			print("ERROR")
		
		flattened_1 = [val for sublist in match_data_1 for val in sublist]
		flattened_1 = [int(x) for x in flattened_1]
		flattened_2 = [val for sublist in match_data_2 for val in sublist]
		flattened_2 = [int(x) for x in flattened_2]

		All_match_data_trn.append(flattened_1)
		All_match_data_trn.append(flattened_2)




tst_movie_rating = []
# all_movie_rating_tst = all_movie_rating_trn
for idx, movie_1 in enumerate(all_movie_rating_tst):
	if idx % (10**2) == 0:
		print('Processed: ' + str(idx) + '/' + str(len(all_movie_rating_tst)))

	# print(movie_1[1])
	tst_movie_rating.append(float(movie_1[1]))
	movie_2_list = all_movie_rating_tst
	# movie_2_list = random.sample(range(len(all_movie_rating_tst))[idx+1:], np.min([15, len(all_movie_rating_tst) - idx - 1]))
	# print(movie_2_list)
	# [:20]
	for movie_2_idx in range(len(movie_2_list)):
	# for movie_2_idx in movie_2_list[:-1]:
		movie_2 = all_movie_rating_tst[movie_2_idx]
		movie_1_data = []
		movie_2_data = []

		# if len(movie_1[0]) ==  and len(movie_2[0]) == 3:
		for kwd in movie_1[0]:
			kwd_idx = all_keywords.index(kwd)
			degree[kwd_idx] += 1
			movie_1_data.append(kwd_idx)

		for kwd in movie_2[0]:
			kwd_idx = all_keywords.index(kwd)
			degree[kwd_idx] += 1
			movie_2_data.append(kwd_idx)
	
		if movie_1[1] > movie_2[1]:
			edge_data = '0' 
		elif movie_1[1] < movie_2[1]:
			edge_data = '1'
		elif movie_1[1] == movie_2[1]:
			edge_data = '1/2'
		# print(movie_1[1], movie_2[1], edge_data)

		if edge_data == '0':
			match_data_1 = [movie_1_data, movie_2_data, '1']
			match_data_2 = [movie_2_data, movie_1_data, '0']
		elif edge_data == '1':
			match_data_1 = [movie_1_data, movie_2_data, '0']
			match_data_2 = [movie_2_data, movie_1_data, '1']
		elif edge_data == '1/2':
			bern = np.random.binomial(1, 0.5, 1)
			if bern == 1:
				match_data_1 = [movie_1_data, movie_2_data, '1']
				match_data_2 = [movie_2_data, movie_1_data, '0']
			else:
				match_data_1 = [movie_1_data, movie_2_data, '0']
				match_data_2 = [movie_2_data, movie_1_data, '1']
		else:
			print(edge_data)
			print("ERROR")
		
		flattened_1 = [val for sublist in match_data_1 for val in sublist]
		flattened_1 = [int(x) for x in flattened_1]
		flattened_2 = [val for sublist in match_data_2 for val in sublist]
		flattened_2 = [int(x) for x in flattened_2]

		All_match_data_tst.append(flattened_1)
		All_match_data_tst.append(flattened_2)	 

# np.random.shuffle(All_match_data)
# print(np.array(All_match_data).shape)


feature = []
n_items = len(all_keywords)
feature.append(range(n_items))
feature.append(range(n_items))
feature.append(range(n_items))
feature = list(map(list, zip(*feature)))
GT = range(n_items)

np.savetxt(os.path.join('data/movie', 'movie.nodes'), feature)
np.savetxt(os.path.join('data/movie', 'movie.GT'), GT)
np.savetxt(os.path.join('data/movie', 'movie.edges'), All_match_data_trn)
np.savetxt(os.path.join('data/movie', 'movie.edges_test'), All_match_data_tst)
np.savetxt(os.path.join('data/movie', 'movie.GT_test'), tst_movie_rating)
