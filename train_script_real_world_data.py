import os
import math
import numpy as np




exp_num_proposed = [190917]


for dataset in ['movie', 'GIFGIF', 'LoL_pro', 'HOTS', 'DOTA']:
	for exp_num in exp_num_proposed:
		for train_rate in [0.99]:
			for baseline_flag in [1, 0]:
				if baseline_flag == 1:
					pass
					os.system('python train.py --dataset {dataset} --exp_num {exp_num} --baseline_flag {baseline_flag} --train_rate {train_rate} --seed {exp_num}'.format(dataset = dataset, exp_num = exp_num, baseline_flag = baseline_flag, train_rate = train_rate, seed = exp_num))
				else:	
					for depth in [20]:
						for lr in [1e-2]:	
							pass
							os.system('python train.py --dataset {dataset} --exp_num {exp_num} --train_rate {train_rate} --lr {lr} --depth {depth} --seed {exp_num}'.format(dataset = dataset, exp_num = exp_num, baseline_flag = baseline_flag, train_rate = train_rate, lr = lr, depth = depth, seed = exp_num))
