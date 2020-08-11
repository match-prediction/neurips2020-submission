import os
import math
import numpy as np


exp_num = 200701
for seed in range(30):
	for dataset in ['BTL_sum', 'BTL_prod', 'BTL_HOI', 'TrueSkill']:
		pass
		# os.system('python data/{}/Gen_synthetic_data.py'.format(dataset))		
		for train_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
		# for train_rate in [0.6]:
			for baseline_flag in [1, 0]:
				if baseline_flag == 1:
					pass
					# os.system('python train.py --dataset {dataset} --exp_num {exp_num} --seed {seed} --baseline_flag {baseline_flag} --train_rate {train_rate}'.format(dataset = dataset, exp_num = exp_num, seed=seed, baseline_flag = baseline_flag, train_rate = train_rate))
				else:
					for depth in [20]:
						for lr in [1e-2]:
							# pass
							os.system('python train.py --dataset {dataset} --exp_num {exp_num} --seed {seed} --train_rate {train_rate} --lr {lr} --depth {depth}'.format(dataset = dataset, exp_num = exp_num, seed=seed, baseline_flag = baseline_flag, train_rate = train_rate, lr = lr, depth = depth))