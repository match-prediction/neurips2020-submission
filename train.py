from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from sklearn.metrics import roc_auc_score
import os
import glob
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from utils import load_data, accuracy
from models import MatchNet
from naive_model import singlenet

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
parser.add_argument('--depth', type=int, default=20, help='Number of iterations.')
parser.add_argument('--dataset', type=str, default='BTL_prod', help='Dataset')
parser.add_argument('--exp_num', type=int, default=0, help='experiment number')
parser.add_argument('--baseline_flag', type=int, default=0, help='Baseline flag')
parser.add_argument('--train_rate', type=float, default=0.1, help='training sample rate')




args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)
torch.set_default_tensor_type('torch.cuda.FloatTensor')

cross_entropy = nn.BCELoss()

# Load data
dataset = args.dataset
path = "./data/" + dataset + "/"
N, edges_train, edges_dev, edges_test, features = load_data(path, dataset, args)
features, edges_train, edges_dev, edges_test = Variable(features), Variable(edges_train), Variable(edges_dev), Variable(edges_test)

# Model and optimizer
model = MatchNet(depth_T = args.depth,  N = N, M = int((edges_train.shape[1] - 1)/2))
# model = singlenet(depth = args.depth,  N = N, M = int((edges_train.shape[1] - 1)/2))
print("==============================================================")

if args.cuda:
	model = model.cuda()
	features = features.cuda()
	edges_train = edges_train.cuda()
	edges_dev = edges_dev.cuda()
	edges_test = edges_test.cuda()

def train(epoch, input_idx, output_idx):
	t = time.time()
	M = int((edges_train.shape[1] - 1)/2)
	
	graph_edge = edges_train[input_idx, :]
	graph_edge = torch.cat([graph_edge, torch.cat([torch.cat([graph_edge[:, M:2*M].view(-1, M), graph_edge[:, :M].view(-1, M)], dim = 1), (1 - graph_edge[:, -1]).view(-1, 1)], dim = 1)])
	graph_label = edges_train[output_idx, :]
	graph_label = torch.cat([graph_label, torch.cat([torch.cat([graph_label[:, M:2*M].view(-1, M), graph_label[:, :M].view(-1, M)], dim = 1), (1 - graph_label[:, -1]).view(-1, 1)], dim = 1)])
	graph_edge[:, -1] = torch.where(graph_edge[:, -1] == torch.zeros_like(graph_edge[:, -1]), -torch.ones_like(graph_edge[:, -1]), graph_edge[:, -1])

	model.train()   
	optimizer.zero_grad()
	output = model(features, graph_edge, graph_label[:, :-1])
	# output = model(graph_edge) # for naive
	
	loss_train = cross_entropy(output.squeeze(), graph_label[:, -1])
	
	loss_train.backward()
	optimizer.step()


	# Evaluation per epoch
	with torch.no_grad():
		model.eval()

		edges_train_full = torch.cat([edges_train, edges_dev], dim = 0)
		edges_train_epoch = torch.zeros_like(edges_train_full)
		edges_train_epoch[:, :-1] = edges_train_full[:, :-1]
		edges_train_epoch[:, -1] = torch.where(edges_train_full[:, -1] == torch.zeros_like(edges_train_full[:, -1]), -torch.ones_like(edges_train_full[:, -1]), edges_train_full[:, -1])
		edges_train_epoch = torch.cat([edges_train_epoch, torch.cat([torch.cat([edges_train_epoch[:, M:2*M].view(-1, M), edges_train_epoch[:, :M].view(-1, M)], dim = 1), - edges_train_epoch[:, -1].view(-1, 1)], dim = 1)])
		

		# deactivates dropout during validation run.
		output_train_mnt = model(features, edges_train_epoch, edges_train[:, :-1])
		output_mnt = model(features, edges_train_epoch, edges_dev[:, :-1])
		output_test_mnt = model(features, edges_train_epoch, edges_test[:, :-1])
		
		# output_train_mnt = model(edges_train[:, :-1]) # for naive
		# output_mnt = model(edges_dev[:, :-1]) # for naive
		# output_test_mnt = model(edges_test[:, :-1]) # for naive
		
		loss_train_mnt = cross_entropy(output_train_mnt.squeeze(), edges_train[:, -1])
		loss_val_mnt = cross_entropy(output_mnt.squeeze(), edges_dev[:, -1])
		loss_test_mnt = cross_entropy(output_test_mnt.squeeze(), edges_test[:, -1])
		


		acc_train, exacc_train, auc_train, Hinge_loss_train, CE_loss_train, MSE_loss_train = accuracy(output_train_mnt, edges_train[:, -1])
		acc_val, exacc_val, auc_val, Hinge_loss_val, CE_loss_val, MSE_loss_val = accuracy(output_mnt, edges_dev[:, -1])
		acc_test, exacc_test, auc_test, Hinge_loss_test, CE_loss_test, MSE_loss_test = accuracy(output_test_mnt, edges_test[:, -1])
		
		print('Epoch: {:04d}'.format(epoch+1),
			  'loss_train: {:.4f}'.format(CE_loss_train),
			  'acc_train: {:.4f}'.format(acc_train),
			  'loss_val: {:.4f}'.format(CE_loss_val),
			  'acc_val: {:.4f}'.format(acc_val),
			  'loss_test: {:.4f}'.format(CE_loss_test),
			  'acc_test: {:.4f}'.format(acc_test),
			  'time: {:.4f}s'.format(time.time() - t))

	return CE_loss_train, acc_train, auc_train, CE_loss_val, acc_val, auc_val, CE_loss_test, acc_test, auc_test

def compute_test(no_save = False):
	with torch.no_grad():
		model.eval()
		M = int((edges_train.shape[1] - 1)/2)

		edges_train_full = torch.cat([edges_train, edges_dev], dim = 0)  
		# edges_train_full = edges_train  

		edges_train_epoch = torch.zeros_like(edges_train_full)
		edges_train_epoch[:, :-1] = edges_train_full[:, :-1]
		edges_train_epoch[:, -1] = torch.where(edges_train_full[:, -1] == torch.zeros_like(edges_train_full[:, -1]), -torch.ones_like(edges_train_full[:, -1]), edges_train_full[:, -1])
		edges_train_epoch = torch.cat([edges_train_epoch, torch.cat([torch.cat([edges_train_epoch[:, M:2*M].view(-1, M), edges_train_epoch[:, :M].view(-1, M)], dim = 1),  - edges_train_epoch[:, -1].view(-1, 1)], dim = 1)])

		output_val = model(features, edges_train_epoch, edges_dev[:, :-1])
		output = model(features, edges_train_epoch, edges_test[:, :-1])
		
		# output_val = model(edges_dev[:, :-1]) # for naive
		# output = model(edges_test[:, :-1]) # for naive
		
		acc_val, exacc_val, auc_val, Hinge_loss_val, CE_loss_val, MSE_loss_val = accuracy(output_val, edges_dev[:, -1])
		acc_test, exacc_test, auc_test, Hinge_loss_test, CE_loss_test, MSE_loss_test = accuracy(output, edges_test[:, -1])
		
		test_result = {}
		test_result["proposed"] = [acc_val, CE_loss_val, auc_val, acc_test, CE_loss_test, auc_test, Hinge_loss_test, MSE_loss_test] 
		# print(test_result)
		if no_save == False:	
			np.save(os.path.join('result', args.dataset + '_expnum=' + str(args.exp_num) + '_seed=' + str(args.seed) + '_train_rate=' + str(args.train_rate) + '_lr=' + str(args.lr) + '_depth=' + str(args.depth) + '_' + 'test'), test_result)

	
# Train model
t_total = time.time()
loss_values = []
acc_values = []
bad_counter = 0
best = 1e+5
best_epoch = 0


optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)


train_monitor = {}
train_monitor["train_loss"] = []
train_monitor["train_acc"] = []
train_monitor["train_auc"] = []
train_monitor["val_loss"] = []
train_monitor["val_acc"] = []
train_monitor["val_auc"] = []
train_monitor["test_loss"] = []
train_monitor["test_acc"] = []
train_monitor["test_auc"] = []


# fraction of D_train for 1st stage
# num_epoch: # of total training epochs
if args.baseline_flag == 0:
	if args.dataset == 'DOTA':
		batch_frac = 1
		# batch_frac = 3
		num_epoch = 300
	elif args.dataset == 'HOTS':
		batch_frac = 1
		# batch_frac = 3
		num_epoch = 300
	elif args.dataset == 'LoL':
		batch_frac = 1
		# batch_frac = 3
		num_epoch = 300
	elif args.dataset == 'LoL_pro':
		batch_frac = 1
		# batch_frac = 3
		num_epoch = 300
	elif args.dataset == 'movie':
		batch_frac = 1
		# batch_frac = 10
		num_epoch = 600
	elif args.dataset == 'GIFGIF':
		batch_frac = 1
		# batch_frac = 3
		num_epoch = 1000
	elif args.dataset == 'BTL_sum':
		# batch_frac = 1
		batch_frac = 5
		num_epoch = 300
	elif args.dataset == 'BTL_prod':
		# batch_frac = 1
		batch_frac = 5
		num_epoch = 400
	elif args.dataset == 'TrueSkill':
		# batch_frac = 1
		batch_frac = 5
		num_epoch = 400
	elif args.dataset == 'BTL_HOI':
		# batch_frac = 1
		batch_frac = 5
		num_epoch = 300


batch_idx = 0
new_batch_flag = 1
for epoch in range(num_epoch):

	if (batch_idx%batch_frac) == 0:
		rand_split = np.random.permutation(edges_train.size()[0])
		batch_idx = 0

	input_idx = rand_split[int(batch_idx/batch_frac*len(rand_split)):int((batch_idx + 1)/batch_frac*len(rand_split))]
	output_idx = rand_split
	batch_idx = batch_idx + 1
	

	train_loss_epoch, train_acc_epoch, train_auc_epoch, val_loss_epoch, val_acc_epoch, val_auc_epoch, test_loss_epoch, test_acc_epoch, test_auc_epoch = train(epoch, input_idx, output_idx)

	train_monitor["train_loss"].append(train_loss_epoch) 
	train_monitor["train_acc"].append(train_acc_epoch)
	train_monitor["train_auc"].append(train_auc_epoch)
	train_monitor["val_loss"].append(val_loss_epoch)
	train_monitor["val_acc"].append(val_acc_epoch)
	train_monitor["val_auc"].append(val_auc_epoch)
	train_monitor["test_loss"].append(test_loss_epoch)
	train_monitor["test_acc"].append(test_acc_epoch)
	train_monitor["test_auc"].append(test_auc_epoch)
	
	if epoch % 5 == 0:
		pass
		print("================================== EPOCH: {} ==================================".format(epoch))
		# print("Loss-tr: {} Loss-dev: {} Loss-test: {}".format(train_loss_epoch, val_loss_epoch, test_loss_epoch))
		# print("-------------------------------------------------------------------------------")
		# print("Acc-tr: {} Acc-dev: {} Acc-test: {}".format(train_acc_epoch, val_acc_epoch, test_acc_epoch))
		# print("-------------------------------------------------------------------------------")
		# print("Auc-tr: {} Auc-dev: {} Auc-test: {}".format(train_auc_epoch, val_auc_epoch, test_auc_epoch))
		# print("===============================================================================")
	
	if epoch > 0:
		loss_values.append(val_loss_epoch)
		acc_values.append(val_acc_epoch)
		torch.save(model.state_dict(), './data/{dataset}/{epoch}.pkl'.format(dataset = args.dataset, epoch = epoch))
		


		if loss_values[-1] < best:
			best = loss_values[-1]
			best_epoch = epoch
			bad_counter = 0
		else:
			bad_counter += 1


	files = glob.glob('./data/{dataset}/*.pkl'.format(dataset = args.dataset))
	for file in files:
		parsed_file = file[len('./data/{dataset}/'.format(dataset = args.dataset)):]
		epoch_nb = int(parsed_file.split('.')[0])
		if epoch_nb < best_epoch:
			os.remove(file)

np.save(os.path.join('result', args.dataset + '_expnum=' + str(args.exp_num) + '_seed=' + str(args.seed) + '_train_rate=' + str(args.train_rate) + '_lr=' + str(args.lr) + '_depth=' + str(args.depth) + '_' + 'train'), train_monitor)



files = glob.glob('./data/{dataset}/*.pkl'.format(dataset = args.dataset))
for file in files:
	parsed_file = file[len('./data/{dataset}/'.format(dataset = args.dataset)):]
	epoch_nb = int(parsed_file.split('.')[0])
	if epoch_nb > best_epoch:
		os.remove(file)

	
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
# best_epoch = 384
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('./data/{dataset}/{epoch}.pkl'.format(dataset = args.dataset, epoch = best_epoch)))


t_test = time.time()
compute_test()
print("Total TEST time elapsed: {:.4f}s".format(time.time() - t_test))
