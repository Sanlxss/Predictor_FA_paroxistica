import os
import logging
import pickle
import sys

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math 
import glob
import re
from shutil import copyfile
import sklearn as sk
import subprocess
import datetime

def makedirs(dirname):
	if not os.path.exists(dirname):
		os.makedirs(dirname)

def get_logger(logpath, filepath, package_files=[],
			   displaying=True, saving=True, debug=False):
	logger = logging.getLogger()
	if debug:
		level = logging.DEBUG
	else:
		level = logging.INFO
	logger.setLevel(level)
	if saving:
		info_file_handler = logging.FileHandler(logpath, mode='w')
		info_file_handler.setLevel(level)
		logger.addHandler(info_file_handler)
	if displaying:
		console_handler = logging.StreamHandler()
		console_handler.setLevel(level)
		logger.addHandler(console_handler)
	logger.info(filepath)

	for f in package_files:
		logger.info(f)
		with open(f, 'r') as package_f:
			logger.info(package_f.read())
	
	return logger

def inf_generator(iterable):
	"""Allows training with DataLoaders in a single infinite loop:
		for i, (x, y) in enumerate(inf_generator(train_loader)):
	"""
	iterator = iterable.__iter__()
	while True:
		try:
			yield iterator.__next__()
		except StopIteration:
			iterator = iterable.__iter__()

def split_last_dim(data):
	last_dim = data.size()[-1]
	last_dim = last_dim//2

	if len(data.size()) == 3:
		res = data[:,:,:last_dim], data[:,:,last_dim:]

	if len(data.size()) == 2:
		res = data[:,:last_dim], data[:,last_dim:]
	return res

def init_network_weights(net, std = 0.1):
	for m in net.modules():
		if isinstance(m, nn.Linear):
			nn.init.normal_(m.weight, mean=0, std=std)
			nn.init.constant_(m.bias, val=0)

def get_device(tensor):
	device = torch.device("cpu")
	if tensor.is_cuda:
		device = tensor.get_device()
	return device

def sample_standard_gaussian(mu, sigma):
	device = get_device(mu)

	d = torch.distributions.normal.Normal(torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))
	r = d.sample(mu.size()).squeeze(-1)
	return r * sigma.float() + mu.float()

def get_next_batch(dataloader):
    	# Make the union of all time points and perform normalization across the whole dataset
	data_dict = dataloader.__next__()
	# Obtener las etiquetas
	batch_dict = get_dict_template()
	

	# remove the time points where there are no observations in this batch
	non_missing_tp = torch.sum(data_dict["observed_data"],(0,2)) != 0.
	batch_dict["observed_data"] = data_dict["observed_data"][:, non_missing_tp]
	batch_dict["observed_tp"] = data_dict["observed_tp"][non_missing_tp]

	if ("observed_mask" in data_dict) and (data_dict["observed_mask"] is not None):
		batch_dict["observed_mask"] = data_dict["observed_mask"][:, non_missing_tp]

	batch_dict[ "data_to_predict"] = data_dict["data_to_predict"]
	batch_dict["tp_to_predict"] = data_dict["tp_to_predict"]

	non_missing_tp = torch.sum(data_dict["data_to_predict"],(0,2)) != 0.
	batch_dict["data_to_predict"] = data_dict["data_to_predict"][:, non_missing_tp]
	batch_dict["tp_to_predict"] = data_dict["tp_to_predict"][non_missing_tp]

	if ("mask_predicted_data" in data_dict) and (data_dict["mask_predicted_data"] is not None):
		batch_dict["mask_predicted_data"] = data_dict["mask_predicted_data"][:, non_missing_tp]

	if ("labels" in data_dict) and (data_dict["labels"] is not None):
		batch_dict["labels"] = data_dict["labels"]
			
	batch_dict["mode"] = data_dict["mode"]
	return batch_dict

def get_ckpt_model(ckpt_path, model, device):
	if not os.path.exists(ckpt_path):
		raise Exception("Checkpoint " + ckpt_path + " does not exist.")
	# Load checkpoint.
	checkpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
	ckpt_args = checkpt['args']
	state_dict = checkpt['state_dict']
	model_dict = model.state_dict()

	# 1. filter out unnecessary keys
	state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
	# 2. overwrite entries in the existing state dict
	model_dict.update(state_dict) 
	# 3. load the new state dict
	model.load_state_dict(state_dict)
	model.to(device)

def update_learning_rate(optimizer, decay_rate = 0.999, lowest = 1e-3):
	for param_group in optimizer.param_groups:
		lr = param_group['lr']
		lr = max(lr * decay_rate, lowest)
		param_group['lr'] = lr

def linspace_vector(start, end, n_points):
	# start is either one value or a vector
	size = np.prod(start.size())

	assert(start.size() == end.size())
	if size == 1:
		# start and end are 1d-tensors
		res = torch.linspace(start, end, n_points)
	else:
		# start and end are vectors
		res = torch.Tensor()
		for i in range(0, start.size(0)):
			res = torch.cat((res, 
				torch.linspace(start[i], end[i], n_points)),0)
		res = torch.t(res.reshape(start.size(0), n_points))
	return res

def create_net(n_inputs, n_outputs, n_layers = 3,n_units = 100, nonlinear = nn.Tanh):
	layers = [nn.Linear(n_inputs, n_units)]

	for i in range(n_layers):
		layers.append(nonlinear())
		layers.append(nn.Linear(n_units, n_units))

	layers.append(nonlinear())
	layers.append(nn.Linear(n_units, n_outputs))
	return nn.Sequential(*layers)

def get_dict_template():
	return {"observed_data": None,
			"observed_tp": None,
			"data_to_predict": None,
			"tp_to_predict": None,
			"observed_mask": None,
			"mask_predicted_data": None,
			"labels": None
			}

def normalize_masked_data(data, mask, att_min, att_max):
    	# we don't want to divide by zero
	att_max[ att_max == 0.] = 1.

	if (att_max != 0.).all():
		data_norm = (data - att_min) / (att_max - att_min)
	else:
		raise Exception("Zero!")

	if torch.isnan(data_norm).any():
		raise Exception("nans!")

	# set masked out elements back to zero 
	data_norm[mask == 0] = 0

	return data_norm, att_min, att_max

def split_data_interp(data_dict):
	device = get_device(data_dict["data"])

	split_dict = {"observed_data": data_dict["data"].clone(),
				"observed_tp": data_dict["time_steps"].clone(),
				"data_to_predict": data_dict["data"].clone(),
				"tp_to_predict": data_dict["time_steps"].clone()}

	split_dict["observed_mask"] = None 
	split_dict["mask_predicted_data"] = None 
	split_dict["labels"] = None 

	if "mask" in data_dict and data_dict["mask"] is not None:
		split_dict["observed_mask"] = data_dict["mask"].clone()
		split_dict["mask_predicted_data"] = data_dict["mask"].clone()

	if ("labels" in data_dict) and (data_dict["labels"] is not None):
		split_dict["labels"] = data_dict["labels"].clone()

	split_dict["mode"] = "interp"
	return split_dict

def add_mask(data_dict):
	data = data_dict["observed_data"]
	mask = data_dict["observed_mask"]

	if mask is None:
		mask = torch.ones_like(data).to(get_device(data))

	data_dict["observed_mask"] = mask
	return data_dict

def split_and_subsample_batch(data_dict, args, data_type = "train"):
	if data_type == "train":
		# Training set
		processed_dict = split_data_interp(data_dict)

	else:
		# Test set
		processed_dict = split_data_interp(data_dict)

	# add mask
	processed_dict = add_mask(processed_dict)

	return processed_dict

def compute_loss_all_batches(model,
	test_dataloader, args,
	n_batches, experimentID, device,
	n_traj_samples = 1, kl_coef = 1., 
	max_samples_for_eval = None,
	tipo="train", n_experiment=0, reduced_features= None):


	total = {}
	total["loss"] = 0
	total["likelihood"] = 0
	total["mse"] = 0
	total["kl_first_p"] = 0
	total["std_first_p"] = 0
	total["pois_likelihood"] = 0
	total["ce_loss"] = 0

	n_test_batches = 0
	
	classif_predictions = torch.Tensor([]).to(device)
	all_test_labels =  torch.Tensor([]).to(device)

	

	for i in range(n_batches):
		print("Computing loss... " + str(i))
		
		batch_dict = get_next_batch(test_dataloader)
		
		results  = model.compute_all_losses(batch_dict, args,
			n_traj_samples = n_traj_samples, kl_coef = kl_coef, tipo="train", n_experiment=n_experiment)
		
		if args.classif:
			n_labels = model.n_labels
			n_traj_samples = results["label_predictions"].size(0)
			classif_predictions = torch.cat((classif_predictions, results["label_predictions"].reshape(n_traj_samples, -1, n_labels)),1)

			all_test_labels = torch.cat((all_test_labels, batch_dict["labels"].reshape(-1, n_labels)),1)
			
		for key in total.keys(): 
			if key in results:
				var = results[key]
				if isinstance(var, torch.Tensor):
					var = var.detach()
				total[key] += var
		
		
		n_test_batches += 1

	
	print("{:.6f}".format(total["loss"].detach()))


	print("n_test_batches: ", n_test_batches)
	if n_test_batches > 0:
		for key, value in total.items():
			total[key] = total[key] / n_test_batches
	
	if args.classif:
		if args.dataset == "fibrillation":
			#all_test_labels = all_test_labels.reshape(-1)
			# For each trajectory, we get n_traj_samples samples from z0 -- compute loss on all of them
			all_test_labels = all_test_labels.repeat(n_traj_samples,1,1)		

			if args.classif_tipe_multiclass:
				idx_not_nan = ~torch.isnan(all_test_labels).any(dim=(1, 2)) #añadimos anydim para que no me reduzca a un vector unidimensional y poder mantener las matrices
			else:
				idx_not_nan = ~torch.isnan(all_test_labels) # en caso de binaria si que podemos reducir a una unica dimension

			
			classif_predictions = classif_predictions[idx_not_nan]
			all_test_labels = all_test_labels[idx_not_nan]
			
			total["auc"] = 0.
			if torch.sum(all_test_labels) != 0.:
				if args.classif_tipe_multiclass:
					print("\tAUC-MULTICLASE:")	
					
					num_mortality_0 = 0
					num_mortality_1 = 0
					num_mortality_2 = 0

					for example_labels in all_test_labels:
						for label in example_labels:
							if torch.all(label == torch.tensor([0., 1., 0.]).to(device)):
								num_mortality_1 += 1
							elif torch.all(label == torch.tensor([1., 0., 0.]).to(device)):
								num_mortality_0 += 1
							elif torch.all(label == torch.tensor([0., 0., 1.]).to(device)):
								num_mortality_2 += 1

					print("Number of labeled examples: {}".format(len(all_test_labels.reshape(-1))))
					print("Number of examples with mortality 2: {}".format(num_mortality_2))
					print("Number of examples with mortality 1: {}".format(num_mortality_1))
					print("Number of examples with mortality 0: {}".format(num_mortality_0))


					all_test_labels = all_test_labels.reshape(-1, all_test_labels.size(-1))
					classif_predictions=classif_predictions.reshape(-1, classif_predictions.size(-1))


					#AUC
					auc = sk.metrics.roc_auc_score(all_test_labels.cpu().numpy(), classif_predictions.cpu().numpy(), multi_class='ovr')
					total["auc"]= auc                
					print("\t\tAUC: ", auc)


				else:
					print("\tAUC-BINARIA:")	
					print("\t\tNumber of labeled examples: {}".format(len(all_test_labels.reshape(-1))))
					print("\t\tNumber of examples with mortality 1: {}".format(torch.sum(all_test_labels == 1.)))
							
					# Cannot compute AUC with only 1 class
					
					auc = sk.metrics.roc_auc_score(all_test_labels.cpu().numpy().reshape(-1),   ##CAMBIAR AQUI
						classif_predictions.cpu().numpy().reshape(-1), multi_class='ovr') #añdimos "Multiclas=ovr"
					total["auc"]= auc                
					print("\t\tAUC: ", auc)
			
			else:
				print("Warning: Couldn't compute AUC -- all examples are from the same class")
		
	return total

def check_mask(data, mask):
	#check that "mask" argument indeed contains a mask for data
	n_zeros = torch.sum(mask == 0.).cpu().numpy()
	n_ones = torch.sum(mask == 1.).cpu().numpy()

	# mask should contain only zeros and ones
	assert((n_zeros + n_ones) == np.prod(list(mask.size())))

	# all masked out elements should be zeros
	assert(torch.sum(data[mask == 0.] != 0.) == 0)





