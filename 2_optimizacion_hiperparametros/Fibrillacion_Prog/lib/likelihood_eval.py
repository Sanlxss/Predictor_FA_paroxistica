import gc
import numpy as np
import sklearn as sk
import numpy as np
#import gc
import torch
import torch.nn as nn
from torch.nn.functional import relu

import lib.utils as utils
from lib.utils import get_device
from lib.encoder_decoder import *
from lib.likelihood_eval import *

from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence, Independent


def gaussian_log_likelihood(mu_2d, data_2d, obsrv_std, indices = None):
	n_data_points = mu_2d.size()[-1]

	if n_data_points > 0:
		gaussian = Independent(Normal(loc = mu_2d, scale = obsrv_std.repeat(n_data_points)), 1)
		log_prob = gaussian.log_prob(data_2d) 
		log_prob = log_prob / n_data_points 
	else:
		log_prob = torch.zeros([1]).to(get_device(data_2d)).squeeze()
	return log_prob


def compute_binary_CE_loss(label_predictions, mortality_label, iter_batch, tipo="train", n_experiment=0):

	mortality_label = mortality_label.reshape(-1)

	if len(label_predictions.size()) == 1:
		label_predictions = label_predictions.unsqueeze(0)
 
	n_traj_samples = label_predictions.size(0)
	label_predictions = label_predictions.reshape(n_traj_samples, -1)
	
	idx_not_nan = ~torch.isnan(mortality_label)
	if len(idx_not_nan) == 0.:
		print("All are labels are NaNs!")
		ce_loss = torch.Tensor(0.).to(get_device(mortality_label))

	label_predictions = label_predictions[:,idx_not_nan]
	mortality_label = mortality_label[idx_not_nan]

	if torch.sum(mortality_label == 0.) == 0 or torch.sum(mortality_label == 1.) == 0:
		print("Warning: all examples in a batch belong to the same class -- please increase the batch size.")

	assert(not torch.isnan(label_predictions).any())
	assert(not torch.isnan(mortality_label).any())

	# For each trajectory, we get n_traj_samples samples from z0 -- compute loss on all of them
	mortality_label = mortality_label.repeat(n_traj_samples, 1)
	ce_loss = nn.BCEWithLogitsLoss()(label_predictions, mortality_label)

	# divide by number of patients in a batch
	ce_loss = ce_loss / n_traj_samples
	print("\t\tCE_LOSS: {:.6f}".format(ce_loss))


	# Guardar ce_loss en un archivo
	if tipo == "train":
		nombre_archivo = "ficheros_resultados/"+str(n_experiment)+"/ce_loss_"+str(tipo)+".txt"
		with open(nombre_archivo, 'a') as archivo:
			archivo.write(f"Batch: {iter_batch}, CE Loss: {ce_loss}\n")
	elif tipo =="test":
		nombre_archivo = "ficheros_resultados/"+str(n_experiment)+"/ce_loss_"+str(tipo)+".txt"
		with open(nombre_archivo, 'a') as archivo:
			archivo.write(f"Batch: {iter_batch}, CE Loss: {ce_loss}\n")


	return ce_loss


def compute_multiclass_CE_loss(label_predictions, true_label, mask, iter_batch, tipo="train", n_experiment=0):

	# Solo consideramos la clasificación en el estado inicial, por lo tanto, desechamos las dimensiones relacionadas con los tiempos de predicción
	n_traj_samples, n_traj, n_dims = label_predictions.size()
	n_traj_samples, n_traj, n_dims = label_predictions.size()

    # Repetimos las etiquetas verdaderas para que tengan la misma forma que las predicciones de etiquetas
	true_label = true_label.repeat(n_traj_samples, 1, 1)

    # Aplanamos las predicciones y las etiquetas verdaderas para facilitar el cálculo de la pérdida
	label_predictions = label_predictions.reshape(n_traj_samples * n_traj, n_dims)
	true_label = true_label.reshape(n_traj_samples * n_traj, n_dims)

    # Calculamos la pérdida de entropía cruzada
	ce_loss = nn.CrossEntropyLoss()(label_predictions, true_label)
	# divide by number of patients in a batch
	ce_loss = ce_loss / n_traj_samples

	print("\t\tCE_LOSS: ", ce_loss)

	# Guardar ce_loss en un archivo
	if tipo == "train":
		nombre_archivo = "ficheros_resultados/"+str(n_experiment)+"/ce_loss_"+str(tipo)+".txt"
		with open(nombre_archivo, 'a') as archivo:
			archivo.write(f"Batch: {iter_batch}, CE Loss: {ce_loss}\n")
	elif tipo =="test":
		nombre_archivo = "ficheros_resultados/"+str(n_experiment)+"/ce_loss_"+str(tipo)+".txt"
		with open(nombre_archivo, 'a') as archivo:
			archivo.write(f"Batch: {iter_batch}, CE Loss: {ce_loss}\n")
    			
	
	
	return ce_loss


def compute_masked_likelihood(mu, data, mask, likelihood_func):
	# Compute the likelihood per patient and per attribute so that we don't priorize patients with more measurements
	n_traj_samples, n_traj, n_timepoints, n_dims = data.size()

	res = []
	for i in range(n_traj_samples):
		for k in range(n_traj):
			for j in range(n_dims):
				data_masked = torch.masked_select(data[i,k,:,j], mask[i,k,:,j].bool())
				
				#assert(torch.sum(data_masked == 0.) < 10)

				mu_masked = torch.masked_select(mu[i,k,:,j], mask[i,k,:,j].bool())
				log_prob = likelihood_func(mu_masked, data_masked, indices = (i,k,j))
				res.append(log_prob)
	# shape: [n_traj*n_traj_samples, 1]

	res = torch.stack(res, 0).to(get_device(data))
	res = res.reshape((n_traj_samples, n_traj, n_dims))
	# Take mean over the number of dimensions
	res = torch.mean(res, -1) # !!!!!!!!!!! changed from sum to mean
	res = res.transpose(0,1)
	return res


def masked_gaussian_log_density(mu, data, obsrv_std, mask = None):
	# these cases are for plotting through plot_estim_density
	if (len(mu.size()) == 3):
		# add additional dimension for gp samples
		mu = mu.unsqueeze(0)

	if (len(data.size()) == 2):
		# add additional dimension for gp samples and time step
		data = data.unsqueeze(0).unsqueeze(2)
	elif (len(data.size()) == 3):
		# add additional dimension for gp samples
		data = data.unsqueeze(0)

	n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()

	assert(data.size()[-1] == n_dims)

	# Shape after permutation: [n_traj, n_traj_samples, n_timepoints, n_dims]
	if mask is None:
		mu_flat = mu.reshape(n_traj_samples*n_traj, n_timepoints * n_dims)
		n_traj_samples, n_traj, n_timepoints, n_dims = data.size()
		data_flat = data.reshape(n_traj_samples*n_traj, n_timepoints * n_dims)
	
		res = gaussian_log_likelihood(mu_flat, data_flat, obsrv_std)
		res = res.reshape(n_traj_samples, n_traj).transpose(0,1)
	else:
		# Compute the likelihood per patient so that we don't priorize patients with more measurements
		func = lambda mu, data, indices: gaussian_log_likelihood(mu, data, obsrv_std = obsrv_std, indices = indices)
		res = compute_masked_likelihood(mu, data, mask, func)
	return res


def mse(mu, data, indices = None):
	n_data_points = mu.size()[-1]

	if n_data_points > 0:
		mse = nn.MSELoss()(mu, data)
	else:
		mse = torch.zeros([1]).to(get_device(data)).squeeze()
	return mse


def compute_mse(mu, data, mask = None):
	# these cases are for plotting through plot_estim_density
	if (len(mu.size()) == 3):
		# add additional dimension for gp samples
		mu = mu.unsqueeze(0)

	if (len(data.size()) == 2):
		# add additional dimension for gp samples and time step
		data = data.unsqueeze(0).unsqueeze(2)
	elif (len(data.size()) == 3):
		# add additional dimension for gp samples
		data = data.unsqueeze(0)

	n_traj_samples, n_traj, n_timepoints, n_dims = mu.size()
	assert(data.size()[-1] == n_dims)

	# Shape after permutation: [n_traj, n_traj_samples, n_timepoints, n_dims]
	if mask is None:
		mu_flat = mu.reshape(n_traj_samples*n_traj, n_timepoints * n_dims)
		n_traj_samples, n_traj, n_timepoints, n_dims = data.size()
		data_flat = data.reshape(n_traj_samples*n_traj, n_timepoints * n_dims)
		res = mse(mu_flat, data_flat)
	else:
		# Compute the likelihood per patient so that we don't priorize patients with more measurements
		res = compute_masked_likelihood(mu, data, mask, mse)
	return res


	

