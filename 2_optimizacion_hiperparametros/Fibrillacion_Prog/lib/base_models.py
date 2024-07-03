import psutil
import torch
import torch.nn as nn
from torch.nn.functional import relu

import lib.utils as utils
from lib.encoder_decoder import *
from lib.likelihood_eval import *

from torch.distributions.normal import Normal

from torch.distributions.normal import Normal


def create_classifier(z0_dim, n_labels): ##Función que crea el Clasificador.
	return nn.Sequential(
			nn.Linear(z0_dim, 300),
			nn.ReLU(),
			nn.Linear(300, 300),
			nn.ReLU(),
			nn.Linear(300, n_labels),)

class VAE_Baseline(nn.Module):
	def __init__(self, input_dim, latent_dim, 
		z0_prior, device,
		obsrv_std = 0.01, 
		use_binary_classif = False,
		classif_per_tp = False,
		use_poisson_proc = False,
		linear_classifier = False,
		n_labels = 3,
		train_classif_w_reconstr = False):

		super(VAE_Baseline, self).__init__()
		
		self.input_dim = input_dim
		self.latent_dim = latent_dim
		self.device = device
		self.n_labels = n_labels

		self.obsrv_std = torch.Tensor([obsrv_std]).to(device)

		self.z0_prior = z0_prior
		self.use_binary_classif = use_binary_classif
		self.classif_per_tp = classif_per_tp
		self.use_poisson_proc = use_poisson_proc
		self.linear_classifier = linear_classifier
		self.train_classif_w_reconstr = train_classif_w_reconstr

		z0_dim = latent_dim
		if use_poisson_proc:
			z0_dim += latent_dim

		if use_binary_classif: 
			if linear_classifier:
				self.classifier = nn.Sequential(
					nn.Linear(z0_dim, n_labels))
			else:
				self.classifier = create_classifier(z0_dim, n_labels)
			utils.init_network_weights(self.classifier)


	def get_gaussian_likelihood(self, truth, pred_y, mask = None):
		n_traj, n_tp, n_dim = truth.size()
		
		truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)


		if mask is not None:
			mask = mask.repeat(pred_y.size(0), 1, 1, 1)
		log_density_data = masked_gaussian_log_density(pred_y, truth_repeated, 
			obsrv_std = self.obsrv_std, mask = mask)
		log_density_data = log_density_data.permute(1,0)
		log_density = torch.mean(log_density_data, 1)

		return log_density


	def get_mse(self, truth, pred_y, mask = None):
		n_traj, n_tp, n_dim = truth.size()

		# Compute likelihood of the data under the predictions
		truth_repeated = truth.repeat(pred_y.size(0), 1, 1, 1)
		
		if mask is not None:
			mask = mask.repeat(pred_y.size(0), 1, 1, 1)

		# Compute likelihood of the data under the predictions
		log_density_data = compute_mse(pred_y, truth_repeated, mask = mask)

		return torch.mean(log_density_data)


	def compute_all_losses(self, batch_dict, args, n_traj_samples = 3, kl_coef = 1., iter_batch=0, tipo="train", n_experiment=0): #EN TRAIN USAMOS ESTE COMPUTE ALL LOSES
		# Condition on subsampled points
		# Make predictions for all the points

		if args.classif_tipe_multiclass:
			pred_y, info = self.get_reconstruction(batch_dict["tp_to_predict"].float(), #Aqui añadimos la conversion a float para que siempre sea float el vector.->ESto en caso de BINARIO no se hace
					batch_dict["observed_data"], batch_dict["observed_tp"], 
					mask = batch_dict["observed_mask"], n_traj_samples = n_traj_samples,
					mode = batch_dict["mode"])
		else:
			pred_y, info = self.get_reconstruction(batch_dict["tp_to_predict"], 
				batch_dict["observed_data"], batch_dict["observed_tp"], 
				mask = batch_dict["observed_mask"], n_traj_samples = n_traj_samples,
				mode = batch_dict["mode"])		
		
		print("\tget_reconstruction done -- computing likelihood")

		fp_mu, fp_std, fp_enc = info["first_point"]
		fp_std = fp_std.abs()
		try:
			fp_distr = Normal(fp_mu, fp_std)
			assert(torch.sum(fp_std < 0) == 0.)
			kldiv_z0 = kl_divergence(fp_distr, self.z0_prior)
			ex = False
		except ValueError as e:
			fp_distr = -1
			kldiv_z0 = torch.full_like(fp_mu, fill_value=-1.0*float('inf'))
			ex=True	

		if torch.isnan(kldiv_z0).any():
			raise Exception("kldiv_z0 is Nan!")
		

		# Mean over number of latent dimensions
		# kldiv_z0 shape: [n_traj_samples, n_traj, n_latent_dims] if prior is a mixture of gaussians (KL is estimated)
		# kldiv_z0 shape: [1, n_traj, n_latent_dims] if prior is a standard gaussian (KL is computed exactly)
		# shape after: [n_traj_samples]
		kldiv_z0 = torch.mean(kldiv_z0,(1,2))
		

		# Compute likelihood of all the points
		rec_likelihood = self.get_gaussian_likelihood(
			batch_dict["data_to_predict"], pred_y,
			mask = batch_dict["mask_predicted_data"])
		
		mse = self.get_mse(
			batch_dict["data_to_predict"], pred_y,
			mask = batch_dict["mask_predicted_data"])

		pois_log_likelihood = torch.Tensor([0.]).to(get_device(batch_dict["data_to_predict"]))



	
		################################
		# Compute CE loss for binary classification on Fibrillation
		device = get_device(batch_dict["data_to_predict"])
		ce_loss = torch.Tensor([0.]).to(device)



		if (batch_dict["labels"] is not None) and self.use_binary_classif:
			if (batch_dict["labels"].size(-1) == 1) or (len(batch_dict["labels"].size()) == 1): 
				print("\tPERDIDA_BINARIO")
				ce_loss = compute_binary_CE_loss(
					info["label_predictions"], 
					batch_dict["labels"],
					iter_batch=iter_batch,
					tipo=tipo,
					n_experiment=n_experiment)
			else:
				print("\tPERDIDA_MULTICLASS")
				ce_loss = compute_multiclass_CE_loss(
					info["label_predictions"], 
					batch_dict["labels"],
					mask = batch_dict["mask_predicted_data"], 
					iter_batch=iter_batch,
					tipo=tipo,
					n_experiment=n_experiment)

  
		# IWAE loss
		loss = - torch.logsumexp(rec_likelihood -  kl_coef * kldiv_z0,0)
		if torch.isnan(loss):
			loss = - torch.mean(rec_likelihood - kl_coef * kldiv_z0,0)
		if self.use_poisson_proc:
			loss = loss - 0.1 * pois_log_likelihood 
		if self.use_binary_classif:
			if self.train_classif_w_reconstr:
				loss = loss +  ce_loss * 100
			else:
				loss =  ce_loss
		
		if ex==True:
			loss = torch.zeros_like(loss).detach()
		
		results = {}
		results["loss"] = torch.mean(loss)
		results["likelihood"] = torch.mean(rec_likelihood).detach()
		results["mse"] = torch.mean(mse).detach()
		results["pois_likelihood"] = torch.mean(pois_log_likelihood).detach()
		results["ce_loss"] = torch.mean(ce_loss).detach()
		results["kl_first_p"] =  torch.mean(kldiv_z0).detach()
		results["std_first_p"] = torch.mean(fp_std).detach()

		print("\tLOSS: ", results["loss"])
		
		if batch_dict["labels"] is not None and self.use_binary_classif:
			results["label_predictions"] = info["label_predictions"].detach()
		return results



