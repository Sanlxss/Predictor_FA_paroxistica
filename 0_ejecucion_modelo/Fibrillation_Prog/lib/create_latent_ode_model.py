###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import os
import numpy as np

import torch
import torch.nn as nn
from torch.nn.functional import relu

import lib.utils as utils
from lib.latent_ode import LatentODE
from lib.encoder_decoder import *
from lib.diffeq_solver import DiffeqSolver

from torch.distributions.normal import Normal
from lib.ode_func import ODEFunc

#####################################################################################################

def create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device, 
	classif_per_tp = False, n_labels = 3):

	dim = args.latents
		
	print("\n-----------------------------\n"+
	   "Empezamos el CREATE_NET\n-------------------------------")
	print("\tDimension Input: ", input_dim)
	print("\tTam Estado Latente: ", args.latents)
	print("\tProfundidad Modelo Generacion: ", args.gen_layers)
	print("\tNum Unidades Capa ODE: ", args.units)

	ode_func_net = utils.create_net(dim, args.latents, n_layers = args.gen_layers, n_units = args.units, nonlinear = nn.Tanh)

	
	gen_ode_func = ODEFunc(input_dim = input_dim, latent_dim = args.latents, ode_func_net = ode_func_net, device = device).to(device)

	z0_diffeq_solver = None
	n_rec_dims = args.rec_dims
	enc_input_dim = int(input_dim) * 2 # we concatenate the mask
	gen_data_dim = input_dim
	
	z0_dim = args.latents
	#print("Z0_dim: ", z0_dim)
	if args.z0_encoder == "odernn":
		ode_func_net = utils.create_net(n_rec_dims, n_rec_dims,	n_layers = args.rec_layers, n_units = args.units, nonlinear = nn.Tanh)
		
		rec_ode_func = ODEFunc(	input_dim = enc_input_dim, latent_dim = n_rec_dims,	ode_func_net = ode_func_net, device = device).to(device)

		z0_diffeq_solver = DiffeqSolver(enc_input_dim, rec_ode_func, "euler", args.latents,	odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)
		
		encoder_z0 = Encoder_z0_ODE_RNN(n_rec_dims, enc_input_dim, z0_diffeq_solver, z0_dim = z0_dim, n_gru_units = args.gru_units, device = device).to(device)

	else:
		raise Exception("Unknown encoder for Latent ODE model: " + args.z0_encoder)

	decoder = Decoder(args.latents, gen_data_dim).to(device)

	diffeq_solver = DiffeqSolver(gen_data_dim, gen_ode_func, 'dopri5', args.latents, odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)

	model = LatentODE(
		input_dim = gen_data_dim, 
		latent_dim = args.latents, 
		encoder_z0 = encoder_z0, 
		decoder = decoder, 
		diffeq_solver = diffeq_solver, 
		z0_prior = z0_prior, 
		device = device,
		obsrv_std = obsrv_std, 
		use_classif = args.classif,
		classif_per_tp = classif_per_tp,
		n_labels = n_labels,
		train_classif_w_reconstr = (args.dataset == "fibrillation")
		).to(device)

	return model
