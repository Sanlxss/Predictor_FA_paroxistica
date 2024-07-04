import os
import numpy as np

import torch
import torch.nn as nn

import lib.utils as utils
from torch.utils.data import DataLoader
from fibrillation import Fibrillation, variable_time_collate_fn, get_data_min_max

from sklearn import model_selection
from imblearn.over_sampling import SMOTE

#####################################################################################################
def parse_datasets(args, device):
	
	dataset_name = args.dataset

	##################################################################
	# Fibrillation dataset
	if dataset_name == "fibrillation":
		train_dataset_obj = Fibrillation('data/Fibrillation', train=True, 
										download=True, 
										device = device, 
										classif_tipe_multiclass=args.classif_tipe_multiclass)
		
		
		etiquetas = torch.tensor([label.item() for _, _, _, _, label in train_dataset_obj.data])
		etiquetas_unicas = torch.unique(etiquetas)
		num_etiquetas = len(etiquetas_unicas)
		print("NUMERO DE ETIQUETAS CLASIFICACION: ", num_etiquetas)
		
		# Combine and shuffle samples from Fibrillation Train and Fibrillation Test
		total_dataset = train_dataset_obj[:len(train_dataset_obj)]
		
		# Shuffle and split
		train_data, test_data = model_selection.train_test_split(total_dataset, train_size= 0.8,  random_state = 42, shuffle = True)
		
		record_id, tt, vals, mask, labels = train_data[0]
				

		n_samples = len(total_dataset)
		print("Numero de pacientes: ",n_samples)

		input_dim = vals.size(-1)
		print("Numero caracteristicas ECG: ",input_dim)
		
		batch_size = min(min(len(train_dataset_obj), args.batch_size), args.n)
		print("Tamaño del BATCH: ", batch_size)

		#Se obtiene el valor Minimo/Maximo que hay para cada variable que tenemos en cuenta en el modelo.
		data_min, data_max = get_data_min_max(total_dataset)
					
		test_batch_size = 1 #n_samples

		train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn(batch, args, device, data_type = "train",
				data_min = data_min, data_max = data_max))
		test_dataloader = DataLoader(test_data, batch_size = test_batch_size, shuffle=False, 
			collate_fn= lambda batch: variable_time_collate_fn(batch, args, device, data_type = "test",
				data_min = data_min, data_max = data_max))
		
		if args.classif_tipe_multiclass == False: #en el caso de clasificiacion binaria solo se hace una etiqueta 
			num_etiquetas -=1	#se usa un array de len(1) y se pone [0] o [1], en vez de tener un array de len(2)
		print("Len etiquetas, para modelo: ", num_etiquetas)
		

		attr_names = train_dataset_obj.params
		data_objects = {"dataset_obj": train_dataset_obj, 
					"train_dataloader": utils.inf_generator(train_dataloader), 
					"test_dataloader": utils.inf_generator(test_dataloader),
					"input_dim": input_dim,
					"n_train_batches": len(train_dataloader),
					"n_test_batches": len(test_dataloader),
					"attr": attr_names, #optional
					"classif_per_tp": False, #optional
					"n_labels": num_etiquetas} #optional  #CAMBIAMOS AQUI A 3 el NUMERO DE ETIQUETAS


		return data_objects
