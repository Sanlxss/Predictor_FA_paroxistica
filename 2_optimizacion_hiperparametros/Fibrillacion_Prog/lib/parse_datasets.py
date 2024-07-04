import os
import numpy as np

import torch
import torch.nn as nn

import lib.utils as utils
from torch.utils.data import DataLoader
from fibrillation import Fibrillation, variable_time_collate_fn, get_data_min_max

from sklearn import model_selection
from sklearn.model_selection import KFold

#####################################################################################################
def parse_datasets(args, device, n_folds, rutaBD, selected_features):
	
	dataset_name = args.dataset

	##################################################################
	# Fibrillation dataset
	if dataset_name == "fibrillation":
		train_dataset_obj = Fibrillation('data/Fibrillation', train=True, 
										download=True,
										device = device, 
										classif_tipe_multiclass=args.classif_tipe_multiclass,
										rutaBD=rutaBD, selected_features=selected_features)
		
		
		etiquetas = torch.tensor([label.item() for _, _, _, _, label in train_dataset_obj.data])
		etiquetas_unicas = torch.unique(etiquetas)
		num_etiquetas = len(etiquetas_unicas)
		
		# Combine and shuffle samples from fibrillation Train and fibrillation Test
		total_dataset = train_dataset_obj[:len(train_dataset_obj)]

		# Shuffle and split
		train_data, test_data = model_selection.train_test_split(total_dataset, train_size= 0.8,  random_state = 42, shuffle = True)
		
		print("TAMAÑO TRAIN: ", len(train_data))
		print("TAMAÑO TEST: ", len(test_data))
		
		
		###########################################
		kf = model_selection.StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)


		print("\tIniciando Folds")
		id_f = 0
		for id_f, (train_indices, test_indices) in enumerate(kf.split(train_data, [etiqueta.item() for _, _, _, _, etiqueta in train_data])):
			print(f'\n\t\t\t\t\t\t---------\n\t\t\t\t\t\t\tFOLD {id_f}')
			print(f'\t\t\t\t\t\tTRIAIN INDICES: {train_indices}')
			print(f'\t\t\t\t\t\tTEST INDICES: {test_indices}')

			train_data_fold = [total_dataset[i] for i in train_indices]
			test_data_fold = [total_dataset[i] for i in test_indices]

			print(f"\t\t\t\t\t\tTAMAÑO TRAIN (fold->{id_f}): {len(train_data_fold)}")
			print(f"\t\t\t\t\t\tTAMAÑO TEST (fold->{id_f}): {len(test_data_fold)}")
			print("\n\n\n\n\n\n")
			record_id, tt, vals, mask, labels = train_data[0]		

			n_samples = len(total_dataset)

			input_dim = vals.size(-1)
			
			batch_size = min(min(len(train_dataset_obj), args.batch_size), args.n)

			
			#Se obtiene el valor Minimo/Maximo que hay para cada variable que tenemos en cuenta en el modelo.
			data_min, data_max = get_data_min_max(total_dataset)

					
			
			train_dataloader = DataLoader(train_data, batch_size= batch_size, shuffle=False, 
				collate_fn= lambda batch: variable_time_collate_fn(batch, args, device, data_type = "train",
					data_min = data_min, data_max = data_max))
			test_dataloader = DataLoader(test_data, batch_size = n_samples, shuffle=False, 
				collate_fn= lambda batch: variable_time_collate_fn(batch, args, device, data_type = "test",
					data_min = data_min, data_max = data_max))
			

			if args.classif_tipe_multiclass == False: #en el caso de clasificiacion binaria solo se hace una etiqueta 
				num_etiquetas -=1	#se usa un array de len(1) y se pone [0] o [1], en vez de tener un array de len(2)
			

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
			
			print("\tADDED")
			yield data_objects
