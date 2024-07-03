import os
import pandas as pd
import glob
import math
import random
import lib.utils as utils
import numpy as np
import torch
from lib.utils import get_device


from tqdm import tqdm

# get minimum and maximum for each feature across the whole dataset
def get_data_min_max(records):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	data_min, data_max = None, None
	inf = torch.Tensor([float("Inf")])[0].to(device)

	for b, (record_id, tt, vals, mask, labels) in enumerate(records):
		n_features = vals.size(-1)

		batch_min = []
		batch_max = []
		for i in range(n_features):
			non_missing_vals = vals[:,i][mask[:,i] == 1]
			if len(non_missing_vals) == 0:
				batch_min.append(inf)
				batch_max.append(-inf)
			else:
				batch_min.append(torch.min(non_missing_vals))
				batch_max.append(torch.max(non_missing_vals))

		batch_min = torch.stack(batch_min)
		batch_max = torch.stack(batch_max)

		if (data_min is None) and (data_max is None):
			data_min = batch_min
			data_max = batch_max
		else:
			data_min = torch.min(data_min, batch_min)
			data_max = torch.max(data_max, batch_max)

	return data_min, data_max


class Fibrillation(object):
	params=[]
	
	
	def __init__(self, root, train=True, download=False, device = torch.device("cpu"), classif_tipe_multiclass=True, 
		rutaBD="", selected_features=[]):

		self.root = root
		self.train = train
		self.reduce = "average"
		self.classif_tipe_multiclass = classif_tipe_multiclass

		self.rutaBD = rutaBD
		self.selected_features = selected_features

		if self.classif_tipe_multiclass:
			self.labels = [ "SRR", "FA_persistente", "FA_paroxistico" ]
			self.labels_dict = {k: i for i, k in enumerate(self.labels)}
			if download:
				self.download_multiclass(device)

		else:
			self.labels = [ "FA" ]
			self.labels_dict = {k: i for i, k in enumerate(self.labels)}
			if download:
				self.download_binary(device)

	def download_multiclass(self, device):
		num_ficheros_cargados=[0,0,0]
		total_dataset = []
		random.seed(41)
		features = self.selected_features

		f = pd.read_csv(self.rutaBD+'1_SR.csv')
		columnas = f.columns

		# Crear una lista de nombres de columnas correspondientes a las posiciones en features
		nombres_features = [columnas[indice] for indice in features]
		self.params = nombres_features
        
		files = glob.glob(self.rutaBD+'*.csv')
		files = sorted(files)
		for filename in tqdm(files, desc="Cargando Datos: "):
			record_id = filename.split('/')[2].split('\\')[1].split('.')[0]
			#record_id = filename.split('/')[3].split('.')[0] #para linux se usa este

			if 'FA_paroxistico' in record_id: 
				label = 2
			elif 'FA_persistente' in record_id:
				label = 1
			else:
				label = 0


			f = pd.read_csv(filename)
			longitud = f['tiempo'].size
			tt = []
			offset = random.randrange(0,100,1)
			time = 0
			for i in range(longitud):
				time = f['tiempo'][i]
				tt.append(time+offset)
			tt = torch.tensor(tt).to(device)
			
            
			vals = np.array(f[columnas[features]])
			mask = np.ones(vals.shape)
			fil = vals.shape[0]
			col = vals.shape[1]
			for i in range(fil):
				for j in range(col):
					if math.isnan(vals[i,j]):
						mask[i,j] = 0
						vals[i,j] = 0
			vals = torch.tensor(vals).type(torch.float32).to(device)
			
			mask = torch.tensor(mask).type(torch.float32).to(device)
			label = torch.tensor(label).to(device)
			total_dataset.append((record_id, tt, vals, mask, label))

			num_ficheros_cargados[label]+=1
			
		print(f"Se han cargado un total de {num_ficheros_cargados} pacientes")
		print(f"\t Pacientes Normales: {num_ficheros_cargados[0]}")
		print(f"\t Pacientes Paroxisticos: {num_ficheros_cargados[1]}")
		print(f"\t Pacientes Persistentes: {num_ficheros_cargados[2]}")
		self.data = total_dataset
        
	def download_binary(self, device):
		num_ficheros_cargados=[0,0]
		random.seed(41)
		total_dataset = []
		features = self.selected_features
		f = pd.read_csv(self.rutaBD+'1_SR.csv')
		columnas = f.columns

		# Crear una lista de nombres de columnas correspondientes a las posiciones en features
		nombres_features = [columnas[indice] for indice in features]
		self.params = nombres_features
        
		files = glob.glob(self.rutaBD+'*.csv')
		files = sorted(files)
		for filename in tqdm(files, desc="Cargando Datos: "):
			#record_id = filename.split('/')[2].split('\\')[1].split('.')[0]
			record_id = filename.split('/')[3].split('.')[0] #para linux se usa este
			
			if 'FA' in record_id:
				label = 1
			else:
				label = 0


			f = pd.read_csv(filename)
			longitud = f['tiempo'].size 
			

			tt = []
			offset = random.randrange(0,100,1)
			time = 0
			for i in range(longitud):
				time = f['tiempo'][i]
				tt.append(time+offset)
			tt = torch.tensor(tt).to(device)
			
            
			vals = np.array(f[columnas[features]])
			mask = np.ones(vals.shape)
			fil = vals.shape[0]
			col = vals.shape[1]
			for i in range(fil):
				for j in range(col):
					if math.isnan(vals[i,j]):
						mask[i,j] = 0
						vals[i,j] = 0
			vals = torch.tensor(vals).type(torch.float32).to(device)
			
			mask = torch.tensor(mask).type(torch.float32).to(device)
			label = torch.tensor(label).to(device)
			total_dataset.append((record_id, tt, vals, mask, label))


			num_ficheros_cargados[label]+=1

		print(f"Se han cargado un total de {num_ficheros_cargados} pacientes")
		print(f"\t Pacientes Normales: {num_ficheros_cargados[0]}")
		print(f"\t Pacientes Fibrilantes: {num_ficheros_cargados[1]}")

		self.data = total_dataset
		

	def __getitem__(self, index):
		return self.data[index]

	def __len__(self):
		return len(self.data)

	def get_label(self, record_id):
		return self.labels[record_id]
	

def variable_time_collate_fn(batch, args, device = torch.device("cpu"), data_type = "train", 
	data_min = None, data_max = None):
	"""
	Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
		- record_id is a patient id
		- tt is a 1-dimensional tensor containing T time values of observations.
		- vals is a (T, D) tensor containing observed values for D variables.
		- mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
		- labels is a list of labels for the current patient, if labels are available. Otherwise None.
	Returns:
		combined_tt: The union of all time observations.
		combined_vals: (M, T, D) tensor containing the observed values.
		combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
	"""
	D = batch[0][2].shape[1]
	combined_tt, inverse_indices = torch.unique(torch.cat([ex[1] for ex in batch]), sorted=True, return_inverse=True)
	combined_tt = combined_tt.to(device)

	offset = 0
	combined_vals = torch.zeros([len(batch), len(combined_tt), D]).to(device)
	combined_mask = torch.zeros([len(batch), len(combined_tt), D]).to(device)
	
	combined_labels = None
	if args.classif_tipe_multiclass:
		N_labels = 3 #Modificamos el numero de etiquetas
	else:
		N_labels = 1

	combined_labels = torch.zeros(len(batch), N_labels) + torch.tensor(float('nan'))
	combined_labels = combined_labels.to(device = device)

	
	for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
		tt = tt.to(device)
		vals = vals.to(device)
		mask = mask.to(device)
		if labels is not None:
			labels = labels.to(device)

		
		indices = inverse_indices[offset:offset + len(tt)]
		#print("INDICES: ", indices, record_id)
		
		offset += len(tt)

		combined_vals[b, indices] = vals
		#print(combined_vals[b])
		combined_mask[b, indices] = mask
		#print(combined_mask[b])

		if labels is not None and N_labels==3:
			# Codificar las etiquetas de manera one-hot
			one_hot_labels = torch.zeros(N_labels).to(device)
			one_hot_labels[labels] = 1
			combined_labels[b] = one_hot_labels

		elif labels is not None and N_labels==1:
			combined_labels[b] = labels


	combined_vals, _, _ = utils.normalize_masked_data(combined_vals, combined_mask, 
		att_min = data_min, att_max = data_max)


	if torch.max(combined_tt) != 0.:
		combined_tt = combined_tt / torch.max(combined_tt)
		
	data_dict = {
		"data": combined_vals, 
		"time_steps": combined_tt,
		"mask": combined_mask,
		"labels": combined_labels}
	
	

	data_dict = utils.split_and_subsample_batch(data_dict, args, data_type = data_type)


	return data_dict