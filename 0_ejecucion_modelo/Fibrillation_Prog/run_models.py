import os
import sys

import time
import argparse
import numpy as np
from random import SystemRandom
import sklearn as sk

import torch
from torch.nn.functional import relu
import torch.optim as optim
import torch.nn.functional as F

import lib.utils as utils
from torch.distributions.normal import Normal

#from lib.rnn_baselines import *
from lib.create_latent_ode_model import create_LatentODE_model
from lib.parse_datasets import parse_datasets

from lib.utils import compute_loss_all_batches

# Generative model for noisy data based on ODE
parser = argparse.ArgumentParser('Latent ODE')
parser.add_argument('-n',  type=int, default=100, help="Size of the dataset")
parser.add_argument('--niters', type=int, default=300)
parser.add_argument('--lr',  type=float, default=1e-2, help="Starting learning rate.")
parser.add_argument('-b', '--batch-size', type=int, default=50)

parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="ID of the experiment to load for evaluation. If None, run a new experiment.")
parser.add_argument('-r', '--random-seed', type=int, default=1991, help="Random_seed")

parser.add_argument('--dataset', type=str, default='fibrillation', help="Dataset to load. Available: fibrillation")
parser.add_argument('-s', '--sample-tp', type=float, default=None, help="Number of time points to sub-sample."
	"If > 1, subsample exact number of points. If the number is in [0,1], take a percentage of available points per time series. If None, do not subsample")

parser.add_argument('-c', '--cut-tp', type=int, default=None, help="Cut out the section of the timeline of the specified length (in number of points)."
	"Used for periodic function demo.")

parser.add_argument('--latent-ode', action='store_true', help="Run Latent ODE seq2seq model")
parser.add_argument('--z0-encoder', type=str, default='odernn', help="Type of encoder for Latent ODE model: odernn or rnn")

parser.add_argument('--classic-rnn', action='store_true', help="Run RNN baseline: classic RNN that sees true points at every point. Used for interpolation only.")
parser.add_argument('--rnn-cell', default="gru", help="RNN Cell type. Available: gru (default), expdecay")
parser.add_argument('--input-decay', action='store_true', help="For RNN: use the input that is the weighted average of impirical mean and previous value (like in GRU-D)")

parser.add_argument('--ode-rnn', action='store_true', help="Run ODE-RNN baseline: RNN-style that sees true points at every point. Used for interpolation only.")

parser.add_argument('--rnn-vae', action='store_true', help="Run RNN baseline: seq2seq model with sampling of the h0 and ELBO loss.")

parser.add_argument('-l', '--latents', type=int, default=6, help="Size of the latent state")
parser.add_argument('--rec-dims', type=int, default=20, help="Dimensionality of the recognition model (ODE or RNN).")

parser.add_argument('--rec-layers', type=int, default=1, help="Number of layers in ODE func in recognition ODE")
parser.add_argument('--gen-layers', type=int, default=1, help="Number of layers in ODE func in generative ODE")

parser.add_argument('-u', '--units', type=int, default=100, help="Number of units per layer in ODE func")
parser.add_argument('-g', '--gru-units', type=int, default=100, help="Number of units per layer in each of GRU update networks")

parser.add_argument('--classif', action='store_true', help="Include binary classification loss -- used for fibrillation dataset for hospiral mortality")

parser.add_argument('--max-t',  type=float, default=5., help="We subsample points in the interval [0, args.max_tp]")
parser.add_argument('--noise-weight', type=float, default=0.01, help="Noise amplitude for generated traejctories")

#AÑADIMOS UN NUEVO ARGUMENTO PARA INDICAR EL TIPO DE CLASIFICACION: (binaria/multiclase)
parser.add_argument('--classif_tipe_multiclass', action='store_true', help="Include multiclass classification loss -- used for fibrillation dataset for hospiral mortality")
	#En el caso de que incluyamos este argumento se ejecutará clasificacion multiclase
	#En caso contrario se ejecutará clasificacion binaria


args = parser.parse_args(['--load', '52344', '--niters', '100', '-n', '11636', '-l', '20', '--dataset', 'fibrillation', '--latent-ode', '--rec-dims', '40', '--rec-layers', '3', '--gen-layers', '3', '--units', '50', '--gru-units', '50', '--classif', '--classif_tipe_multiclass'])
#args = parser.parse_args(['--niters', '100', '-n', '11636', '-l', '100', '-b', '100', '--dataset', 'fibrillation', '--latent-ode', '--rec-dims', '40', '--rec-layers', '10', '--gen-layers', '3', '--units', '200', '--gru-units', '200', '--classif', '--classif_tipe_multiclass'])

#args = parser.parse_args(['--load', '36437', '--niters', '100', '-n', '11636', '-l', '100', '--dataset', 'fibrillation', '--latent-ode', '--rec-dims', '40', '--rec-layers', '3', '--gen-layers', '3', '--units', '200', '--gru-units', '200', '--classif', '--classif_tipe_multiclass'])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
file_name = os.path.basename(__file__)[:-3]
utils.makedirs(args.save)

#####################################################################################################


if __name__ == '__main__':
	torch.manual_seed(args.random_seed)
	np.random.seed(args.random_seed)

	experimentID = args.load
	if experimentID is None:
		# Make a new experiment ID
		experimentID = int(SystemRandom().random()*100000)
	ckpt_path = os.path.join(args.save, "experiment_" + str(experimentID) + '.ckpt')

	start = time.time()
	

	
	input_command = sys.argv
	ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
	if len(ind) == 1:
		ind = ind[0]
		input_command = input_command[:ind] + input_command[(ind+2):]
	input_command = " ".join(input_command)	
	
	##########################################################################
	print("EXPERIMENTO ID: ", experimentID)
	print("Sampling dataset of {} training examples".format(args.n))
	if args.classif_tipe_multiclass :
		print("Tipo de clasificacion: MULTICLASE")
	else:
		print("Tipo de clasificacion: BINARIA")
	
	print("---------------------------------------------------------------- \n")

	#####################################
	print("Iniciamos carga de los datos")
	#####################################
	
	##################################################################
	data_obj = parse_datasets(args, device)
	input_dim = data_obj["input_dim"]
	
	classif_per_tp = False
	if ("classif_per_tp" in data_obj):
		# do classification per time point rather than on a time series as a whole
		classif_per_tp = data_obj["classif_per_tp"]


	n_labels = data_obj['n_labels'] #NUMERO CATEGORIAS
	if args.classif:
		if ("n_labels" in data_obj):
			n_labels = data_obj["n_labels"]
		else:
			raise Exception("Please provide number of labels for classification task")
	print("NUMERO CLASES: ", n_labels)
	print("Numero Batches TRAIN: ", data_obj["n_train_batches"])
	print("Numero Batches TEST: ", data_obj["n_test_batches"])

	# Creamos una carpeta en la que se guardan los resultados del expiremento (test o train)
	nombre_carpeta = str(experimentID)
	ruta_carpeta = "ficheros_resultados/"
	ruta_completa = os.path.join(ruta_carpeta, nombre_carpeta)
	if not os.path.exists(ruta_completa):
		os.makedirs(ruta_completa)

	#####################################
	print("Carga realizada\n Iniciamos la creación del modelo")
	#####################################
	
	##################################################################
	# Create the model
	obsrv_std = 0.01
	obsrv_std = torch.Tensor([obsrv_std]).to(device)
	#print("OBSRV_STD: ", obsrv_std)

	z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))
	#print("Z0_prior: ", z0_prior)

	if args.latent_ode:
		print("CREACION LATENT ODE.....")
		model = create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device, classif_per_tp = classif_per_tp, n_labels = n_labels)
		
		#####################################
		print("\tSe ha creado una LATENT_ODE")
		#####################################
	
	else:
		#####################################
		print("\tNO se ha podido crear ningún modelo")
		#####################################

		raise Exception("Model not specified")
		

	#Load checkpoint and evaluate the model
	if args.load is not None:
    	
		#CREAMOS LOS ARCHIVOS QUE GUARDAN LOS RESULTADOS
		nombre_archivo = "ficheros_resultados/"+str(experimentID)+"/resultados_test.txt"
		with open(nombre_archivo, 'w') as archivo:
			archivo.write(f"EXPERIMENTO: {experimentID}")
			archivo.write(f"\t\t Prueba\n-------------------------------------------------------------\n")

		nombre_archivo = "ficheros_resultados/"+str(experimentID)+"/ce_loss_test.txt"
		with open(nombre_archivo, 'w') as archivo:
			archivo.write(f"EXPERIMENTO: {experimentID}")
			archivo.write(f"\t\t Prueba\n-------------------------------------------------------------\n")

			
		#####################################
		print("EVALUANDO EL MODELO y CREANDO CHECKPOINTS")
		#####################################

		utils.get_ckpt_model(ckpt_path, model, device)
				
		train_res = compute_loss_all_batches(model, 
					data_obj["test_dataloader"], args,
					n_batches = data_obj["n_test_batches"],
					experimentID = experimentID,
					device = device,
					n_traj_samples = 1, 
					tipo="train", n_experiment=experimentID)
		
		if args.classif_tipe_multiclass:
			x = np.array(train_res['labels'].cpu())
			y = train_res['label_predictions'].cpu() #guardamos prediciones
			y = y.unsqueeze(0)

			y_auc= y.squeeze() #eliminamos una dimension de y
			y_auc = F.softmax(y_auc, dim=1) #normalizamos los valores de y (la suma de las tres clases = 1)
			y_auc= y_auc.detach().numpy() #convertimos a array 
			
			y_one_hot = torch.zeros_like(y) #creamos una one-hot de ceros
			max_indices=torch.argmax(y, dim=2) #obtenemos los indices de la clase
			y_one_hot.scatter_(2, max_indices.unsqueeze(2), 1) #damos valores a la one-hot
			y = y_one_hot.numpy() #convertimos en array 
			y = y.squeeze() #eliminamos una dimension
				
			x_1d=np.argmax(x, axis=1) #convertimos en etiquetas de una dimension
			y_1d=np.argmax(y, axis=1) #convertimos en etiquetas de una dimension

			#AUC
			auc = sk.metrics.roc_auc_score(x, y_auc, multi_class='ovr')
			print("AUC: ", auc)
			#Accuracy
			accuracy = sk.metrics.accuracy_score(x_1d, y_1d)
			print("Accuracy: ", accuracy)
			# F1-score
			f1 = sk.metrics.f1_score(x_1d, y_1d, average='weighted')
			print("F1-score:", f1)
			#Matriz Confusión
			conf_matrix = sk.metrics.confusion_matrix(x_1d,y_1d)
			print("Matriz Confusion: \n", conf_matrix)
			# Precision
			precision = sk.metrics.precision_score(x, y, average='weighted')
			print("Precision:", precision)
			# Recall
			recall = sk.metrics.recall_score(x, y, average='weighted')
			print("Recall:", recall)
			# Average precision score
			average_precision = sk.metrics.average_precision_score(x, y)
			print("Average precision score:", average_precision)


		else:
			x = np.array(train_res['labels']) #guardamos valores reales como array
			y = np.array(train_res['label_predictions']).T #guardamos valores reales (TRASPUESTOS) como array
			fpr, tpr, thresholds = sk.metrics.roc_curve(x, y) #calculamos la tase de verdaderos positivos y falsos positivos 
			
			y=torch.round(torch.sigmoid(train_res['label_predictions'])) #se convierten los valores de las prediciones en 0 o 1
			y = np.array(y).T #se convierte en array y se traspone

			#AUC
			auc = sk.metrics.roc_auc_score(x, y)
			print("AUC: ", auc)
			#Accuracy
			accuracy = sk.metrics.accuracy_score(x, y)
			print("Accuracy: ", accuracy)
			# F1-score
			f1 = sk.metrics.f1_score(x, y)
			print("F1-score:", f1)
			#Matriz Confusión
			conf_matrix = sk.metrics.confusion_matrix(x,y)
			print("Matriz Confusion: \n", conf_matrix)
			# Precision
			precision = sk.metrics.precision_score(x, y)
			print("Precision:", precision)
			# Recall
			recall = sk.metrics.recall_score(x, y)
			print("Recall:", recall)
			# Average precision score
			average_precision = sk.metrics.average_precision_score(x, y)
			print("Average precision score:", average_precision)

		# Abrir el archivo en modo de escritura (se sobrescribirá si ya existe)
		nombre_archivo = "ficheros_resultados/"+str(experimentID)+"/resultados_test.txt"
		with open(nombre_archivo, "a") as archivo:
			archivo.write("AUC: "+ str(auc)+"\n")
			archivo.write("Accuracy: "+str(accuracy)+"\n")
			archivo.write("F1-score: " + str(f1) + "\n")
			archivo.write("Matriz Confusion: \n"+ str(conf_matrix)+ "\n")
			archivo.write("Precisión: "+ str(precision) + "\n")
			archivo.write("Recall: " + str(recall) + "\n")
			archivo.write("Average precision score: " + str(average_precision) + "\n")
        
		#####################################
		print("FIN EVALUACIÓN del MODELO")
		#####################################

		exit()
	
		

	##################################################################
	# Training

	#####################################
	print("INICIANDO ENTRENAMIENTO DEL MODELO")
	#####################################

	log_path = "logs/" + file_name + "_" + str(experimentID) + ".log"
	if not os.path.exists("logs/"):
		utils.makedirs("logs/")
	logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
	logger.info(input_command)

	
	optimizer = optim.Adamax(model.parameters(), lr=args.lr)

	num_batches = data_obj["n_train_batches"]
	batch_actual_iter=0
	num_iters_batch=0

	#CREAMOS LOS FICHEROS DE RESULTADOS DE TRAIN
	nombre_archivo = "ficheros_resultados/"+str(experimentID)+"/ce_loss_train.txt"
	with open(nombre_archivo, 'w') as archivo:
		archivo.write(f"EXPERIMENTO: {experimentID}")
		archivo.write(f"\t\t EPOCA {batch_actual_iter}\n-------------------------------------------------------------\n")

	nombre_archivo = "ficheros_resultados/"+str(experimentID)+"/resultados_batch_train.txt"
	with open(nombre_archivo, 'w') as archivo:
		archivo.write(f"EXPERIMENTO: {experimentID}")
		archivo.write(f"\t\t EPOCA {batch_actual_iter}\n-------------------------------------------------------------\n")

	nombre_archivo = "ficheros_resultados/"+str(experimentID)+"/balanceoClases.txt"
	with open(nombre_archivo, 'w') as archivo:
		archivo.write(f"EXPERIMENTO: {experimentID}")
		archivo.write(f"\t\t EPOCA {batch_actual_iter}\n-------------------------------------------------------------\n")


	for itr in range(1, num_batches * (args.niters + 1)):
		optimizer.zero_grad()
		utils.update_learning_rate(optimizer, decay_rate = 0.999, lowest = args.lr / 10)

		wait_until_kl_inc = 10
		if itr // num_batches < wait_until_kl_inc:
			kl_coef = 0.
		else:
			kl_coef = (1-0.99** (itr // num_batches - wait_until_kl_inc))
		
				
		print("\n------------------------------------------------")
		print("SIGUIENTE GRUPO DE ELEMS EPOCA: ", batch_actual_iter, " Batch: ", num_iters_batch )
		batch_dict = utils.get_next_batch(data_obj["train_dataloader"])

		#PRINT
		#################################################################################################################
		#COMPROBACION DE CUANTOS EJEMPLOS DE CADA TIPO TENEMOS:
		labelsX = batch_dict['labels']
		labelsX_cpu = labelsX.cpu()
		labelsX_cpu_numpy = labelsX_cpu.numpy()
		# Contar el número de elementos de cada clase
		unique_classes, counts = np.unique(labelsX_cpu_numpy, axis=0, return_counts=True)

		#Guardar Balanceo de clases:
		###########################################################################################	
		nombre_archivo = "ficheros_resultados/"+str(experimentID)+"/balanceoClases.txt"
		with open(nombre_archivo, 'a') as archivo:
			archivo.write(f"Batch: {num_iters_batch}:\n")
		
			for i, class_count in enumerate(counts):
				print(f"Clase {unique_classes[i]}: {class_count} elementos")
				archivo.write(f"Clase {unique_classes[i]}: {class_count} elementos\n")
		###########################################################################################	
		
		
		print("\tCALCULAMOS LAS PERDIDAS BATCH")
		train_res = model.compute_all_losses(batch_dict, args, n_traj_samples = 3, kl_coef = kl_coef, iter_batch=num_iters_batch, tipo="train", n_experiment=experimentID)
		print("\tFIN CALCULO PERDIDAS BATCH")

		#actualizamos las iteracion en la que nos encontramos en un batch
		num_iters_batch+=1

		if train_res["loss"].grad_fn is not None:
			train_res["loss"].backward()
			#print("TRAIN_res_loss: ", train_res["loss"])
			optimizer.step()

		
		n_iters_to_viz = 1
		if itr % (n_iters_to_viz * num_batches) == 0:
			print("\n\n\n----------------------------------------------\n\n\n")
			print("CALCULO PERDIDAS FIN EPOCA")
			with torch.no_grad():				
				test_res = compute_loss_all_batches(model, 
					data_obj["test_dataloader"], args,
					n_batches = data_obj["n_test_batches"],
					experimentID = experimentID,
					device = device,
					n_traj_samples = 3, kl_coef = kl_coef,
					tipo="train", n_experiment=experimentID)
				
				message = 'Epoch {:04d} [Test seq (cond on sampled tp)] | Loss {:.6f} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|'.format(
					itr//num_batches, 
					test_res["loss"].detach(), test_res["likelihood"].detach(), 
					test_res["kl_first_p"], test_res["std_first_p"])
		 	
				logger.info("Experiment " + str(experimentID))
				logger.info(message)
				logger.info("KL coef: {}".format(kl_coef))
				logger.info("Train loss (one batch): {}".format(train_res["loss"].detach()))
				logger.info("Train CE loss (one batch): {}".format(train_res["ce_loss"].detach()))
				
				if "auc" in test_res:
					logger.info("Classification AUC (TEST): {:.4f}".format(test_res["auc"]))

				if "mse" in test_res:
					logger.info("Test MSE: {:.4f}".format(test_res["mse"]))

				if "accuracy" in train_res:
					logger.info("Classification accuracy (TRAIN): {:.4f}".format(train_res["accuracy"]))

				if "accuracy" in test_res:
					logger.info("Classification accuracy (TEST): {:.4f}".format(test_res["accuracy"]))

				if "ce_loss" in test_res:
					logger.info("CE loss: {}".format(test_res["ce_loss"]))
				

				#GUARDAMOS EN UN FICHERO PROPIO
					# Guardar los resultados en el archivo de resultados del lote de entrenamiento
				with open("ficheros_resultados/"+str(experimentID)+"/resultados_batch_train.txt", 'a') as archivo_resultados:  # 'a' para abrir en modo de agregar
					archivo_resultados.write("\t##########################################\n")
					
					
					archivo_resultados.write(message + '\n')
					archivo_resultados.write("KL coef: {}\n".format(kl_coef))
					archivo_resultados.write("Train loss (one batch): {}\n".format(train_res["loss"].detach()))
					archivo_resultados.write("Train CE loss (one batch): {}\n".format(train_res["ce_loss"].detach()))
					
					if "auc" in test_res:
						archivo_resultados.write("Classification AUC (TEST): {:.4f}\n".format(test_res["auc"]))

					if "mse" in test_res:
						archivo_resultados.write("Test MSE: {:.4f}\n".format(test_res["mse"]))

					if "accuracy" in train_res:
						archivo_resultados.write("Classification accuracy (TRAIN): {:.4f}\n".format(train_res["accuracy"]))

					if "accuracy" in test_res:
						archivo_resultados.write("Classification accuracy (TEST): {:.4f}\n".format(test_res["accuracy"]))

					if "ce_loss" in test_res:
						archivo_resultados.write("CE loss: {}\n".format(test_res["ce_loss"]))
					
					archivo_resultados.write("\t##########################################\n")

					

				#Actualizamos el Numero de BATCH
				batch_actual_iter+=1

				#ESCRIBIMOS EN EL FICHERO
				nombre_archivo = "ficheros_resultados/"+str(experimentID)+"/ce_loss_train.txt"
				with open(nombre_archivo, 'a') as archivo:
					archivo.write(f"\t\t EPOCA {batch_actual_iter}\n-------------------------------------------------------------\n")

				nombre_archivo = "ficheros_resultados/"+str(experimentID)+"/resultados_batch_train.txt"
				with open(nombre_archivo, 'a') as archivo:
					archivo.write(f"\t\t EPOCA {batch_actual_iter}\n-------------------------------------------------------------\n")

				#reiniciamos iteración dentro del BATCH
				num_iters_batch=0
			torch.save({
				'args': args,
				'state_dict': model.state_dict(),
			}, ckpt_path)
			
	torch.save({
		'args': args,
		'state_dict': model.state_dict(),
	}, ckpt_path)
	print('END')

