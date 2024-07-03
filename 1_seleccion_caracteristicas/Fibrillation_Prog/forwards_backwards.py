import os
import sys

import time
import argparse
import numpy as np
from random import SystemRandom
import psutil
import sklearn as sk
import pandas as pd

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



args = parser.parse_args(['--niters', '10', '-n', '11636', '-l', '15', '-b', '16', '--dataset', 'fibrillation', '--latent-ode', '--rec-dims', '40', '--rec-layers', '3', '--gen-layers', '3', '--units', '300', '--gru-units', '100', '--classif', '--classif_tipe_multiclass'])



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
file_name = os.path.basename(__file__)[:-3]
utils.makedirs(args.save)

#####################################################################################################


def ejecucion_modelo(reduced_features, feature, foldID):
    obsrv_std = 0.01
    obsrv_std = torch.Tensor([obsrv_std]).to(device)
    #print("OBSRV_STD: ", obsrv_std)

    z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))
    #print("Z0_prior: ", z0_prior)

    if args.latent_ode:
        print("\n-----------------------------\n")
        print("CREACION LATENT ODE.....")
        print("N_lavels creacion modelo: ", n_labels)
        print("\n-----------------------------\n")
        model = create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device, classif_per_tp = classif_per_tp, n_labels = n_labels)
        
        #####################################
        print("\t\tSe ha creado una LATENT_ODE")
        #####################################
    else:
        #####################################
        print("\t\tNO se ha podido crear ningún modelo")
        #####################################

        raise Exception("Model not specified")
    
    ##################################################################
    # Training
    #####################################
    print("\t\nINICIANDO ENTRENAMIENTO DEL MODELO")
    #####################################
    
    
    optimizer = optim.Adamax(model.parameters(), lr=args.lr)

    num_batches = data_obj["n_train_batches"]
    batch_actual_iter=0
    num_iters_batch=0

    #CREAMOS LOS FICHEROS DE RESULTADOS DE TRAIN
    print("\n\n\n#############################\nNUEVO CONJUNTO DE CARACTERISTICAS")
    print("FOLD: ", foldID)
    print("CARACTERISTICAS: \n", reduced_features)
    print("CARACTERISTICA ELIMINADA: ", feature)
    print("NUMERO CARACTERISTICAS: ", len(reduced_features))

    nombre_archivo = "ficheros_resultados/"+str(experimentID)+"/ce_loss_train.txt"
    with open(nombre_archivo, 'a') as archivo:
        archivo.write(f"\n\n-----------------------------------------\nNUEVO CONJUNTO DE CARACTERISTICAS\n")
        archivo.write(f"EXPERIMENTO: {experimentID}\n")
        archivo.write(f"CARACTERISTICAS:\n{reduced_features}\n")
        archivo.write(f"\n-------------------------------------------------------------\n")

    nombre_archivo = "ficheros_resultados/"+str(experimentID)+"/resultados_batch_train.txt"
    with open(nombre_archivo, 'a') as archivo:
        archivo.write(f"\n\n-----------------------------------------\nNUEVO CONJUNTO DE CARACTERISTICAS\n")
        archivo.write(f"EXPERIMENTO: {experimentID}\n")
        archivo.write(f"CARACTERISTICAS:\n{reduced_features}\n")
        archivo.write(f"\n-------------------------------------------------------------\n")

    nombre_archivo = "ficheros_resultados/"+str(experimentID)+"/balanceoClases.txt"
    with open(nombre_archivo, 'a') as archivo:
        archivo.write(f"\n\n-----------------------------------------\nNUEVO CONJUNTO DE CARACTERISTICAS\n")
        archivo.write(f"EXPERIMENTO: {experimentID}\n")
        archivo.write(f"CARACTERISTICAS:\n{reduced_features}\n")
        archivo.write(f"\t\t EPOCA {batch_actual_iter}\n-------------------------------------------------------------\n")

    ckpt_path_train = os.path.join(args.save, "experiment_" + str(experimentID) + '_train.ckpt')

    for itr in range(1, num_batches * (args.niters + 1)):
        try:
            optimizer.zero_grad()
            utils.update_learning_rate(optimizer, decay_rate = 0.999, lowest = args.lr / 10)

            wait_until_kl_inc = 10
            if itr // num_batches < wait_until_kl_inc:
                kl_coef = 0.
            else:
                kl_coef = (1-0.99** (itr // num_batches - wait_until_kl_inc))
                            
            print("\t\n------------------------------------------------")
            print("\tSIGUIENTE GRUPO DE ELEMS EPOCA: ", batch_actual_iter, " Batch: ", num_iters_batch )
            batch_dict = utils.get_next_batch(data_obj["train_dataloader"], reduced_features)

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
                archivo.write(f"Fold {fold_ID} - Batch: {num_iters_batch}:\n")
            
                for i, class_count in enumerate(counts):
                    print(f"\tClase {unique_classes[i]}: {class_count} elementos")
                    archivo.write(f"Clase {unique_classes[i]}: {class_count} elementos\n")
            ###########################################################################################	 

            print("\n\t\tCALCULAMOS LAS PERDIDAS BATCH")
            train_res = model.compute_all_losses(batch_dict, args, n_traj_samples = 3, kl_coef = kl_coef, iter_batch=num_iters_batch, tipo="train", n_experiment=experimentID)
            print("\t\tFIN CALCULO PERDIDAS BATCH")

            #actualizamos las iteracion en la que nos encontramos en un batch
            num_iters_batch+=1

            if train_res["loss"].grad_fn is not None:
                train_res["loss"].backward()
                optimizer.step()
                            
            n_iters_to_viz = 1            

            if itr % (n_iters_to_viz * num_batches) == 0:
                print("\t\n\n\n----------------------------------------------\n\n\n")
                print("\tCALCULO PERDIDAS FIN EPOCA")
                with open("ficheros_resultados/"+str(experimentID)+"/resultados_batch_train.txt", 'a') as archivo:  # 'a' para abrir en modo de agregar
                    archivo.write(f"\t\tFold {fold_ID} -  EPOCA {batch_actual_iter}\n-------------------------------------------------------------\n")
                with torch.no_grad():				
                    test_res = compute_loss_all_batches(model, 
                        data_obj["test_dataloader"], args,
                        n_batches = data_obj["n_test_batches"],
                        experimentID = experimentID,
                        device = device,
                        n_traj_samples = 3, kl_coef = kl_coef,
                        tipo="train", n_experiment=experimentID, reduced_features=reduced_features)

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

                    if "pois_likelihood" in test_res:
                        logger.info("Poisson likelihood: {}".format(test_res["pois_likelihood"]))

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

                        if "pois_likelihood" in test_res:
                            archivo_resultados.write("Poisson likelihood: {}\n".format(test_res["pois_likelihood"]))

                        if "ce_loss" in test_res:
                            archivo_resultados.write("CE loss: {}\n".format(test_res["ce_loss"]))
                        
                        archivo_resultados.write("\t##########################################\n")

                    #Actualizamos el Numero de BATCH
                    batch_actual_iter+=1

                    #ESCRIBIMOS EN EL FICHERO
                    nombre_archivo = "ficheros_resultados/"+str(experimentID)+"/ce_loss_train.txt"
                    with open(nombre_archivo, 'a') as archivo:
                        archivo.write(f"\t\tFold {fold_ID} -  EPOCA {batch_actual_iter}\n-------------------------------------------------------------\n")

                    
                    #reiniciamos iteración dentro del BATCH
                    num_iters_batch=0
                    
                torch.save({
                    'args': args,
                    'state_dict': model.state_dict(),
                }, ckpt_path_train)
        except ValueError as e:
            print("Error en:\n Epoca: ", batch_actual_iter, " Batch: ", num_iters_batch,"\n pasamos a la siguiente" )
			
    torch.save({
		'args': args,
		'state_dict': model.state_dict(),
	}, ckpt_path_train)

    print('\n---------------------\n\tEND TRAINING')

    #####################################
    print("\n---------------------\n\tCarga realizada\n Iniciamos la creación del modelo test")
    ##########################################################################
    # Create the model
    obsrv_std = 0.01
    obsrv_std = torch.Tensor([obsrv_std]).to(device)

    z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))

    if args.latent_ode:
        print("\tCREACION LATENT ODE.....")
        print("\tN_lavels creacion modelo: ", n_labels)
        model = create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device, classif_per_tp = classif_per_tp, n_labels = n_labels)
        
        #####################################
        print("\t\tSe ha creado una LATENT_ODE")
        #####################################
    else:
        #####################################
        print("\t\tNO se ha podido crear ningún modelo")
        #####################################

        raise Exception("Model not specified")
    ##########################################################################

    #CREAMOS LOS ARCHIVOS QUE GUARDAN LOS RESULTADOS
    nombre_archivo = "ficheros_resultados/"+str(experimentID)+"/resultados_test.txt"
    with open(nombre_archivo, 'a') as archivo:
        archivo.write(f"\n\n-----------------------------------------\nNUEVO CONJUNTO DE CARACTERISTICAS\n")
        archivo.write(f"EXPERIMENTO: {experimentID}\n")
        archivo.write(f"CARACTERISTICAS:\n{reduced_features}\n")
        archivo.write(f"\t\t Prueba\n-------------------------------------------------------------\n")

    nombre_archivo = "ficheros_resultados/"+str(experimentID)+"/ce_loss_test.txt"
    with open(nombre_archivo, 'a') as archivo:
        archivo.write(f"\n\n-----------------------------------------\nNUEVO CONJUNTO DE CARACTERISTICAS\n")
        archivo.write(f"EXPERIMENTO: {experimentID}\n")
        archivo.write(f"CARACTERISTICAS:\n{reduced_features}\n")
        archivo.write(f"\t\t TEST\n-------------------------------------------------------------\n")


    utils.get_ckpt_model(ckpt_path_train, model, device)

    test_res = compute_loss_all_batches(model, 
                        data_obj["test_dataloader"], args,
                        n_batches = data_obj["n_test_batches"],
                        experimentID = experimentID,
                        device = device,
                        n_traj_samples = 1, kl_coef = kl_coef,
                        tipo="test", n_experiment=experimentID, reduced_features=reduced_features)
    
    nombre_archivo = "ficheros_resultados/"+str(experimentID)+"/ce_loss_test.txt"
    with open(nombre_archivo, 'a') as archivo:
        archivo.write(f'LOSS: {test_res["loss"]}\n')

    if args.classif_tipe_multiclass:
        x = np.array(test_res['labels'].cpu())
        y = test_res['label_predictions'].cpu() #guardamos prediciones
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

        print("CARACTERISTICAS: \n", reduced_features)
        print("NUMERO CARACTERISTICAS: ", len(reduced_features))
        print("CARACTERISTICA ELIMINADA: ", feature)
        #AUC
        auc = sk.metrics.roc_auc_score(x, y_auc, multi_class='ovr')
        print("\tAUC: ", auc)
        #Accuracy
        accuracy = sk.metrics.accuracy_score(x_1d, y_1d)
        print("\tAccuracy: ", accuracy)
        # F1-score
        f1 = sk.metrics.f1_score(x_1d, y_1d, average='weighted')
        print("\tF1-score:", f1)
        #Matriz Confusión
        conf_matrix = sk.metrics.confusion_matrix(x_1d,y_1d)
        print("\tMatriz Confusion: \n\t", conf_matrix)
        # Precision
        precision = sk.metrics.precision_score(x, y, average='weighted', zero_division=0)
        print("\tPrecision:", precision)
        # Recall
        recall = sk.metrics.recall_score(x, y, average='weighted')
        print("\tRecall:", recall)
        # Average precision score
        average_precision = sk.metrics.average_precision_score(x, y)
        print("\tAverage precision score:", average_precision)

       
    else:
        x = np.array(batch_dict['labels']) #guardamos valores reales como array
        y = np.array(train_res['label_predictions']).T #guardamos valores reales (TRASPUESTOS) como array
        fpr, tpr, thresholds = sk.metrics.roc_curve(x, y) #calculamos la tase de verdaderos positivos y falsos positivos 
        
        y=torch.round(torch.sigmoid(train_res['label_predictions'])) #se convierten los valores de las prediciones en 0 o 1
        y = np.array(y).T #se convierte en array y se traspone

        #AUC
        auc = sk.metrics.roc_auc_score(x, y)
        print("\tAUC: ", auc)
        #Accuracy
        accuracy = sk.metrics.accuracy_score(x, y)
        print("\tAccuracy: ", accuracy)
        # F1-score
        f1 = sk.metrics.f1_score(x, y)
        print("\tF1-score:", f1)
        #Matriz Confusión
        conf_matrix = sk.metrics.confusion_matrix(x,y)
        print("\tMatriz Confusion: \n", conf_matrix)
        # Precision
        precision = sk.metrics.precision_score(x, y)
        print("\tPrecision:", precision)
        # Recall
        recall = sk.metrics.recall_score(x, y)
        print("\tRecall:", recall)
        # Average precision score
        average_precision = sk.metrics.average_precision_score(x, y)
        print("\tAverage precision score:", average_precision)

    
    nombre_archivo = "ficheros_resultados/"+str(experimentID)+"/resultados_test.txt"
    with open(nombre_archivo, "a") as archivo:
        archivo.write("CARACTERISTICAS: " + str(reduced_features)+ "\n")
        archivo.write("NUMERO CARACTERISTICAS: " + str(len(reduced_features))+ "\n")
        archivo.write("CARACTERISTICA ELIMINADA: " + str(feature)+ "\n")
        archivo.write("AUC: "+ str(auc)+"\n")
        archivo.write("Accuracy: "+str(accuracy)+"\n")
        archivo.write("F1-score: " + str(f1) + "\n")
        archivo.write("Matriz Confusion: \n"+ str(conf_matrix)+ "\n")
        archivo.write("Precisión: "+ str(precision) + "\n")
        archivo.write("Recall: " + str(recall) + "\n")
        archivo.write("Average precision score: " + str(average_precision) + "\n")
        
        
    torch.save({
        'args': args,
        'state_dict': model.state_dict(),
    }, ckpt_path)
    #####################################
    print("\n---------------------\n\tFIN EVALUACIÓN del MODELO\n\n\n-----------------------------------")
    ##################################### 
    return f1

def crear_ficheros():
    # Creamos una carpeta en la que se guardan los resultados del expiremento (test o train)
    nombre_carpeta = str(experimentID)
    ruta_carpeta = "ficheros_resultados/"
    ruta_completa = os.path.join(ruta_carpeta, nombre_carpeta)
    if not os.path.exists(ruta_completa):
        os.makedirs(ruta_completa)
    

    nombre_archivo = "ficheros_resultados/"+str(experimentID)+"/ce_loss_train.txt"
    with open(nombre_archivo, 'w') as archivo:
        archivo.write(f"\t\t FOLD {fold_ID}\n-------------------------------------------------------------\n")

    nombre_archivo = "ficheros_resultados/"+str(experimentID)+"/resultados_batch_train.txt"
    with open(nombre_archivo, 'w') as archivo:
        archivo.write(f"EXPERIMENTO: {experimentID}")
        archivo.write(f"\t\t FOLD {fold_ID}\n-------------------------------------------------------------\n")

    nombre_archivo = "ficheros_resultados/"+str(experimentID)+"/balanceoClases.txt"
    with open(nombre_archivo, 'w') as archivo:
        archivo.write(f"EXPERIMENTO: {experimentID}")
        archivo.write(f"\t\t FOLD {fold_ID}\n-------------------------------------------------------------\n")
        
    nombre_archivo = "ficheros_resultados/"+str(experimentID)+"/resultados_test.txt"
    with open(nombre_archivo, 'w') as archivo:
        archivo.write(f"EXPERIMENTO: {experimentID}")
        archivo.write(f"\t\t Prueba\n-------------------------------------------------------------\n")

    nombre_archivo = "ficheros_resultados/"+str(experimentID)+"/ce_loss_test.txt"
    with open(nombre_archivo, 'w') as archivo:
        archivo.write(f"EXPERIMENTO: {experimentID}")
        archivo.write(f"\t\t Prueba\n-------------------------------------------------------------\n")

    nombre_archivo = "ficheros_resultados/"+str(experimentID)+"/características.txt"
    with open(nombre_archivo, 'w') as archivo:
        archivo.write(f"EXPERIMENTO: {experimentID}")
        archivo.write(f"\n-------------------------------------------------------------\n")


    ###########################################################################

if __name__ == '__main__':
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    experimentID_general = args.load 
    if experimentID_general is None:
        experimentID_general = int(SystemRandom().random()*100000)
    start = time.time()
    print("EXPERIMENTO: ", experimentID_general)
    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind+2):]
    input_command = " ".join(input_command)

    ###############################
	#Creamos el conjunto de características Inicial
    rutaBD = '../BD_pacientes_FINAL/BD_COMPLETA/'
    
    df_contar_columnas=pd.read_csv(rutaBD+'1_SR.csv')
    num_features = len(df_contar_columnas.columns)
    print("NUM FEATURES: ", num_features)
    loaded_features = list(range(num_features))
    loaded_features = loaded_features[:-2] #eliminamos los dos ultimos elementos [FA][TIEMPO], puesto que no se consideran características	
    
    
    ####################################
    features_glob = [0, 1, 5, 8, 12, 13, 14, 18, 22, 23, 30, 36, 46, 59, 70, 71, 80, 89, 91, 97, 107, 108, 146, 149, 166, 185, 209, 220, 227, 257, 271, 279, 295, 312, 315, 360, 375, 381, 382, 402, 417, 429, 439, 442, 462, 474, 475, 476, 479, 492]
    backup_features = [0, 1, 8, 22, 30, 59, 70, 97, 220, 439, 474, 476]
    
    selected_features = features_glob   

    fold_ejecucion = 0
    n_folds = 6
    max_features = 19
    max_add_sprint = 5
    reduc_fetures_num = 3
    ####################################
    

    logger=None
    log_path = "logs/" + file_name + "_" + str(experimentID_general) +".log"
    if not os.path.exists("logs/"):
        utils.makedirs("logs/")
    logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(input_command)

    f1_scores_per_fold = []
    caracteristicas_per_fold = []

    nombre_carpeta_general = str(experimentID_general)
    ruta_carpeta = "ficheros_resultados/"
    ruta_completa_general = os.path.join(ruta_carpeta, nombre_carpeta_general) 
    if not os.path.exists(ruta_completa_general):
        os.makedirs(ruta_completa_general)

    nombre_archivo = "ficheros_resultados/"+str(experimentID_general)+"/resultados_FOLDS.txt"
    with open(nombre_archivo, 'w') as archivo:
        archivo.write(f"EXPERIMENTO: {experimentID_general}")
        archivo.write(f"\t\t RESULDATOS FOLD\n-------------------------------------------------------------\n")
        
    for fold_ID, data_obj in enumerate(parse_datasets(args, device, n_folds, rutaBD, loaded_features, experimentID_general)):
        print("\n\n\t---------------------------------------\n")
        print("EJECUCION FOLD_: ", fold_ID)
        print("\n---------------------------------------\n")	
        print("Total_Datos: ", len(data_obj["dataset_obj"]))	
        print("Num_Features: ", data_obj["input_dim"])
        print("N_train_batches= ", data_obj["n_train_batches"])
        print("N_test_batches= ", data_obj["n_test_batches"])
        print("N_labels= ", data_obj["n_labels"]) 
                    
        non_used_features_in_fold = selected_features.copy()
        # Eliminamos las características que están en backup_features
        non_used_features_in_fold = [feat for feat in non_used_features_in_fold if feat not in backup_features]

        selected_features_in_fold = backup_features.copy()
       
        ##CREAMOS un ID para cada FOLD
        experimentID = experimentID_general
        experimentID = f'{experimentID}_fold_{fold_ID}'
        ckpt_path = os.path.join(args.save, "experiment_" + str(experimentID) + '.ckpt')
        
        crear_ficheros()       

        input_dim = data_obj["input_dim"]
        n_labels = data_obj['n_labels'] #NUMERO CATEGORIAS
        if args.classif:
            if ("n_labels" in data_obj):
                n_labels = data_obj["n_labels"]
            else:
                raise Exception("Please provide number of labels for classification task")
        print("\tNUMERO CLASES: ", n_labels)
        print("\tNumero Batches TRAIN: ", data_obj["n_train_batches"])
        print("\tNumero Batches TEST: ", data_obj["n_test_batches"])

        classif_per_tp = False
        if ("classif_per_tp" in data_obj):
            # do classification per time point rather than on a time series as a whole
            classif_per_tp = data_obj["classif_per_tp"]  

       
        #####################################
        print("\t\nCarga realizada\nIniciamos la creación selección de características")
        
        
        input_dim = len(selected_features_in_fold)
        f1 = ejecucion_modelo(reduced_features=selected_features_in_fold, feature=-1, foldID=fold_ID)
        best_F1 = f1
        last_add_sprint = 1

        while (len(selected_features_in_fold) < max_features):
            best_intern_F1 = -1
            best_intern_feature = None
            
            #############
            print("CARAC:\n", selected_features_in_fold)
            #############

            nombre_archivo_car = "ficheros_resultados/"+str(experimentID)+"/características.txt"
            with open(nombre_archivo_car, 'a') as archivo:
                archivo.write("\n\n\n\n-----------------------------------------\n")
                archivo.write(f"NUMERO CARACTERISTICAS ELEGIDAS HASTA EL MOMENTO: {len(selected_features_in_fold)}\n")
                archivo.write(f'PROCEDEMOS A AÑADIR UNA MAS 1+{len(selected_features_in_fold)} = {1 +len(selected_features_in_fold)}\n')
                archivo.write(f'CARACTERISTICAS HASTA EL MOMENTO:  {selected_features_in_fold}\n\n')

            for feature in non_used_features_in_fold:
                reduced_features = selected_features_in_fold.copy()
                reduced_features.append(feature)
                nombre_archivo_car = "ficheros_resultados/"+str(experimentID)+"/características.txt"
                with open(nombre_archivo_car, 'a') as archivo:
                    archivo.write(f"NUEVA CARACTERISTICA = {feature}\t||\t")
                    archivo.write(f"CARACTERISTICAS ACTUALES: {reduced_features}\t||\t")
                input_dim = len(reduced_features)
                f1 = ejecucion_modelo(reduced_features=reduced_features, feature=feature, foldID=fold_ID)
                with open(nombre_archivo_car, 'a') as archivo:
                    archivo.write(f"F1: {f1}\n")
                #Compruebo si añadiendo esa característica consigo mejores resultados y la marco como la peor para eliminarla
				#añado la característca con la que al eliminarla, consigo los mejores resultados.
                if f1 > best_intern_F1:
                    if f1 > best_intern_F1:
                        best_intern_F1 = f1
                        best_intern_feature = feature

            ###############################
            #Comprobar que una vez se ha probado con todas las características los resultados mejoran,
            #para poder aumentar el tamaño del conjunto de caracterisitcas            
            
            if best_F1 < best_intern_F1 :
                ###############################
                #Una vez sabemos la mejor característica par aun tamaño determinado, la eliminamos
                selected_features_in_fold.append(best_intern_feature)
                non_used_features_in_fold.remove(best_intern_feature)
                print("\n\t\t\t\t\t\t\t\tSe ha añadido la caracteristica Numero: ", best_intern_feature, "\n")
                ###############################
            else:
                print("\n\t\t\t\t\t\t\t\tFold: ", fold_ID)
                print("\n\t\t\t\t\t\t\t\tNo se han mejorado los resultados con este NUMERO DE CARACTERISTICAS")
                break
            ###############################

            last_add_sprint +=1

            ###############################
            #En el caso de que se hayan añadido un total de 5 caracteristicas seguidas, procedemos a eliminar dos.
            if last_add_sprint % max_add_sprint == 0 and last_add_sprint != 0:
                f1_feature_del = {}
                nombre_archivo_car = "ficheros_resultados/"+str(experimentID)+"/características.txt"
                with open(nombre_archivo_car, 'a') as archivo:
                        archivo.write("ELIMINAMOS DOS CARACTERISTICAS\n\n")

                for feature_del in selected_features_in_fold:
                    reduced_features_del = selected_features_in_fold.copy()
                    reduced_features_del.remove(feature_del)
                    input_dim = len(reduced_features_del)
                    nombre_archivo_car = "ficheros_resultados/"+str(experimentID)+"/características.txt"
                    with open(nombre_archivo_car, 'a') as archivo:
                        archivo.write(f"CARACTERISTICAS_TRAS_ELIMINACION: {reduced_features_del}\t||\t")
                    f1_del = ejecucion_modelo(reduced_features=reduced_features_del, feature=feature_del, foldID=fold_ID)
                    f1_feature_del[feature_del] = f1_del
                    with open(nombre_archivo_car, 'a') as archivo:
                        archivo.write(f"F1: {f1_del}\n")
                
                
                # Ordenar las claves por sus valores en el diccionario
                sorted_features_del = sorted(f1_feature_del, key=f1_feature_del.get)
                # Seleccionar las dos claves con los valores más pequeños
                worst_features = sorted_features_del[:reduc_fetures_num]

                for feature_deleted in worst_features:
                    selected_features_in_fold.remove(feature_deleted)
                    non_used_features_in_fold.append(feature_deleted)

                nombre_archivo_car = "ficheros_resultados/"+str(experimentID)+"/características.txt"
                with open(nombre_archivo_car, 'a') as archivo:
                        archivo.write(f"\t\tCARACTERISTICAS ELIMINADS: {worst_features}\n")
                
                last_add_sprint = 0
               
            ###############################

        ###############################
        print("\n\tSE HA ALCANZADO EL LIMITE DE CARACTERISTICAS O EL MODELO NO MEJORA\n")        
        reduced_features = selected_features_in_fold
        input_dim = len(reduced_features)
        mejores_resultados = ejecucion_modelo(reduced_features=reduced_features, feature=-1, foldID=fold_ID)      
        

        print("\n\n\n\n--------------------------------------------\nFEATURES_FINAL FOLD", fold_ID,": \n", selected_features_in_fold, "\n-------------------------\n")
        print("F1_final: ", mejores_resultados)
        print("--------------------------------------------\n\n\n\n")

        f1_scores_per_fold.append(mejores_resultados)
        caracteristicas_per_fold.append(reduced_features)

        nombre_archivo = "ficheros_resultados/"+str(experimentID_general)+"/resultados_FOLDS.txt"
        with open(nombre_archivo, 'a') as archivo:
            archivo.write(f"\n\nEXPERIMENTO: {experimentID_general}")
            archivo.write(f"\t\t RESULTADOS FOLD {fold_ID}\n-------------------------------------------------------------\n")
            archivo.write(f"F1-score: {mejores_resultados}\n")
            archivo.write(f"Caracteristicas: \n{reduced_features}")
            archivo.write(f"\n-------------------------------------------------------------\n")
        ###############################
    
    for x, f1 in enumerate(f1_scores_per_fold):
        print("FOLD_EJECUCION::> ", fold_ejecucion)
        print(f"Fold-> {x}: {f1}")

    for x, car in enumerate(caracteristicas_per_fold):
        print("FOLD_EJECUCION::> ", fold_ejecucion)
        print(f"Fold-> {x}: \n{car}")            
                
       
