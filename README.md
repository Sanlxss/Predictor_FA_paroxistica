# Predicción de Fibrilación Auricular Paroxística mediante Aprendizaje Automático en Individuos Sanos

Este repositorio contiene el código y los recursos utilizados en el Trabajo de Fin de Grado titulado "Predicción de Fibrilación Auricular Paroxística mediante Aprendizaje Automático en Individuos Sanos" presentado en la Escuela Técnica Superior de Ingeniería de la Universidad de Santiago de Compostela.

## Descripción

El objetivo de este proyecto es desarrollar y validar un modelo de predicción de fibrilación auricular (FA) paroxística en individuos sanos utilizando técnicas de aprendizaje automático. A diferencia de otros enfoques que se centran en la detección de FA en individuos ya enfermos, este estudio busca anticipar la aparición de la enfermedad en personas aún sanas.

Para ello, se ha utilizado un histórico de 58,582 electrocardiogramas (ECGs) de 11,636 pacientes, que cuentan con un identificador y un total de 493 características electrocardiográficas.

## Estructura de Directorios

```plaintext
.
├── Fibrillation_Prog_Binario
│   ├── LICENSE
│   ├── README.md
│   ├── __pycache__
│   │   ├── fibrillation.cpython-311.pyc
│   │   ├── ...
│   ├── fibrillation.py
│   ├── lib
│   │   ├── __pycache__
│   │   │   ├── base_models.cpython-310.pyc
│   │   │   ├── ...
│   │   ├── base_models.py
│   │   ├── create_latent_ode_model.py
│   │   ├── diffeq_solver.py
│   │   ├── encoder_decoder.py
│   │   ├── latent_ode.py
│   │   ├── likelihood_eval.py
│   │   ├── ode_func.py
│   │   ├── parse_datasets.py
│   │   ├── plotting.py
│   │   ├── utils.py
│   ├── run_models.py
├── requirements.txt
