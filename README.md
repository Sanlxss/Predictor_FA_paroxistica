# Predicción de Fibrilación Auricular Paroxística mediante Aprendizaje Automático en Individuos Sanos

Este repositorio contiene el código y los recursos utilizados en el Trabajo de Fin de Grado titulado "Predicción de Fibrilación Auricular Paroxística mediante Aprendizaje Automático en Individuos Sanos" presentado en la Escuela Técnica Superior de Ingeniería de la Universidad de Santiago de Compostela.

## Descripción

El objetivo de este proyecto es desarrollar y validar un modelo de predicción de fibrilación auricular (FA) paroxística en individuos sanos utilizando técnicas de aprendizaje automático. A diferencia de otros enfoques que se centran en la detección de FA en individuos ya enfermos, este estudio busca anticipar la aparición de la enfermedad en personas aún sanas.

Para ello, se ha utilizado un histórico de 58,582 electrocardiogramas (ECGs) de 11,636 pacientes, que cuentan con un identificador y un total de 493 características electrocardiográficas.


## Configuración del Entorno

### Requisitos Previos

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Pasos para Configurar el Entorno

1. **Descargar Miniconda**:
    ```sh
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    ```

2. **Instalar Miniconda**:
    ```sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    ```

3. **Crear y Activar un Entorno con Conda**:
    ```sh
    conda create --name fibrilacion_env python=3.8
    conda activate fibrilacion_env
    ```

4. **Instalar los Paquetes Necesarios**:
    ```sh
    pip install -r requirements.txt
    ```

## Ejecución del Proyecto

### Entrenamiento del Modelo

En el fichero `run_models.py`, en la línea 87, se encuentran los comandos para el entrenamiento del modelo. Comentar la anterior que es la asociada a la de validación

### Validación del Modelo
En el fichero `run_models.py`, en la línea 86, se encuentran los comandos para el entrenamiento del modelo. Comentar la siguiente que es la asociada a la de entrenamiento
