#Por: Jesús Durán y Giuliano Frieri
#Estructura del Computador 2
#Noviembre 2022
import numpy as np
import pandas as pd
import os
from os.path import exists

if (exists('prediccion.txt') & exists('resultado.txt')):
    datos_pred = np.loadtxt('prediccion.txt')
    resultado = np.loadtxt('resultado.txt')

    print("El arriendo de un inmueble de", datos_pred, "metros cuadrados es aproximadamente:", resultado,"rupias")
    os.remove('prediccion.txt')
    os.remove('resultado.txt')
else:
    #Dataset tomado de: https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset
    df_casas = pd.read_csv('https://raw.githubusercontent.com/gfrieri/EDC2_EntregaFinal_MachineLearning/main/House_Rent_Dataset.csv')

    dentrada = df_casas.loc[:,'Size']
    dsalida = df_casas.loc[:,'Rent']

    dentrada = np.array(dentrada, dtype=float)
    dsalida = np.array(dsalida, dtype=float)

    np.savetxt('dentrada.txt', dentrada)
    np.savetxt('dsalida.txt', dsalida)
