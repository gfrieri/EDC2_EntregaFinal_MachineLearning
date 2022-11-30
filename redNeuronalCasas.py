#Por: Jesús Durán y Giuliano Frieri
#Estructura del Computador 2
#Noviembre 2022
import tensorflow as tf
import numpy as np
import pandas as pd

#Dataset tomado de: https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset
df_casas = pd.read_csv('https://raw.githubusercontent.com/gfrieri/EDC2_EntregaFinal_MachineLearning/main/House_Rent_Dataset.csv')

dentrada = df_casas.loc[:,'Size']
dsalida = df_casas.loc[:,'Rent']

dentrada = np.array(dentrada, dtype=float)
dsalida = np.array(dsalida, dtype=float)

#Capa densa, conexiones para todas las neuronas
#Unidades: Cantidad de Neuronas en la capa
#input_shape: Se le indica la cantidad de neuronas de la capa de entrada
#Se utiliza un modelo Secuencial ->
oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
oculta2 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([oculta1, oculta2, salida])

#Optimizador usado: Algoritmo Adam con tasa de aprendizaje 0.001
#Función de perdida: Error cuadrático medio
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='mean_squared_error'
)

print("Inicio entramiento")
#Valores de entrada y salida: Tamaño en mt^2 y renta
#Cantidad de vueltas: 10
historial = modelo.fit(dentrada, dsalida, epochs=100, verbose=False)
print("Fin entrenamiento")

'''
#Esto es para visualizar el aprendizaje con base a la función de perdida
import matplotlib.pyplot as plt
plt.xlabel("# Epochs")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])
'''

#Predicción con un valor de entrada arbitrario
#Ejemplo, 2000mt^2
datos_pred=(2000)
resultado = modelo.predict([datos_pred])
print("El arriendo de un inmueble de", datos_pred, "metros cuadrados es aproximadamente: " + str(resultado[0,0]) + " rupias")
