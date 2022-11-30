import tensorflow as tf
import numpy as np
import os

dentrada = np.loadtxt('dentrada.txt')
dsalida = np.loadtxt('dsalida.txt')

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
datos_pred = float(input('Ingrese el tamaño del inmueble en metros cuadrados: '))
resultado = modelo.predict([datos_pred])
resultado = np.array(resultado, dtype=float)

np.savetxt('prediccion.txt', [datos_pred])
np.savetxt('resultado.txt', resultado)

os.remove('dentrada.txt')
os.remove('dsalida.txt')