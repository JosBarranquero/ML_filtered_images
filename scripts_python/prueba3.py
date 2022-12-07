from time import time
import utilidades as u
import cv2 as cv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


# Listas que contendrán las imágenes originales y filtradas
originales = []
filtradas = []

for i in range(1, 101):
    ruta_original = '{0}imagen{1}.bmp'.format('../originales/', i)
    ruta_filtrada = '{0}imagen{1}-bajo.bmp'.format('../filtradas/', i)

    # Leemos las imágenes
    imagen_original = cv.imread(ruta_original, cv.IMREAD_GRAYSCALE)
    imagen_filtrada = cv.imread(ruta_filtrada, cv.IMREAD_GRAYSCALE)

    # Guardamos las imágenes como pasando de un array de 32x32 a un vector de (32*32) elementos
    """originales.append(np.array(cv.imdecode(imagen_original, cv.IMREAD_UNCHANGED)))
    filtradas.append(np.array(cv.imdecode(imagen_filtrada, cv.IMREAD_UNCHANGED)))"""
    originales.append(np.reshape(imagen_original, newshape=(1, np.product(imagen_original.shape)))[0])
    filtradas.append(np.reshape(imagen_filtrada, newshape=(1, np.product(imagen_filtrada.shape)))[0])

# Convertimos las listas a un Pandas DataFrame
df_original = pd.DataFrame(np.array(originales, dtype=np.uint8))
df_filtrada = pd.DataFrame(np.array(filtradas, dtype=np.uint8))

# Creamos los conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df_original, df_filtrada, test_size=0.10)
# X_train = df_original.head(75)
# X_test = df_original.tail(25)
# y_train = df_filtrada.head(75)
# y_test = df_filtrada.tail(25)

# Entrenando el árbol de regresíon
regressor = LinearRegression()
# regressor = svm.SVR(kernel='linear', C=100.0)
t0 = time()
regressor.fit(X_train, y_train)
print("Training Time:", round(time()-t0, 3), "s")

# Se predice en base al conjunto de pruebas
t0 = time()
y_pred = regressor.predict(X_test)
print("Predicting Time:", round(time()-t0, 3), "s")

# Vemos el error cometido
mse = mean_squared_error(y_pred, y_test)

# obtenemos las imagenes en base a las predicciones
for i in range (0, len(y_pred)):
    imagen_pred = np.reshape(y_pred[i], newshape=(32, 32))
    imagen_pred = imagen_pred.astype(np.uint8)
    
    # imagen_real = y_test.loc[i+75,:]
    # imagen_real = np.reshape(imagen_real.to_numpy(dtype='uint8'), newshape=(32, 32))

    cv.imwrite('../prediccion/prediccion-{0}.bmp'.format(i), imagen_pred)
    # cv.imwrite('../prediccion/prediccion-{0}-real.bmp'.format(i), imagen_real)

# obtenemos las imagenes reales
indices = y_test.index
for index, value in enumerate(indices):
    imagen_real = y_test.loc[value,:]
    imagen_real = np.reshape(imagen_real.to_numpy(dtype='uint8'), newshape=(32, 32))
    
    cv.imwrite('../prediccion/prediccion-{0}-real.bmp'.format(index), imagen_real)

print("MSE = {0}".format(mse))

# La muestro por pantalla, porque no sé mostrarlo con imshow
# imagen_pred = np.reshape(y_pred[0], newshape=(32, 32))
# imagen_pred = imagen_pred.astype(np.uint8)
# imagen_real = np.reshape(y_test.head(1).to_numpy(dtype='uint8'), newshape=(32, 32))

# cv.imshow('Predicho', imagen_pred)
# cv.imshow('Real', imagen_real)

# cv.waitKey(0)