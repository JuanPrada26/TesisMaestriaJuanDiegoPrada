import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
import seaborn as sns

from datetime import datetime

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_privacy
from sklearn.metrics import mean_squared_error, mean_absolute_error,median_absolute_error
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy



"""**Prueba CON PD**"""
def LSTM_CON_PD(datos,fechas,nodos1,nodos2,paciencia,epocas,batch,window_size,t_pridiccion,norm_clip,ruido,microBatches,lr):

    
    estandarizacion = MinMaxScaler().fit(datos)
    scaled_data = estandarizacion.transform(datos)



    # dividir en train, test
    X, y = [], []
    Xf,yf = [],[]

    for i in range(len(scaled_data) - window_size - t_pridiccion):
        X.append(scaled_data[i:i+window_size])
        y.append(scaled_data[i+window_size:i+window_size+t_pridiccion])

        Xf.append(fechas[i:i+window_size])
        yf.append(fechas[i+window_size:i+window_size+t_pridiccion])

    X, y = np.array(X), np.array(y)
    Xf,yf = np.array(Xf),np.array(yf)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)


    fecha_X_train, fecha_X_test, fecha_y_train, fecha_y_test = train_test_split(Xf, yf, test_size=0.1, shuffle=False)



    model = Sequential()

    model.add(LSTM(nodos1,activation= "tanh", input_shape=(window_size,1)))
    model.add(Dense(nodos2, activation="relu"))
    model.add(Dense(t_pridiccion , activation="linear"))

 

    if batch % microBatches != 0:
        raise ValueError('Batch size should be an integer multiple of the number of microbatches')



    # agregar la privacidad diferencial en el optimizador 
    optimizer = tensorflow_privacy.DPKerasSGDOptimizer(
        l2_norm_clip=norm_clip,
        noise_multiplier=ruido,
        num_microbatches=microBatches,
        learning_rate=lr)

    # Función de pérdida para regresión
    loss = tf.keras.losses.MeanSquaredError(reduction=tf.losses.Reduction.NONE)


    model.compile(optimizer=optimizer, loss=loss)

    early_stopping = EarlyStopping(monitor='loss', patience=paciencia, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=epocas,validation_split = 0.2, verbose=1, batch_size=batch,shuffle = False, callbacks=[early_stopping])


    # guardar los archivo a usar en la carpeta 
    rutaAGuardar = f'Modelo {nodos1} - nodos1 - {nodos2} - nodos2 - {epocas} Epocas con PD.keras'
    model.save(rutaAGuardar)

    # hacer la predicción  y desestandarizar
    y_hat = model.predict(X_test, verbose=1)

    y_hat = estandarizacion.inverse_transform(y_hat)

    #desestandarizar y_test
    y_test1 = y_test.reshape(-1, 1)

    y_test1 = estandarizacion.inverse_transform(y_test1)

    y_test1 = y_test1.reshape(-1,24,1)

    predicciones24 = []
    reales24 = []
    for i in range(24):
            
        pred = []
        for Predicciones in y_hat:
            pred.append(Predicciones[i])
            
        real = []
        for reales in y_test1:
            real.append(reales[i])
            
        predicciones24.append(pred)
        reales24.append(real)
        



    MAES = {}
    RMSE = {}
    ER_Medios = {}
    ER_Medianos = {}
    epsilon = 1e-10
    for i in range(24):
        MAE = round(mean_absolute_error(predicciones24[i],reales24[i]),2)
        MSE = round(mean_squared_error(reales24[0],predicciones24[i]),2)
        Error_Relativo_Medio = round((np.mean(np.abs((np.array(reales24[i]) - np.array(predicciones24[i])) / (np.array(reales24[i])+epsilon)))*100),2)
        Error_Relativo_Mediano = round((np.median(np.abs((np.array(reales24[i]) - np.array(predicciones24[i])) / (np.array(reales24[i])+epsilon)))*100),2)
        
        
        MAES[i] = MAE
        RMSE[i] = np.sqrt(MSE)
        ER_Medianos[i] = Error_Relativo_Mediano
        ER_Medios[i] = Error_Relativo_Medio

        


    return history, y_hat,y_test1,fecha_y_test, MAES,RMSE,ER_Medianos,ER_Medios


"""**Prueba SIN PD**"""
def LSTM_SIN_DP(datos,fechas,nodos1,nodos2,paciencia,epocas,batch,window_size,t_pridiccion,lr):

        
    estandarizacion = MinMaxScaler().fit(datos)
    scaled_data = estandarizacion.transform(datos)

    window_size = 168
    t_pridiccion = 24


    # dividir en train, test
    X, y = [], []
    Xf,yf = [],[]

    for i in range(len(scaled_data) - window_size - t_pridiccion):
        X.append(scaled_data[i:i+window_size])
        y.append(scaled_data[i+window_size:i+window_size+t_pridiccion])

        Xf.append(fechas[i:i+window_size])
        yf.append(fechas[i+window_size:i+window_size+t_pridiccion])

    X, y = np.array(X), np.array(y)
    Xf,yf = np.array(Xf),np.array(yf)




    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)
    fecha_X_train, fecha_X_test, fecha_y_train, fecha_y_test = train_test_split(Xf, yf, test_size=0.1, shuffle=False)
        
    model = Sequential()

    model.add(LSTM(nodos1,activation= "tanh", input_shape=(window_size,1)))
    model.add(Dense(nodos2, activation="relu"))
    model.add(Dense(t_pridiccion , activation="linear"))

    model.compile(optimizer="Adam", loss='mse')

    early_stopping = EarlyStopping(monitor='loss', patience=paciencia, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=epocas,validation_split = 0.2, verbose=1, batch_size=batch,shuffle = False, callbacks=[early_stopping])



    # guardar los archivo a usar en la carpeta 
    rutaAGuardar = f'Modelo {nodos1} - nodos1 - {nodos2} - nodos2 - {epocas} Epocas sin PD.keras'
    model.save(rutaAGuardar)


        
    y_hat = model.predict(X_test, verbose=1)
    y_hat = estandarizacion.inverse_transform(y_hat)

    y_test1 = y_test.reshape(-1, 1)

    y_test1 = estandarizacion.inverse_transform(y_test1)

    y_test1 = y_test1.reshape(-1,24,1)


    predicciones24 = []
    reales24 = []
    for i in range(24):
            
        pred = []
        for Predicciones in y_hat:
            pred.append(Predicciones[i])
            
        real = []
        for reales in y_test1:
            real.append(reales[i])
            
        predicciones24.append(pred)
        reales24.append(real)
        



    MAES = {}
    RMSE = {}
    ER_Medios = {}
    ER_Medianos = {}
    epsilon = 1e-10
    for i in range(24):
        MAE = round(mean_absolute_error(predicciones24[i],reales24[i]),2)
        MSE = round(mean_squared_error(reales24[0],predicciones24[i]),2)
        Error_Relativo_Medio = round((np.mean(np.abs((np.array(reales24[i]) - np.array(predicciones24[i])) / (np.array(reales24[i])+epsilon)))*100),2)
        Error_Relativo_Mediano = round((np.median(np.abs((np.array(reales24[i]) - np.array(predicciones24[i])) / (np.array(reales24[i])+epsilon)))*100),2)
        
        
        MAES[i] = MAE
        RMSE[i] = np.sqrt(MSE)
        ER_Medianos[i] = Error_Relativo_Mediano
        ER_Medios[i] = Error_Relativo_Medio


    return history, y_hat,y_test1,fecha_y_test, MAES,RMSE,ER_Medianos,ER_Medios


