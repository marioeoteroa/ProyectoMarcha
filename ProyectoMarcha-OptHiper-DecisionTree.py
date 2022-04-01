# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 20:32:45 2022

@author: Mario
"""

#pip install openpyxl



# Importar librerías aquí

import glob
import os
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_validate
from statistics import mean
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.svm import SVR

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model

from sklearn.model_selection import cross_validate
from statistics import mean
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor


from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

from matplotlib.pyplot import figure

all_files = glob.glob('D:/Users/Mario/Documents/TrabajoRevista/*.xlsx')

len(all_files)

targets = pd.read_excel('D:/Users/Mario/Documents/TrabajoRevista/Target/ParametrosMarchaNUEVO.xlsx')

targets.shape

#Se definen los hiperparametros a optimizar
max_depth = 2 #ponerlo de 10 en 10 hasta 100
minsamplesplit = 2 #dejarlo de 1 en 1 hasta 10
minsampleleaf = 2 #dejarlo de 1 en 1 hasta 10
#list(range(1, max_depth+1,10))
param_grid = {"model__max_depth": list(range(1, max_depth+1)),
              "model__min_samples_split": list(range(1, minsamplesplit+1)),
              "model__min_samples_leaf": list(range(1, minsampleleaf+1))
                                        }

import time
start_time = time.time()

print('-------------------INICIO PROCESAMIENTO-----------------')

DataDecisionTree = []
DataDecisionTreeINDIVIDUAL = []

z = 0


for f in all_files:
    train_features = pd.read_excel(f)
    # Train, test split.
    data_split1 = train_test_split(train_features, targets, test_size=0.3, random_state=4444)
    X_train, X_test, targets_train, targets_test = data_split1

    # Find the best model per target.
    for target in range(targets.shape[1]):
        # Getting the target.
        y_train = targets_train.iloc[:, target]
        y_test = targets_test.iloc[:, target]

    # Model definition.
        pipe = Pipeline([("minmax", MinMaxScaler((-1, 1))),
                          ("model", DecisionTreeRegressor(random_state=4444))])
    # By default GridSearchCV uses a 5-kfold validation strategy.
        search = GridSearchCV(pipe, param_grid, cv=10, n_jobs=-1)  
        saveinfo = search.fit(X_train.values, y_train.values)
        informacion = search.cv_results_
        

    # Getting the test score.
        y_hat_test = search.predict(X_test.values)
        test_score = r2_score(y_test.values, y_hat_test)

    # Printing stats.
        #print(f"Columna: {targets.columns[target]}")
        #print(f"Best CV score: {search.best_score_:0.3f}")
        #print(f"Test score: {test_score:0.3f}")
        #print(f"Best Parameters:\n {search.best_params_}")
        #print("")

    #Save stats
        buscarstd = pd.DataFrame(informacion)
        mejorstd = buscarstd[['std_test_score']].iloc[search.best_index_]
        mejorstd = pd.DataFrame(mejorstd)
        mejorstd = mejorstd.iloc[0, 0]
        mejordepth = buscarstd[['param_model__max_depth']].iloc[search.best_index_]
        mejordepth = pd.DataFrame(mejordepth)
        mejordepth = mejordepth.iloc[0, 0]
        mejorleaf = buscarstd[['param_model__min_samples_leaf']].iloc[search.best_index_]
        mejorleaf = pd.DataFrame(mejorleaf)
        mejorleaf = mejorleaf.iloc[0, 0]
        mejorsplit = buscarstd[['param_model__min_samples_leaf']].iloc[search.best_index_]
        mejorsplit = pd.DataFrame(mejorsplit)
        mejorsplit = mejorsplit.iloc[0, 0]
    
        DataDecisionTree.append([f,targets.columns[target],search.best_score_,mejorstd,test_score,mejordepth,mejorleaf,mejorsplit,search.best_params_])
        DataDecisionTreeINDIVIDUAL.append([f,targets.columns[target],informacion])
        
    z = z+1
    print(z)

print('--------------------------------------------------------------')
print('PROCESAMIENTO FINALIZADO EXITOSAMENTE!!!')
print('--------------------------------------------------------------')

TIEMPO = (time.time() - start_time)/60

print("--- %s Minutos ---" % (TIEMPO))

#search.best_index_

DataDecisionTreeINDIVIDUAL = pd.DataFrame(DataDecisionTreeINDIVIDUAL)
DataDecisionTree = pd.DataFrame(DataDecisionTree)
DataDecisionTreeINDIVIDUAL.to_csv('DataDecisionTreeINDIVIDUALFINAL.csv')
DataDecisionTree.to_csv('DataDecisionTreeFINAL.csv')

#datosguardados = pd.read_csv('D:/Users/Mario/Documents/TrabajoRevista/DataDecisionTreeFINAL.csv')
#datosguardados

#DataDecisionTreeINDIVIDUAL.iloc[0,2]
#NewDataDecisionTreeINDIVIDUAL = pd.DataFrame(DataDecisionTreeINDIVIDUAL.iloc[0,2])
#NewDataDecisionTreeINDIVIDUAL = NewDataDecisionTreeINDIVIDUAL[['mean_test_score','param_model__max_depth','param_model__min_samples_leaf','param_model__min_samples_split','params']]
#NewDataDecisionTreeINDIVIDUAL