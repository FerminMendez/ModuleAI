# -*- coding: utf-8 -*-

# Logistic regresion model by Fermín Méndez A01703366
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# main.py

# Import other libraries


# 1. Import the dataset
# Read the CSV file into a DataFrame

# Read the original files
df_mat = pd.read_csv('student-mat.csv',sep=';')
df_por = pd.read_csv('student-por.csv',sep=';')
#common_columns = ["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet"]

"""## 1.2 Obtenemos un dataset "data" estandarizado

1.   Vamos a crear un booleano donde math=1 significa que el registro corresponde a la materia de matemáticas mientras que
2.   Unamos el dataset en Data
3.   Sustituimos "yes" por un 1 y "no" por un 0
4.   Creamos una promedio de todas las calificaciones
"""

# 1
df_mat["math"]=1
df_por["math"]=0
# 2
data= pd.concat([df_mat, df_por], axis=0)
data.reset_index(drop=True, inplace=True)

# 3
data = data.replace({'yes': 1, 'no': 0})
data['grade']=data[['G1','G2','G3']].mean(axis=1)
data['math'].describe()
#Map values
#School
data = data.replace({'GP': 1, 'MS': 0})
#Sex
data = data.replace({'F': 1, 'M': 0})
# adress
data = data.replace({'U': 1, 'R': 0})
# famsize
data = data.replace({'LE3': 1, 'GT3': 0})
# P status
data = data.replace({'T': 1, 'A': 0})

"""# 2 Modelo de predicción"""

from sklearn.metrics import mean_squared_error, r2_score

def result_model(model,X_train, X_test, y_train, y_test):
  y_fit_train = model.predict(X_train)
  mse_train = mean_squared_error(y_train, y_fit_train)
  r2_train = r2_score(y_train, y_fit_train)
  print("TRAIN RESULTS:")
  print("MSE: ",mse_train)
  print("R^2: ",r2_train)
  y_fit_test = model.predict(X_test)
  mse_test = mean_squared_error(y_test, y_fit_test)
  r2_test = r2_score(y_test, y_fit_test)
  print("\nTEST RESULTS:")
  print("MSE: ",mse_test)
  print("R^2: ",r2_test)
  return

"""## 2.1 Modelo de regresión lineal"""

from sklearn.model_selection import train_test_split
df=data.copy()
selected = ['school','sex','address','famsize','Pstatus','internet', 'studytime', 'absences', 'failures', 'traveltime', 'health','schoolsup', 'famsup', 'paid', 'higher', 'famrel', 'freetime', 'Medu', 'Fedu','Dalc', 'Walc','math']
target = ['grade']
df_X = df[selected]
df_Y = df[target]
X_train, X_test, y_train, y_test = train_test_split(df_X, df_Y, test_size=0.2, random_state=42)

"""### 2.1.1 Separacion de datos en train y test

Vamos a correr un modelo de regresión lineal
"""

from sklearn.linear_model import LinearRegression
model_lr_1 = LinearRegression(fit_intercept=True)

model_lr_1.fit(X_train, y_train)

y_fit = model_lr_1.predict(X_test)

result_model(model_lr_1, X_train, X_test, y_train, y_test)

"""## 2.2 Mejora al modelo lineal

En este caso eliminmos las variables con correlación por debajo del 0.1 y 'Fedu' y 'Walc' que tenian alta correlación con 'Medu' y 'Dalc'
"""

df=data.copy()
selected = ['school','address','internet', 'studytime', 'failures', 'traveltime', 'schoolsup', 'higher', 'Medu', 'Walc','math']
target = ['grade']
X = df[selected]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_lr_2 = LinearRegression(fit_intercept=True)
model_lr_2.fit(X_train, y_train)
y_fit = model_lr_2.predict(X_test)
mse = mean_squared_error(y_test, y_fit)
result_model(model_lr_2, X_train, X_test, y_train, y_test)
