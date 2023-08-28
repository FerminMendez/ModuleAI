
# Logistic regresion model by Fermín Méndez A01703366
import pandas as pd
import numpy as np

# main.py

#Import from logistic model functions 
from logistic_model import include_bias, plot_errors_GD
from logistic_model import updateParamsDesendentGradient as train
from logistic_model import test_model, plot_model_result
from logistic_model import scale_dataframe



######################################### 1. Import the dataset 
# Read the CSV file into a DataFrame

# Specify the column names
names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
         'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
         'hours-per-week', 'native-country', 'class']
#Read the original files
df = pd.read_csv('adult.data', names=names)
df_test=pd.read_csv('adult.test', names=names)

######################  2. Clean the data and define train and test data: df_x_train, df_y_train, df_x_test and df_y_tet.

#Define y_train and y_test
#train
df_y_train= df[['class']]
df_y_train= pd.get_dummies(df_y_train, columns=['class'], prefix='', prefix_sep='')
df_y_train=df_y_train.iloc[:, 1]
#test
df_y_test= df_test[['class']]
df_y_test= pd.get_dummies(df_y_test, columns=['class'], prefix='', prefix_sep='')
df_y_test=df_y_test.iloc[:, 1]

#Define x train and x test
#Selected columns
selected=['age','capital-gain', 'capital-loss', 'hours-per-week']
#train
df_x_train=df[selected] #Select the numeric columns
df_x_train=scale_dataframe(df_x_train)
df_x_train=include_bias(df_x_train)
#test
df_x_test=df_test[selected] #Select the numeric columns
df_x_test=scale_dataframe(df_x_test)
df_x_test=include_bias(df_x_test)


#df_y_test=df_test[['class']]
#df_x_test=df_test.drop(columns=['class'])


########################################## 3 Call the logistic model functions

num_params = df_x_train.shape[1] #Number of features
# Create a hypotesis
currentParams = np.random.rand(num_params)
intialGuess = currentParams.copy()
print("Initial hypotesis: ", intialGuess)

# train=updateParamsDesendentGradient(currentParams,df_x,df_y,alfa,periods):
alfa = 0.05  # Alfa is the learning rate
periods = 3000 # Is the number of repetitions training in the gradient desendent

#train with our gradient desendent (GD) function
[currentParams, errors] = train(currentParams, df_x_train, df_y_train, alfa, periods)
# The result is a updated hypotesis and a list of the errors obtained for each iteration in GD

plot_errors_GD(errors)
plot_model_result(currentParams, df_x_train,df_x_test, df_y_train, df_y_test)

#Analize out results
print('Stats train')
print('CurrentParams:',currentParams)
test_model(currentParams,df_x_train,df_y_train,True)
print('Stats test')
test_model(currentParams,df_x_test,df_y_test,True)

#Compare with the random params

print('Initial Random guess')
print('Intial params:',intialGuess)
print('Stats train')
test_model(intialGuess,df_x_train,df_y_train,True)
print('Stats test')
test_model(intialGuess,df_x_test,df_y_test,True)



