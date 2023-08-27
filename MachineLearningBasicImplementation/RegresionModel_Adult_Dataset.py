
import pandas as pd
import numpy as np

#RegresionModel_Adult_Dataset.py
from logistic_model import include_bias, plot_errors_GD
from logistic_model import updateParamsDesendentGradient as train
from logistic_model import test_model, plot_model_result


df_x_train=pd.DataFrame({'nums': [1,2,3,4], 'nums2':[7,8,9,10]})
df_y_train=pd.DataFrame({'class1': [0,1,1,1]})
print(df_x_train)
print(df_y_train)
print(include_bias(df_x_train))


num_params=df_x_train.shape[1]
#Create a hypotesis
currentParams=np.random.rand(num_params)

print("Initial hypotesis: ",currentParams)

# train=updateParamsDesendentGradient(currentParams,df_x,df_y,alfa,periods):
alfa= 0.05 #Alfa is the learning rate
periods= 1000 #Is the number of repetitions training in the gradient desendent


#To use our train function with gradient desendent (GD) we need:
#1 - currentParams: Is the first hypotesis. A numpay array with the same len of the parameters that we can optimize including the bias
#2 - df_x: train data as pandas dataframe
#3 - df_y: pandas serie with hot encoded class. Only 1's and 0's
#4 - alfa: Learning rate
#5 - periods: number of iteration in the gradient desendent
[currentParams,errors]=train(currentParams,df_x_train,df_y_train['class1'],alfa,periods)
#The result is a updated hypotesis and a list of the errors obtained for each iteration in GD

#---------------
plot_errors_GD(errors)

[e,pred]=test_model(currentParams,df_x_train,df_y_train)
#print("e ",e)
#print("pred",pred)

plot_model_result(currentParams,df_x_train,df_x_train,df_y_train,df_y_train)

#Logistic regresion model by Fermín Méndez A01703366

#1. Import the data set to clasify

#2. Clean the data and define train and test data: df_x_train, df_y_train, df_x_test and df_y_tet.

#3 Call the logistic model functions