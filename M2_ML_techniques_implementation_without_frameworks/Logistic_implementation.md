
# Logistic Regression Model Implementation
### Fermín Méndez García A01703366

## Requirements
- Python 3.7.3
- Pandas 1.0.3
- Numpy 1.18.1
- Matplotlib 3.1.3
- Dowload the dataset 'data.csv' and save it in the same folder as the code files.
- Dowload the file 'logistic_model.py' and save it in the same folder as the code files.
- Dowload the file 'main.py' and save it in the same folder as the code files.
- Run the file `main.py` to see the results.

## Introduction
This project content:
-Dataset
-Logistic_model.py file with the functions for the model
-Main.py file with the code for the logistic regression model
-Results folder with the plots of the model

## Dataset
The dataset used for this project is the "Iris" dataset, which contains 150 samples of 3 different species of Iris flowers. The dataset contains 4 features: sepal length, sepal width, petal length, and petal width. The dataset is available in the file `iris.csv`.

## Logistic_model.py

In order to keep clean the code, the functions for the model are in a separate file called `logistic_model.py`. 

As we know a logistic model given a data set needs an activation function, a cost function and a optimization algorithm.
For this case we are going to use:
- Activation function: Sigmoid function
- Cost function: Cross-entropy
- Optimization algorithm: Gradient descent

All the functions are explained below.
This code file contains the functions for preprocessing, activation, cost calculation, updating parameters, testing the model, and plotting results.

Inputs for the functions are:
- `df_x`: DataFrame with the input features. It must include a bias column of ones (you can use the function `include_bias` for this). The function expect a pandas DataFrame.
- `df_y`: DataFrame with the output labels. The function expect a pandas series with the class with the values 0 and 1. (Hot encoded).
- `alfa`: Learning rate.
- `periods`: Number of iterations for gradient descent.

### Preprocessing and Activation

The file `logistic_model.py` contains functions for preprocessing the data and activation function (sigmoid function):

- `include_bias(df_x)`: Adds a bias column of ones to the input DataFrame `df_x`.
- `actvfun(x)`: Implements the sigmoid activation function.

### Cost Calculation and Gradient Descent

- `costfun(y_pred, y_real)`: Calculates the cross-entropy cost between predicted and actual values.
- `updateParamsDesendentGradient(currentParams, df_x, df_y, alfa, periods)`: Implements gradient descent for updating model parameters and return errors list of each iteration.

### Testing the Model

- `testModel(df_x, df_y, params)`: Tests the model using the input features `df_x`, the output labels `df_y`, and the model parameters `params`. Returns the error of the model and the predicted labels.
(Auxiliar function in plot_model_result)

### Plotting Results

- ` plot_errors_GD(errors) `: Plots the error of the model. The function receives the error list `errors`. (This is the error list of the gradient descent algorithm). Use the function `updateParamsDesendentGradient` to get the error list.

- ` plot_model_result(params,df_x_train,df_x_test,df_y_train,df_y_test) `: Plots the results of the model. The function receives the model parameters `params`, the training input features `df_x_train`, the testing input features `df_x_test`, the training output labels `df_y_train`, and the testing output labels `df_y_test`. Use the function `testModel` to get the error of the model and the predicted labels.



## Main.py
This code file demonstrates the implementation of a logistic regression model using gradient descent for binary classification tasks. The model includes functions for preprocessing, activation, cost calculation, updating parameters, testing the model, and plotting results.


