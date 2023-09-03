
# Regresion Model implamantation using a framekwork
### Fermín Méndez García A01703366

## Requirements
- Python 3.7.3
- Pandas 1.0.3
- Numpy 1.18.1
- Matplotlib 3.1.3
- Dowload the file ...
- Run the file `main.py` to see the results.

## Introduction
This project content:
-Dataset ...
-Logistic_model.py file with the functions for the model
-Main.py file with the code for the logistic regression model
-Results folder with the plots of the model

## Dataset
About the dataset...

To train this model we will focus only in numeric variables and we will use the following variables:
- `age`: Age of the person.
- `capital-gain`: Income from investment sources, apart from wages/salary.
- `capital-loss`: Losses from investment sources, apart from wages/salary.
- `hours-per-week`: Hours worked per week.



## Auxiliar.py





## Main.py
This code file demonstrates the implementation of a logistic regression model using gradient descent for binary classification tasks. The model includes functions for preprocessing, activation, cost calculation, updating parameters, testing the model, and plotting results.

We defined the strong functinos in logistic_model.py and we use them in main.py to train the model and test it.


### Cleaning the data

...

### Transforming the data

To transform the data we use the hot encoding method to transform the categorical variables to numeric variables. We use the function `get_dummies` from pandas to do this.
We use the function `scale_dataframe` to scale the data using the min-max scaling method.
Use the function `include_bias` to add a bias column of ones to the input DataFrame.

### Choosing a model


### Training the model

We use the function `updateParamsDesendentGradient` to train the model. This function implements gradient descent for updating model parameters and return errors list of each iteration. The function returns the model parameters and the error list.

### Testing the model

We use the function `testModel` to test the model. This function tests the model using the input features `df_x`, the output labels `df_y`, and the model parameters `params`. Returns the error of the model and the predicted labels. Additional parameter `stats` is used to print the accuracy , precision and recall of the model. (Auxiliar function in plot_model_result)

### Plotting the results

We use the function `plot_errors_GD` to plot the error of the model. The function receives the error list `errors`. (This is the error list of the gradient descent algorithm). Use the function `updateParamsDesendentGradient` to get the error list.

We use the function `plot_model_result` to plot the results of the model. The function receives the model parameters `params`, the training input features `df_x_train`, the testing input features `df_x_test`, the training output labels `df_y_train`, and the testing output labels `df_y_test`. Use the function `testModel` to get the error of the model and the predicted labels.

## Results

To see the results of the model run the file `main.py`. 

We use the following parameters for the model:
alfa = 0.05 
periods = 3000  

The results of the model are:

Initial hypotesis:  [0.17557402 0.14087837 0.72957655 0.59168058 0.76821912]
Finished gradient iterations

The final error of the current hypothesis is 9862.5560


Stats train

CurrentParams: [-2.13694879  1.45386656  1.51824624  1.17568991  1.27390486]
Error: 9862.4073
Precision: 0.9984
Recall: 0.7654
F1-Score: 0.8665
Accuracy: 0.7644
Stats test

Error: 4818.5420
Precision: 0.9975
Recall: 0.7708
F1-Score: 0.8696
Accuracy: 0.7694

Intial params: [0.17557402 0.14087837 0.72957655 0.59168058 0.76821912]
Stats train
Error: 3362.6589
Precision: 0.2408
Recall: 1.0000
F1-Score: 0.3881
Accuracy: 0.2408

Stats test
Error: 1645.5437
Precision: 0.2362
Recall: 1.0000
F1-Score: 0.3822
Accuracy: 0.2362


### Improving the model




## Conclusions

The initial random guess has less error has less error than the trained model. But has a recall of 1 this mean that the model is overfitting. The trained model has a recall of 0.77 and a precision of 0.99. 
This means that the model is not overfitting and has a good performance. 
The model has a good accuracy of 0.76. The model can be improved by using more features and more data.
In both cases the train and test models are very similar, this means that the model is not overfitting.
