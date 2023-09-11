
# Regresion Model implamantation using a framekwork
### Fermín Méndez García A01703366

## Requirements
- Python 3.7.3
- Pandas 1.0.3
- Numpy 1.18.1
- Scikit-learn 0.22.2
- Dowload the file student-mat.csv from the UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Student+Performance
- Dowload the file student-por.csv from the UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Student+Performance
- Run the file `main.py` to see the results.

## Introduction
This project content:
-Dataset Student
-main.py file with the complete model
-Results

The objective is predict the grade of the students using the dataset Student. 
## Dataset
About the dataset. You can find more information in student.txt.
The dataset contains 33 variables and 649 instances. 
The dataframe data contains is the combination of por and mat dataframes.
The dataframe data has some differences with the original dataset. The difference are:
- All the categorical variables are transformed to numeric variables. Specifically, boolean variables are transformed to 0 and 1.
- Add a feature math to indicate if the student is taking the math course.
- Add a new feature called grade that is the average of the final grade.

## Linear regression model

### Linear regression model v1
To train this model we will use all the variable that we have available. The variables are:
- school
- sex
- address
- famsize
- Pstatus
- internet
- studytime
- absences
- failures
- traveltime
- health
- schoolsup
- famsup
- paid
- higher
- famrel
- freetime
- Medu
- Fedu
- Dalc
- Walc
- math

To train our model we separated the data in train and test. We use 80% of the data for training and 20% for testing. We use the function `train_test_split` from sklearn to do this.

We use the function `linear_regression` to train the model. This function implements the linear regression algorithm selex the option intercept to indicate if we want to use the intercept or not. The function returns the model parameters `params` and the error of the model `error`.

In order to manage the output of the predictions we create the function:
result_model(model,X_train, X_test, y_train, y_test) that receives the model, the train and test data and the train and test labels. This function prints the error of the model in the train and test data.
We use the function `predict` to predict the values of the train and test data. This function receives the model and the train and test data. The function returns the predictions of the train and test data.
### Linear regression model v2

In this model we will use less variables. Using the correlation matrix we can see that the variables that have more correlation with the grade are: 
- school
- address
- internet
- studytime
- failures
- traveltime
- schoolsup
- higher
- Medu
- Walc
- math
We will use the variables that have a correlation greater than 0.1 with the grade. Besides
we exclude the variables that had high correlation with other. For example we Medu and Fedu have a correlation of 0.62. We will use only Medu because it has a higher correlation with the grade.
## Results

1.  Regresión lineal 1

*  TRAIN RESULTS:
MSE:  7.1524760537283365
R^2:  0.30791877023035896

*  TEST RESULTS:
MSE:  8.663829527330574
R^2:  0.1637783640586179

2.  Modelo de regresión lineal 2

*  TRAIN RESULTS:
MSE:  7.309731260988808
R^2:  0.2927025882521228

*  TEST RESULTS:
MSE:  8.630070998710131
R^2:  0.1670366936391927

## Conclusions

Both models are very similar. The first model has a better R^2 in the train data but the second model has a better R^2 in the test data. The second model has a better R^2 in the test data because it has less variables and it is less prone to overfitting. The second model is better than the first model because it has less variables and it is less prone to overfitting.

Also we notice that the linear regression model is not the best model to predict the grade of the students. The R^2 is very low. We can try to use other models to predict the grade of the students. For example, we can use a decision tree model, a random forest model or a neural network model.

The main objective of this folder is to show how to implement a model using a framework for this reason we only use a linear regression model.
But in this folder is available a .ipynb file with the same model but with more explanation and more details and the implementation of other models like decision tree, random forest and neural network.
