
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#logistic_model.py

#This function will add a 'ones' column in the first position
def include_bias(df_x):
    num_rows=df_x.shape[0]
    df_x.insert(loc = 0,column="ones",value = np.ones(num_rows).astype('float16'))
    return df_x

#Define activation function = Sigmoid function
def actvfun(x):
    return 1 / (1 + np.exp(-x))

#Define cost function
def costfun(y_pred,y_real):
    #print("pred shape", type(y_pred), y_pred.shape)
    #print("real shape", type(y_real), y_real.shape)
    epsilon = 1e-15  # Small constant to prevent division by zero
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip predictions to avoid log(0)
    ce = - np.sum(y_real * np.log(y_pred))
    return ce


def updateParamsDesendentGradient(currentParams, df_x, df_y, alfa, periods):
    num_rows = df_x.shape[0]
    num_col = df_x.shape[1]
    debug = False
    if debug:
        if num_col == currentParams.size:
            print("params size correct")
        else:
            print("params size INCORRECT")
        if df_y.shape[0] == num_rows:
            print("df_x and df_y size are CORRECT")
        else:
            print("df_x df_y size INCORRECT")

    delta = 0.00000001
    const_alfa_m = alfa / num_rows  # this is the part of [lr/m] in -> new_teta = teta - [lr/m] * gradient
    errors = list()

    for p in range(periods):
        temp = currentParams
        evaluated_h = df_x.dot(currentParams)
        if(debug):
            print("params applied:\n",evaluated_h)
        # Apply activation function to make the regression logistic model
        evaluated_h = evaluated_h.apply(actvfun)
        if(debug):
            print("activation function applied:\n",evaluated_h)
        diff = evaluated_h - df_y
        if(debug):
            print("diff:\n",diff)
        for i in range(num_col):
            x = diff.dot(df_x.iloc[:, i])
            temp[i] = currentParams[i] - const_alfa_m * x

        currentParams = temp
        curr_error = costfun(evaluated_h, df_y)
        errors.append(curr_error)

        if curr_error < delta:
            print("Repeated %i times to get %f error" % (p, delta))
            break

    print("Finished gradient iterations")
    return [currentParams, errors]


#Once we train our model with GD we check how is learning the model to adjust the learning rate or increase the num of iterations.
def plot_errors_GD(errors):
    plt.plot(errors, color='blue', marker='o', linestyle='-')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title('Error Progression during Gradient Descent')
    plt.grid(True)
    plt.show()
    final_error = errors[-1]
    print(f"The final error of the current hypothesis is {final_error:.4f}")


#Test our logistic model
#Remeber the parameters of the function are:
#1.Params: The hypotesis coeficients
#2.df_x_test: Pandas dataframe to test
#3. df_y_test: Pandas serie of the classifications in hot encoded (only 1's or 0's)
    
def test_model(params,df_x_test,df_y_test):
    predictions=df_x_test.dot(params)
    predictions=predictions.apply(actvfun)
    error=costfun(predictions,df_y_test)[0]
    print("Model tested")
    #print("Type of error: ",type(error))
    return [error,predictions]

def plot_model_result(params,df_x_train,df_x_test,df_y_train,df_y_test):
    [error_train,predictions_train]=test_model(params,df_x_train,df_y_train)
    [error_test,predictions_test]=test_model(params,df_x_test,df_y_test)
    # Assuming you have your df_x_train, predictions_train, df_y_train, df_x_test, predictions_test, and df_y_test calculated

    # Create an array of indices for plotting
    indices_train = df_x_train.index
    indices_test = df_x_test.index

    # Plotting the index of df_x_train versus predictions_train
    plt.figure(figsize=(12, 6))

    # Plot for train data
    plt.subplot(1, 2, 1)
    plt.scatter(indices_train, predictions_train, color='blue', label='Predictions')
    plt.scatter(indices_train, df_y_train, color='orange', label='Actual Labels')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Train Data: Index vs Predictions and Actual Labels')
    plt.legend()

    # Plot for test data
    plt.subplot(1, 2, 2)
    plt.scatter(indices_test, predictions_test, color='blue', label='Predictions')
    plt.scatter(indices_test, df_y_test, color='orange', label='Actual Labels')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Test Data: Index vs Predictions and Actual Labels')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    print("The train error is %f " % error_train)
    print("The test error is %f " % error_test)
    return [predictions_train,predictions_test]