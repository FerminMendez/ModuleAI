
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#logistic_model.py


    
#This function will add a 'ones' column in the first position
def include_bias(df_x):
    num_rows=df_x.shape[0]
    df_x.insert(loc = 0,column="ones",value = np.ones(num_rows).astype('float16'))
    return df_x


def min_max_scale(column):
    #Column must be a serie numeric of pandas
    min_val = column.min()
    max_val = column.max()
    scaled_column = (column - min_val) / (max_val - min_val)
    return scaled_column

def scale_dataframe(df):
    scaled_df = pd.DataFrame()
    for column_name in df.columns:
        scaled_column = min_max_scale(df[column_name])
        scaled_df[column_name] = scaled_column
    return scaled_df

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
    debug = False # This debug boolean allows print eachs step of the gradient desendent implementation
    if debug:
        if num_col == currentParams.size:
            print("params size correct")
        else:
            print("params size INCORRECT")
        if df_y.shape[0] == num_rows:
            print("df_x and df_y size are CORRECT")
        else:
            print("df_x df_y size INCORRECT")

    delta = 0.00000001 # delta is the minimum expected value. If the train process reach a number lower before the last iteration the functino returns.
    const_alfa_m = alfa / num_rows  # This is the part of [lr/m] in -> new_teta = teta - [lr/m] * gradient
    errors = list()

    for p in range(periods):
        temp = currentParams
        evaluated_h = df_x.dot(currentParams)# As we add the 'ones' or bias column in df_x dataframe the matrix dot product is equivalent to evaluate the function for each register.
        if(debug):
            print("params applied:\n",evaluated_h) 
        activated_values = evaluated_h.apply(actvfun) # Apply activation function to make the regression logistic model
        if(debug):
            print("activation function applied:\n",activated_values)
        diff = activated_values - df_y
        if(debug):
            print("diff:\n",diff)
        for i in range(num_col): #For loop to update each teta in params and save them in temp variable
            x = diff.dot(df_x.iloc[:, i])
            temp[i] = currentParams[i] - const_alfa_m * x #Update each teta

        currentParams = temp #Once we update all the tetas we update the hypotesis that are allocated in current params
        curr_error = costfun(activated_values, df_y)
        errors.append(curr_error) #Save the error of each iteration

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
    
def test_model(params,df_x_test,df_y_test,stats):
    
    predictions=df_x_test.dot(params)
    predictions=predictions.apply(actvfun)
    error=costfun(predictions,df_y_test)
    print("Model tested")
    #print("Type of error: ",type(error))
    if(stats):
        [precision, recall, f1_score, accuracy]=stats_model(predictions, df_y_test)
        print("Precision: {:.4f}".format(precision))
        print("Recall: {:.4f}".format(recall))
        print("F1-Score: {:.4f}".format(f1_score))
        print("Accuracy: {:.4f}".format(accuracy))
        
    return [error,predictions]

def plot_model_result(params,df_x_train,df_x_test,df_y_train,df_y_test):
    [error_train,predictions_train]=test_model(params,df_x_train,df_y_train,False)
    [error_test,predictions_test]=test_model(params,df_x_test,df_y_test,False)
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



def assert_condition(pred, real):
    return (pred > 0.5 and real) or (pred <= 0.5 and not real)

def stats_model(df_prediction, df_real):
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for i in range(len(df_prediction)):
        if assert_condition(df_prediction.iloc[i], df_real.iloc[i]):
            true_positive += 1
        else:
            if df_prediction.iloc[i] > 0.5:
                false_positive += 1
            else:
                false_negative += 1

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1_score = 2 * (precision * recall) / (precision + recall)

    accuracy = (true_positive) / len(df_prediction)
    
    return [precision, recall, f1_score, accuracy]
