
# Logistic regresion model by Fermín Méndez A01703366
import pandas as pd
import numpy as np

# main.py

# Import other libraries


# 1. Import the dataset
# Read the CSV file into a DataFrame

# Read the original files
df_mat = pd.read_csv('student-mat.csv')
df_por = pd.read_csv('student-por.csv')
# 2. Clean the data and define train and test data: df_x_train, df_y_train, df_x_test and df_y_tet.
print(df_mat.info())
print(df_por.info)

#Common data #school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"

# Perform a left join
#df = pd.merge(df_mat, df_por, on=["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet"], how="left")
#print(df.info())