#Regresion Model A01703366
#Fermín Méndez García


import pandas as pd
import numpy as np



data_file_path = 'nutrition.csv'
df = pd.read_csv(data_file_path, delimiter=' ')

print(df.head())