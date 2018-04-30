#Preprocessing script which removes all the the annoying rows and converts the output from regression to a classification problem

import pandas as pd

df = pd.read_csv("data/data.csv")
#df.drop(['url', 'timedelta'], axis=1)
df = df.drop(['url',' timedelta'],axis = 1)
df[' shares'] = (df[' shares'] >=1400).astype(int)
df.to_csv('data/data_class.csv')