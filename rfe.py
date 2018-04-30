# Tries RFE selecting best combination of features.
import os
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from time import time
from sklearn import svm
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA as sklearnPCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
# load dataset to pandas dataframe
csv_filename="data/data.csv"

df=pd.read_csv(csv_filename)
popular = df[" shares"] >= 3395
unpopular = df[" shares"] < 3395

df.loc[popular,' shares'] = 1
df.loc[unpopular,' shares'] = 0


# split original dataset into 60% training and 40% testing
features=list(df.columns[2:60])


X=df[features]
y=df[' shares']



X_train, X_test, y_train, y_test = cross_validation.train_test_split(df[features], df[' shares'], test_size=0.2, random_state=0)

start_time=time()
print ("Logistic Regression")
model =   LogisticRegression()
# create the RFE model and select 3 attributes
for i in range(1,58):
	print("Selecting "+str(i)+" features:")
	rfe = RFE(model, i)
	rfe = rfe.fit(X_train,y_train)
	score_rf=rfe.score(X_test,y_test)
	print ("Acurracy: "+str(score_rf))
	print(rfe.support_)
	print(rfe.ranking_)
# summarize the selection of the attributes
end_time=time()
dur_rf=start_time-end_time
print ("time elapsed: "+str(dur_rf))
print ("\n")

