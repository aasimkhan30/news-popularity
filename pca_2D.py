# generate a 2d PCA plot
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

# load dataset to padas dataframe
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


X_norm = (X - X.min())/(X.max() - X.min())


pca = sklearnPCA(n_components=2)  # 2-dimensional PCA
transformed = pd.DataFrame(pca.fit_transform(X_norm))


plt.scatter(transformed[y==1][0], transformed[y==1][1], label='Popular', c='red')
plt.scatter(transformed[y==0][0], transformed[y==0][1], label='Unpopular', c='blue')

plt.legend()
plt.show()