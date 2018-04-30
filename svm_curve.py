# SVM model with increasing train size.
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

csv_filename="data/data.csv"

df=pd.read_csv(csv_filename)

popular = df[" shares"] >= 1400
unpopular = df[" shares"] < 1400

df.loc[popular,' shares'] = 1
df.loc[unpopular,' shares'] = 0

features=list(df.columns.difference(['url', ' timedelta', ' is_weekend',' shares']))
X_train, X_test, y_train, y_test = cross_validation.train_test_split(df[features], df[' shares'], test_size=0.4, random_state=0)

for i in range(0,100,5):
	X_rest, X_trian_part, y_rest, y_train_part= cross_validation.train_test_split(X_train, y_train, test_size=0.049+i/100.0, random_state=0)
	print ("====================== loop: "+str(i))
	print("\n")
	t5 = time()
	print("SVM")
	mysvm = svm.SVC()
	clf_svm = mysvm.fit(X_trian_part, y_train_part)
	score_svm = clf_svm.score(X_test, y_test)
	print("Acurracy: " + str(score_svm))
	t6 = time()
	dur_svm = t6-t5
	print("time elapsed: " + str(dur_svm))

# write result data to excel file
	list1=[]
	list2=[]

	list1.append(score_svm)

	list2.append(dur_svm)

print(list1)
print(list2)


