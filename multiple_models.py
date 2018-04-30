# Tries multiple model manually. Not a good implementation
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


# load dataset to padas dataframe
csv_filename="data/data.csv"

df=pd.read_csv(csv_filename)

popular = df[" shares"] >= 1400
unpopular = df[" shares"] < 1400

df.loc[popular,' shares'] = 1
df.loc[unpopular,' shares'] = 0


# split original dataset into 60% training and 40% testing
features=list(df.columns[2:60])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(df[features], df[' shares'], test_size=0.4, random_state=0)


# increasingly add size of training set 5% of orginal, keep testing size unchanged
for i in range(0,100,5):
	X_rest, X_trian_part, y_rest, y_train_part= cross_validation.train_test_split(X_train, y_train, test_size=0.049+i/100.0, random_state=0)
	print ("====================== loop: "+str(i))
	t0=time()
	print ("DecisionTree")
	dt = DecisionTreeClassifier(min_samples_split=20,random_state=99)
# dt = DecisionTreeClassifier(min_samples_split=20,max_depth=5,random_state=99)
	clf_dt=dt.fit(X_trian_part,y_train_part)
	score_dt=clf_dt.score(X_test,y_test)
	print ("Acurracy: "+str(score_dt))
	t1=time()
	dur_dt=t1-t0
	print ("time elapsed: "+str(dur_dt))
	print ("\n")

	t6=time()
	print ("KNN")
# knn = KNeighborsClassifier(n_neighbors=3)
	knn = KNeighborsClassifier()
	clf_knn=knn.fit(X_trian_part, y_train_part)
	score_knn=clf_knn.score(X_test,y_test)
	print ("Acurracy: "+str(score_knn))
	t7=time()
	dur_knn=t7-t6
	print ("time elapsed: "+str(dur_knn))
	print ("\n")

	t2=time()
	print ("RandomForest")
	rf = RandomForestClassifier(n_estimators=100,n_jobs=-1)
	clf_rf = rf.fit(X_trian_part,y_train_part)
	score_rf=clf_rf.score(X_test,y_test)
	print ("Acurracy: "+str(score_rf))
	t3=time()
	dur_rf=t3-t2
	print ("time elapsed: "+str(dur_rf))
	print ("\n")

	t4=time()
	print ("NaiveBayes")
	nb = BernoulliNB()
	clf_nb=nb.fit(X_trian_part,y_train_part)
	score_nb=clf_nb.score(X_test,y_test)
	print ("Acurracy: "+str(score_nb))
	t5=time()
	dur_nb=t5-t4
	print ("time elapsed: "+str(dur_nb))
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

	list1.append(i/100.0+0.05)
	list1.append(score_dt)
	list1.append(score_knn)
	list1.append(score_rf)
	list1.append(score_nb)
	list1.append(score_svm)

	list2.append(i/100.0+0.05)
	list2.append(dur_dt)
	list2.append(dur_knn)
	list2.append(dur_rf)
	list2.append(dur_nb)
	list2.append(dur_svm)

print(list1)
print(list2)
