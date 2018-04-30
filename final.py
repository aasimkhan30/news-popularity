# Multiple Models Final Script. A very bad way to do it. Using models in a list is better
import pandas as pd
import numpy as np
from time import time
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

csv_filename="data/data.csv"

df=pd.read_csv(csv_filename)

popular = df[" shares"] >= 3395
unpopular = df[" shares"] < 3395

df.loc[popular,' shares'] = 1
df.loc[unpopular,' shares'] = 0

X=df[df.columns[2:60]]
y = df.iloc[:,-1]

print(df.columns)
print(X.columns)
print(y)


for i in [0.5,0.6,0.70,0.75,0.80]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0,stratify=y)
    print("Ratios")
    t0=time()
    model = Sequential()
    model.add(Dense(10, input_dim=58, kernel_initializer='normal', activation='relu'))
    #model.add(Dense(3, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Sequential Model")
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    t1 = time()
    dur_dt = t1-t0
    print("time elapsed: "+str(dur_dt))
    print("\n")

    t0=time()
    print ("DecisionTree")
    dt = DecisionTreeClassifier(min_samples_split = 30,random_state = 99)
    # dt = DecisionTreeClassifier(min_samples_split=20,max_depth=5,random_state=99)
    clf_dt = dt.fit(X_train, y_train)
    score_dt = clf_dt.score(X_test, y_test)
    print("Acurracy: "+str(score_dt))
    t1 = time()
    dur_dt = t1-t0
    print("time elapsed: "+str(dur_dt))
    print("\n")

    t6 = time()
    print("KNN")
    # knn = KNeighborsClassifier(n_neighbors=3)
    knn = KNeighborsClassifier()
    clf_knn = knn.fit(X_train, y_train)
    score_knn = clf_knn.score(X_test, y_test)
    print("Acurracy: "+str(score_knn))
    t7 = time()
    dur_knn = t7-t6
    print("time elapsed: "+str(dur_knn))
    print("\n")

    t2=time()
    print ("RandomForest")
    rf = RandomForestClassifier(n_estimators=100,n_jobs=-1)
    clf_rf = rf.fit(X_train,y_train)
    score_rf=clf_rf.score(X_test,y_test)
    print ("Acurracy: "+str(score_rf))
    t3=time()
    dur_rf=t3-t2
    print ("time elapsed: "+str(dur_rf))
    print ("\n")

    t4=time()
    print ("NaiveBayes")
    nb = BernoulliNB()
    clf_nb=nb.fit(X_train,y_train)
    score_nb=clf_nb.score(X_test,y_test)
    print ("Acurracy: "+str(score_nb))
    t5=time()
    dur_nb=t5-t4
    print ("time elapsed: "+str(dur_nb))
    print("\n")

    t5 = time()
    print("SVM")
    mysvm = SVC()
    clf_svm = mysvm.fit(X_train, y_train)
    score_svm = clf_svm.score(X_test, y_test)
    print("Acurracy: " + str(score_svm))
    t6 = time()
    dur_svm = t6-t5
    print("time elapsed: " + str(dur_svm))