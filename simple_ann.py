# Simple ANN model that gives very bad accuracy
import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# baseline model
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(74, input_dim=58, kernel_initializer='normal', activation='relu'))
    model.add(Dense(8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# load dataset
dataframe = pandas.read_csv("data/data_class.csv")
#print(dataframe.head())

X=dataframe.drop(' shares', axis = 1) #dropping results
X=X.drop('Unnamed: 0',axis=1)
y = dataframe.iloc[:,-1]

print("Inputs to the model")
print(X.head())
print("Outputs to the model")
print(y.head())
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))

'''
seed = 7
estimator = KerasClassifier(build_fn=create_baseline, nb_epoch=100, batch_size=5, verbose=1)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))




'''
model = Sequential()
model.add(Dense(10, 64))
model.add(Activation('tanh'))
model.add(Dense(64, 1))
model.compile(loss='mean_absolute_error', optimizer='rmsprop')
'''


#model.fit(X_train, y_train, nb_epoch=20, batch_size=16)
#score = model.evaluate(X_test, y_test, batch_size=16)
