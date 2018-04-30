# Generates a 3D plot for PCA and rotates it automatically.
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets.samples_generator import make_blobs

from pandas.tools.plotting import parallel_coordinates



dataset = pd.read_csv("data\data_class.csv")



X = dataset.drop(' shares', axis=1)

# Removing serial number
#X = X.drop('Unnamed: 0', axis=1)

y = dataset[' shares']


fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.cla()

X_norm = (X - X.min())/(X.max() - X.min())


pca = sklearnPCA(n_components=3)  # 2-dimensional PCA
transformed = pd.DataFrame(pca.fit_transform(X_norm))

print(transformed)



ax.scatter(transformed[y==1][0], transformed[y==1][1], transformed[y==1][2], label='Popular', c='r',edgecolor='k')
ax.scatter(transformed[y==0][0], transformed[y==0][1], transformed[y==0][2],label='Unpopular', c='b', edgecolor='k')

for angle in range(0,360):
    ax.view_init(30,angle)
    plt.draw()
    plt.pause(.001)
plt.legend()
plt.show()
