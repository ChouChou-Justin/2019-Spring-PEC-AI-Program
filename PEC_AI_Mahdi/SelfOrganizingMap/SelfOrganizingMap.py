# -*- coding: utf-8 -*-
"""
Created on Fri May 24 19:57:26 2019
@author: user
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv("AI-Practice05_CompetitiveLearningNetworks_Data.txt", sep = "\t", header = None)
dataf = data.iloc[1:, [1,2,3,4]].values
data = [[float(each) for each in item]for item in dataf]
dana = list(map(list, zip(*data)))
#print(data,"\n\n")
#print(dana)
# Number of clusters
kmeans = KMeans(n_clusters=5, init = 'random', n_init=1)
# Fitting the input data
kmeans = kmeans.fit(data)
# Getting the cluster labels
labels = kmeans.predict(data)
# Centroid values
centroids = kmeans.cluster_centers_
print('centroids are:\n', centroids)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(dana[1], dana[2], dana[3], c=labels)
plt.show()
print(labels)
#print(data)
#print(dana[1])
#som = MiniSom(6, 6, 4, sigma = 0.5, learning_rate = 0.5)
#som.train_random(data, 100)
#result1 = som.activation_response(data)
#print("Active Percerptrons are:\n", result1)

#img=plt.imread('1.jpg')
#print(img.shape)
#print(img[0,0])
#plt.imshow(img)
#pixels = np.reshape(img, (img.shape[0]*img.shape[1], 3))/255
#print(pixels.shape)