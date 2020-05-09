# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 19:04:36 2019
@author: ChouHsingTing
"""
from minisom import MiniSom
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
import math
startTime = time.time()

def getData(file):
    suicideRate = [float(i) for i in file['suicides/100k pop']] 
#    sexEncode = [i for i in file['sex_encode_']]
    year = [i for i in file['year']]
    ageGroupEncode = [i for i in file['age_encode_']]
    continentEncode = [i for i in file['continent_encode_']] 
#    population = [i for i in file['population']] 
    Data = [[suicideRate[i],year[i],ageGroupEncode[i],continentEncode[i]] for i in range(len(suicideRate))]    
    return Data

def normalize(vector):
    norm = 0
    for each in vector:
        norm += pow(each,2)
    norm = math.sqrt(norm)
    if norm != 0:
        vector = [vector[0]/norm,vector[1]/norm,vector[2]/norm,vector[3]/norm]
    else:
        vector = [0,0,0,0]
    return vector

def competitveLearningNetwork(data):
    W = [[random.random(),random.random(),random.random(),random.random()] for i in range(10)]
    W = [normalize(W[i]) for i in range(len(W))]
    Score = []
    Compare = []
    for i in range(len(data)):
        for j in range(10):
            Compare.append(data[i][0]*W[j][0]+data[i][1]*W[j][1]+data[i][2]*W[j][2]+data[i][3]*W[j][3])
        Maxindex = Compare.index(max(Compare))
        Score.append([Maxindex,max(Compare)])
        for item in range(4):
            W[Maxindex][item] = W[Maxindex][item]+0.3*(data[i][item]-W[Maxindex][item])
        Compare = []  
    Color = ['R','B','Orange','brown','pink','sienna','purple','khaki','G','navy']    
    Vote = [0,0,0,0,0,0,0,0,0,0]    
    for i in range(len(data)):
        for j in range(10):
            Compare.append(data[i][0]*W[j][0]+data[i][1]*W[j][1]+data[i][2]*W[j][2]+data[i][3]*W[j][3])
        Maxindex = Compare.index(max(Compare))
        Score.append([Maxindex,max(Compare)])
        Compare = []  
        plt.scatter(data[i][0],data[i][1],c=Color[Maxindex])
        Vote[Maxindex]=Vote[Maxindex]+1 
        if i % 2000==0:
            print(Vote)
    plt.show()
    
def selfOrganizingMap(data):
    som = MiniSom(6, 6, 4, sigma = 0.5, learning_rate = 0.5)
    som.train_random(data, 100)
    result1 = som.activation_response(data)
    print("Active Percerptrons are:\n", result1)
    
def kMeans(data):
    # Number of clusters
    kmeans = KMeans(n_clusters=6, init = 'random', n_init=1)
    # Fitting the input data
    kmeans = kmeans.fit(data)
    # Getting the cluster labels
    labels = kmeans.predict(data)
    # Centroid values
    centroids = kmeans.cluster_centers_
    print('centroids are:\n', centroids)
    fig = plt.figure()
    dana = list(map(list, zip(*data)))
    ax = Axes3D(fig)
    ax.set_xlabel('suicideRate')
    ax.set_ylabel('year')
    ax.set_zlabel('ageGroupEncode')
    ax.scatter(dana[0], dana[1], dana[2], c=labels)
    plt.show()
#    ax.set_xlabel('sexEncode')
#    ax.set_ylabel('ageGroupEncode')
#    ax.set_zlabel('continentEncode')
#    ax.scatter(dana[1], dana[2], dana[3], c=labels)
#    plt.show()
    
def main():
    file = pd.read_csv('suicide_rates_overview_1985_to_2016_EncodeDataset.csv')
    Data = getData(file)
    Data = [normalize(Data[i]) for i in range(len(Data))]
    
#    competitveLearningNetwork(Data) 
#    selfOrganizingMap(Data)
    kMeans(Data)
main()
runningTime = time.time() - startTime
print ("time: {}".format(runningTime))