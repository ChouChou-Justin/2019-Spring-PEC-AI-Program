# -*- coding: utf-8 -*-
"""
Created on Wed May 22 20:04:09 2019

@author: Hsin-Ting Chou, Shin-Ting Wu, Jarwy Hsue
"""
import random
import math
import matplotlib.pyplot as plt
def normalize(vector):
    norm = 0
    for each in vector:
        norm +=pow(each,2)
    norm = math.sqrt(norm)
    vector = [vector[0]/norm,vector[1]/norm,vector[2]/norm,vector[3]/norm]
    return vector
file = open("AI-Practice05_CompetitiveLearningNetworks_Data.txt", "r", encoding = "utf-8")
data = [(each.replace("\n", "").split("\t")) for each in file]
Data = [[float(data[i+1][1]),float(data[i+1][2]),float(data[i+1][3]),float(data[i+1][4])] for i in range(len(data)-1)]
Data=[normalize(Data[i]) for i in range(len(Data))]
W = [[random.random(),random.random(),random.random(),random.random()] for i in range(10)]
W = [normalize(W[i]) for i in range(len(W))]
Score = []
Compare = []
for i in range(len(Data)):
    for j in range(10):
        Compare.append(Data[i][0]*W[j][0]+Data[i][1]*W[j][1]+Data[i][2]*W[j][2]+Data[i][3]*W[j][3])
    Maxindex = Compare.index(max(Compare))
    Score.append([Maxindex,max(Compare)])
    for item in range(4):
        W[Maxindex][item] = W[Maxindex][item]+0.3*(Data[i][item]-W[Maxindex][item])
    Compare = []  
Color = ['R','B','Orange','brown','pink','sienna','purple','khaki','G','navy']    
Vote = [0,0,0,0,0,0,0,0,0,0]    
for i in range(len(Data)):
    for j in range(10):
        Compare.append(Data[i][0]*W[j][0]+Data[i][1]*W[j][1]+Data[i][2]*W[j][2]+Data[i][3]*W[j][3])
    Maxindex = Compare.index(max(Compare))
    Score.append([Maxindex,max(Compare)])
    Compare = []  
    plt.scatter(Data[i][0],Data[i][1],c=Color[Maxindex])
    Vote[Maxindex]=Vote[Maxindex]+1    
    print(Vote)
plt.show()