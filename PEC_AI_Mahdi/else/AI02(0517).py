# -*- coding: utf-8 -*-
"""
Created on Wed May 15 20:07:19 2019

@author: ShinTing
"""
import seaborn as sns
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import math
import operator
Not_Stressed = pd.read_csv('C:\\Users\\ShinTing\\.spyder-py3\\PEC AI\\Not_Stressed.csv', header=None)
Stressed = pd.read_csv('C:\\Users\\ShinTing\\.spyder-py3\\PEC AI\\Stressed.csv', header=None)
N_W = [float(i) for i in Not_Stressed[1:][1]]
N_H = [float(i) for i in Not_Stressed[1:][2]]
S_W = [float(i) for i in Stressed[1:][1]]
S_H = [float(i) for i in Stressed[1:][2]]
plt.scatter(S_H,S_W,c='b',label='Stressed')
plt.scatter(N_H,N_W,color='r',label='Not Stressed')
plt.legend(loc=(1, 0))
plt.xlabel("Husband")
plt.ylabel("Wife")
plt.title("All incomes annual in $100,000 units")
plt.show()
NData = [[N_W[i],N_H[i]] for i in range(len(N_W))]
SData = [[S_W[i],S_H[i]] for i in range(len(S_W))]

SData_Shuffle=shuffle(SData)
NData_Shuffle=shuffle(NData)

SData_train=SData_Shuffle[0:100]
SData_test=SData_Shuffle[100:]
NData_train=NData_Shuffle[0:100]
NData_test=NData_Shuffle[100:]


plt.scatter([item[1] for item in SData_train],[item[0] for item in SData_train],c='b',label='Stressed')
plt.scatter([item[1] for item in NData_train],[item[0] for item in NData_train],c='r',label='Not Stressed')

plt.legend(loc=(1, 0))
plt.xlabel("Husband")
plt.ylabel("Wife")
plt.title("Training Data")
plt.show()


plt.scatter([item[1] for item in SData_test],[item[0] for item in SData_test],c='b',label='Stressed')
plt.scatter([item[1] for item in NData_test],[item[0] for item in NData_test],c="red",label='Not Stressed')

plt.legend(loc=(1, 0))
plt.xlabel("Husband")
plt.ylabel("Wife")
plt.title("Test Data")

plt.show()

#Mean data
Nx=0
Ny=0
Sx=0
Sy=0
for i in NData_train:
    Nx = i[0] + Nx
    Ny = i[1] + Ny
N_Mean = [Nx/len(NData_train),Ny/len(NData_train)]
for i in SData_train:
    Sx = i[0] + Sx
    Sy = i[1] + Sy
S_Mean = [Sx/len(SData_train),Sy/len(SData_train)]


N_H_Mean=N_Mean[1]
N_W_Mean=N_Mean[0]
S_H_Mean=S_Mean[1]
S_W_Mean=S_Mean[0]


def Classifier(W,H):
    F1 = N_W_Mean*W+N_H_Mean*H-((np.square(N_W_Mean)+np.square(N_H_Mean))/2)
    F2 = S_W_Mean*W+S_H_Mean*H-((np.square(S_W_Mean)+np.square(S_H_Mean))/2)
    F = F2-F1
    return F
TrueS =[]#append(x(H),y(W))
FalseS=[]
TrueNS=[]
for i in range(0,100):
    #Ture for Stressed
    if (Classifier(SData_test[i][0],SData_test[i][1]) >= 0):
        TrueS.append([SData_test[i][0],SData_test[i][1]]) 
        #plt.scatter(S_H_test[i],S_W_test[i],c='b',label='Stressed')
        
    #False for Stressed
    if (Classifier(SData_test[i][0],SData_test[i][1]) < 0):
        FalseS.append([SData_test[i][0],SData_test[i][1]])
        #plt.scatter(S_H_test[i],S_W_test[i],c='g',label='Classified wrong')
        #標記classified wrong 太醜了
        #plt.annotate('Classified wrong', xy=(S_H_test[i],S_W_test[i]), xytext=(10, 15),arrowprops=dict(facecolor='black', shrink=0.00001))
    if (i < 99):
        #Ture for Not_Stressed
        if (Classifier(NData_test[i][0],NData_test[i][1]) <= 0):
            TrueNS.append([NData_test[i][0],NData_test[i][1]])
            #plt.scatter(N_H_test[i],N_W_test[i],c='r',label='Not Stressed')
           # print(abs(N_S_Classifier(N_W_test[i],N_H_test[i])))
           # print(N_W_test[i],N_H_test[i],N_S_Classifier(N_W_test[i],N_H_test[i]))
           # print(abs(S_Classifier(N_W_test[i],N_H_test[i])))
            #print(N_W_test[i],N_H_test[i],S_Classifier(N_W_test[i],N_H_test[i]))
        #False for Not_Stressed
        if (Classifier(NData_test[i][0],NData_test[i][1]) > 0):
            FalseS.append([NData_test[i][0],NData_test[i][1]])
            #plt.scatter(N_H_test[i],N_W_test[i],c='g',label='Classified wrong')
           # print(abs(N_S_Classifier(N_W_test[i],N_H_test[i])))
           # print(N_W_test[i],N_H_test[i],N_S_Classifier(N_W_test[i],N_H_test[i]))
           # print(abs(S_Classifier(N_W_test[i],N_H_test[i])))
           # print(N_W_test[i],N_H_test[i],S_Classifier(N_W_test[i],N_H_test[i]))  
#print(len(TrueS)+len(FalseS)+len(TrueNS))
plt.scatter([item[1] for item in TrueS] ,[item[0] for item in TrueS],c='b',label='Stressed')
plt.scatter([item[1] for item in TrueNS] ,[item[0] for item in TrueNS],c='r',label='Not Stressed')
plt.scatter([item[1] for item in FalseS] ,[item[0] for item in FalseS],c='g',label='Classified wrong')
plt.xlabel("Husband")
plt.ylabel("Wife")
plt.title("Test Data")
plt.legend(loc=(1, 0))           
plt.show()           

# KNN
def Distance(instance1,instance2,length):
    distance = 0 
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2) #sqrt((x1-x2)^2+(1-2)^2)
    return math.sqrt(distance)
def getNeighbors(trainingSet,testInstance,k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist  = Distance(testInstance,trainingSet[x],length)
        distances.append((trainingSet[x],dist))
    distances.sort(key = operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors
def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

  
testSet=[]      
trainingSet=[]
predictions = []
for i in range(0,100):
        trainingSet.append([SData_train[i][0],SData_train[i][1],'S'])
        trainingSet.append([NData_train[i][0],NData_train[i][1],'N'])
        testSet.append([SData_test[i][0],SData_test[i][1],'S'])
        if i < 99:
            testSet.append([NData_test[i][0],NData_test[i][1],'N'])
def KNN(k):                
    miscount = 0
    for x in range(len(testSet)):
        neighbors=getNeighbors(trainingSet,testSet[x],k)
        result = getResponse(neighbors)
        predictions.append(result)
        if (repr(result)!=repr(testSet[x][-1])):
            miscount = miscount +1 
        #print('> predicted=',repr(result),',actual=',repr(testSet[x][-1]))
    return miscount

K=[]
for k in range(3,22,2):K.append([k,KNN(k)])
#print([item[0] for item in K],[item[1] for item in K] )
plt.plot([item[0] for item in K],[item[1] for item in K])
new_ticks = np.linspace(3, 21, 10)
plt.xticks(new_ticks)
plt.xlabel("k")
plt.ylabel("# of misclassified point")
plt.title("KNN")
plt.show()

#print(k)
#accuracy = getAccuracy(testSet,predictions)       
#print('Accuracy: ',repr(accuracy),'%')     

#def getAccuracy(testSet, predictions):
#    correct= 0 
#    for x in range(len(testSet)):
#        if testSet[x][-1] == predictions[x]:
#            correct +=1
#    return (correct/float(len(testSet)))*100.0




