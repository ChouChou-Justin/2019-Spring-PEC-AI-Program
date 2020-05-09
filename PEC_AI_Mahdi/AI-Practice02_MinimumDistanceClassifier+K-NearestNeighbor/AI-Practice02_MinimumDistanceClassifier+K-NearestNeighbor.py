# -*- coding: utf-8 -*-
"""
Created on Thu May 16 19:50:00 2019

@author: Hsin-Ting Chou
"""
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
startTime = time.time()
#Q1----------------------------------------------------------------------------
data = pd.read_csv('AI-Practice02_MinimumDistanceClassifier+K-NearestNeighbor_Data.csv', header = None)
Not_stressed_wife_data = [float(i) for i in data[2:201][0]]
Not_stressed_husband_data = [float(i) for i in data[2:201][1]]
Stressed_wife_data = [float(i) for i in data[203:][0]]
Stressed_husband_data = [float(i) for i in data[203:][1]]
    
#    print(data)
#    print(Not_stressed_wife_data)
#    print(Not_stressed_husband_data)
#    print(Stressed_wife_data)
#    print(Stressed_husband_data)
    
plt.scatter(Not_stressed_husband_data, 
            Not_stressed_wife_data,
            c = 'green',
            marker = 'x',
            label = 'Not Stressed')
plt.scatter(Stressed_husband_data, 
            Stressed_wife_data,
            c = 'orange',
            marker = 'o',
            label = 'Stressed')
plt.legend(loc = 'upper right')
plt.xlabel("Husband")
plt.ylabel("Wife")
plt.title("All incomes annual in $100,000 units")
plt.show()
    
#Q2----------------------------------------------------------------------------
##Getting data method1
#df_Not_stressed_data = data[2:201]
#df_Stressed_data = data[203:]
#Not_stressed_data_array = np.array(df_Not_stressed_data).reshape(len(df_Not_stressed_data), 2)
#Stressed_data_array = np.array(df_Stressed_data).reshape(len(df_Stressed_data), 2)
#Not_stressed_data_pre = [i for i in Not_stressed_data_array]
#Stressed_data_pre = [i for i in Stressed_data_array]
#    
#i = 0 
#Not_stressed_data = []
#while i < 199:
#    x = [ i for i in Not_stressed_data_pre[i]]
#    Not_stressed_data.append(x)
#    i=i+1
#        
#i = 0
#Stressed_data = []
#while i < 200:
#    y = [ i for i in Stressed_data_pre[i]]
#    Stressed_data.append(y)
#    i=i+1

#Getting data method2
Not_stressed_data = [[Not_stressed_wife_data[i], Not_stressed_husband_data[i]] for i in range(len(Not_stressed_wife_data))]
Stressed_data = [[Stressed_wife_data[i], Stressed_husband_data[i]] for i in range(len(Stressed_wife_data))]

#Random.shuffle
random.shuffle(Not_stressed_data) 
random.shuffle(Stressed_data)  
Not_stressed_dataShuffle = Not_stressed_data
Stressed_dataShuffle = Stressed_data
    
#Spliting train data
Not_stressed_data_train = Not_stressed_dataShuffle[0:100]
Stressed_data_train = Stressed_dataShuffle[0:100]
   
#Spliting test data
Not_stressed_data_test = Not_stressed_dataShuffle[100:]
Stressed_data_test = Stressed_dataShuffle[100:]
    
#Spliting wife and husband train data
i = 0
Not_stressed_wife_data_train = []
Not_stressed_husband_data_train = []
Stressed_wife_data_train = []
Stressed_husband_data_train = []
while i < 100:
    xn = Not_stressed_data_train[i][0]
    yn = Not_stressed_data_train[i][1]
    xs = Stressed_data_train[i][0]
    ys = Stressed_data_train[i][1]
    Not_stressed_wife_data_train.append(float(xn))
    Not_stressed_husband_data_train.append(float(yn))
    Stressed_wife_data_train.append(float(xs))
    Stressed_husband_data_train.append(float(ys))
    i += 1
        
#Spliting wife and husband test data
i = 0
Not_stressed_wife_data_test = []
Not_stressed_husband_data_test = []
while i < 99:
    x = Not_stressed_data_test[i][0]
    y = Not_stressed_data_test[i][1]
    Not_stressed_wife_data_test.append(float(x))
    Not_stressed_husband_data_test.append(float(y))
    i += 1
    
i = 0
Stressed_wife_data_test = []
Stressed_husband_data_test = []
while i < 100:
    x = Stressed_data_test[i][0]
    y = Stressed_data_test[i][1]
    Stressed_wife_data_test.append(float(x))
    Stressed_husband_data_test.append(float(y))
    i += 1

plt.scatter(Not_stressed_husband_data_train, 
            Not_stressed_wife_data_train,
            c = 'green',
            marker = 'x',
            label = 'Not Stressed')
plt.scatter(Stressed_husband_data_train, 
            Stressed_wife_data_train,
            c = 'orange',
            marker = 'o',
            label = 'Stressed')
plt.legend(loc = 'upper right')
plt.xlabel("Husband")
plt.ylabel("Wife")
plt.title("Training Data")
plt.show()
    
plt.scatter(Not_stressed_husband_data_test, 
            Not_stressed_wife_data_test,
            c = 'green',
            marker = 'x',
            label = 'Not Stressed')
plt.scatter(Stressed_husband_data_test, 
            Stressed_wife_data_test,
            c = 'orange',
            marker = 'o',
            label = 'Stressed')
plt.legend(loc = 'upper right')
plt.xlabel("Husband")
plt.ylabel("Wife")
plt.title("Test Data")
plt.show()
    
#Q3: Minimum Distance Classifier-----------------------------------------------
#Calculate mean value
Not_stressed_wife_mean = 0
Not_stressed_husband_mean = 0
Stressed_wife_mean = 0
Stressed_husband_mean = 0
    
for i in Not_stressed_wife_data_train:
    Not_stressed_wife_mean = i + Not_stressed_wife_mean
Not_stressed_wife_mean = Not_stressed_wife_mean / len(Not_stressed_wife_data_train)
    
for i in Not_stressed_husband_data_train:
    Not_stressed_husband_mean = i + Not_stressed_husband_mean
Not_stressed_husband_mean = Not_stressed_husband_mean / len(Not_stressed_husband_data_train)
    
for i in Stressed_wife_data_train:
    Stressed_wife_mean = i + Stressed_wife_mean
Stressed_wife_mean = Stressed_wife_mean / len(Stressed_wife_data_train)
    
for i in Stressed_husband_data_train:
    Stressed_husband_mean = i + Stressed_husband_mean
Stressed_husband_mean = Stressed_husband_mean / len(Stressed_husband_data_train)
    
#decisionFunction
def decisionFunction(wife_data_test, husband_data_test):
    F1 = Not_stressed_wife_mean*wife_data_test + Not_stressed_husband_mean*husband_data_test - ((Not_stressed_wife_mean**2 + Not_stressed_husband_mean**2)/2)
    F2 = Stressed_wife_mean*wife_data_test + Stressed_husband_mean*husband_data_test - ((Stressed_wife_mean**2 + Stressed_husband_mean**2)/2)
    F = F2 - F1
    return F

#Classification
Not_stressedTrue = []
Not_stressedFalse = []
StressedTrue = []
StressedFalse = []

for i in range(99):
    if(decisionFunction(Not_stressed_wife_data_test[i], Not_stressed_husband_data_test[i]) <= 0):
        Not_stressedTrue.append([Not_stressed_wife_data_test[i], Not_stressed_husband_data_test[i]])
    elif (decisionFunction(Not_stressed_wife_data_test[i], Not_stressed_husband_data_test[i]) > 0):
        Not_stressedFalse.append([Not_stressed_wife_data_test[i], Not_stressed_husband_data_test[i]])
        
for i in range(100):
    if (decisionFunction(Stressed_wife_data_test[i], Stressed_husband_data_test[i]) >= 0):
        StressedTrue.append([Stressed_wife_data_test[i], Stressed_husband_data_test[i]])
    elif (decisionFunction(Stressed_wife_data_test[i], Stressed_husband_data_test[i]) < 0):
        StressedFalse.append([Stressed_wife_data_test[i], Stressed_husband_data_test[i]])
        
plt.scatter([Not_stressedTrue[i][1] for i in range(len(Not_stressedTrue))], 
            [Not_stressedTrue[i][0] for i in range(len(Not_stressedTrue))],
            c = 'green',
            marker = 'x',
            label = 'Not Stressed True')
plt.scatter([StressedTrue[i][1] for i in range(len(StressedTrue))], 
            [StressedTrue[i][0] for i in range(len(StressedTrue))],
            c = 'orange',
            marker = 'o',
            label = 'Stressed True')   
plt.scatter([Not_stressedFalse[i][1] for i in range(len(Not_stressedFalse))], 
            [Not_stressedFalse[i][0] for i in range(len(Not_stressedFalse))],
            c = 'red',
            marker = 'x',
            label = 'Not Stressed False')
plt.scatter([StressedFalse[i][1] for i in range(len(StressedFalse))], 
            [StressedFalse[i][0] for i in range(len(StressedFalse))],
            c = 'red',
            marker = 'o',
            label = 'Stressed False')

plt.legend(loc = 'upper right')
plt.xlabel("Husband")
plt.ylabel("Wife")
plt.title("Test Data By Minimum Distance Classifier")
plt.show()
    
#Q4: KNN-----------------------------------------------------------------------
#add label to dataset
training_dataset = []
test_dataset = []
for i in range(100):
    training_dataset.append([Not_stressed_wife_data_train[i], Not_stressed_husband_data_train[i], 'Not stressed'])
    training_dataset.append([Stressed_wife_data_train[i], Stressed_husband_data_train[i], 'Stressed'])
    test_dataset.append([Stressed_wife_data_test[i], Stressed_husband_data_test[i], 'Stressed'])
for i in range(99):
    test_dataset.append([Not_stressed_wife_data_test[i], Not_stressed_husband_data_test[i], 'Not stressed'])

#get distance
def getDistance(coordinate1, coordinate2):
    distance = 0
    distance = ((coordinate1[1] - coordinate2[1])**2 + (coordinate1[0] - coordinate2[0])**2)**0.5
    return distance

#get neighbors
def getNeighbors(training_dataset, test_data, k):
    distances = []
    for i in range(len(training_dataset)):
        distance = getDistance(test_data, training_dataset[i])
        distances.append([training_dataset[i], distance])
    distances.sort(key = getSortingkey)
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])   
    return neighbors

#get sorting key
def getSortingkey(distances):
    return distances[1]

#Classifier
def Classifier(neighbors):
    vote = {}
    for i in range(len(neighbors)):
        classified_result = neighbors[i][-1]
        if classified_result in vote:
            vote[classified_result] += 1
        else:
            vote[classified_result] = 1
    result = sorted(vote.items(), key = getSortingkey, reverse=True)
    return result[0][0]

#KNN
def KNN(k):
    predictions = []
    misclassified = 0
    for i in range(len(test_dataset)):
        neighbors = getNeighbors(training_dataset,test_dataset[i],k)
        result = Classifier(neighbors)
        predictions.append(result)
        if(result != test_dataset[i][-1]):
            misclassified += 1 
    return misclassified

K=[]
for k in range(3,22,2):
    K.append([k,KNN(k)])

plt.plot([K[i][0] for i in range(len(K))],[K[i][1] for i in range(len(K))])
ticks = [i for i in range(3, 22, 2)]
plt.xticks(ticks)
plt.xlabel("k")
plt.ylabel("# of misclassified points")
plt.title("KNN")
plt.show()
runningTime = time.time() - startTime
print ("time: {}".format(runningTime))
