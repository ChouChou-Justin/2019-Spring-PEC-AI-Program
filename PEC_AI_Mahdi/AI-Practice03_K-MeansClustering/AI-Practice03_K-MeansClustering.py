# -*- coding: utf-8 -*-
"""
Created on Fri May 17 19:29:48 2019

@author: Hsin-Ting Chou
"""
import random
import matplotlib.pyplot as plt
import time

def stepFunction(y):
    return 0 if y <= 0 else 1

def isStopCriterionMet(testingList):
    misClassifiedCount = 0
    for testingData in testingList:
        yHat = stepFunction(wi * testingData[0] + wj * testingData[1] + w0)
        if yHat != testingData[2]:
            misClassifiedCount += 1
    # print (misClassifiedCount)
    # return True if misClassifiedCount <= 10 else False
    return misClassifiedCount

# read file
file = open("raw.txt", "r", encoding = "utf-8")

# 0 = not stressed, 1 = stressed
data = [(each.replace("\n", "").split("\t")) for each in file]


data = [(float(each[0]),float(each[1]),0) if data.index(each) < 200 else (float(each[0]), float(each[1]), 1) for each in data]
# the first 199 are not stressed, remaining 200 are stressed
notStressed = data[:199]
stressed = data[199:]

plt.scatter([item[1] for item in stressed],[item[0] for item in stressed],c='b',label='Stressed')
plt.scatter([item[1] for item in notStressed],[item[0] for item in notStressed],c='r',label='Not Stressed')
plt.legend(loc=(1, 0))
plt.xlabel("Husband")
plt.ylabel("Wife")
plt.title("All incomes annual in $100,000 units")
plt.show()


random.shuffle(notStressed)
random.shuffle(stressed)
trainingList = notStressed[:99] + stressed[:100]

plt.scatter([item[1] for item in stressed[:100]],[item[0] for item in stressed[:100]],c='b',label='Stressed')
plt.scatter([item[1] for item in notStressed[:99]],[item[0] for item in notStressed[:99]],c='r',label='Not Stressed')
plt.legend(loc='upper right')
plt.xlabel("Husband")
plt.ylabel("Wife")
plt.title("Training Data")
plt.show()

random.shuffle(trainingList)
testingList = notStressed[99:] + stressed[100:]

plt.scatter([item[1] for item in stressed[100:]],[item[0] for item in stressed[100:]],c='b',label='Stressed')
plt.scatter([item[1] for item in notStressed[99:]],[item[0] for item in notStressed[99:]],c='r',label='Not Stressed')
plt.legend(loc='upper right')
plt.xlabel("Husband")
plt.ylabel("Wife")
plt.title("Test Data")
plt.show()

startTime = time.time()

Eta = 0.001 #learningRate
wi = random.random()
wj = random.random()
w0 = random.randint(-1, 1)

misClassifiedList = []
iteration = 0
# while not isStopCriterionMet(testingList):
while True:
    misClassifiedCount = isStopCriterionMet(testingList)
    misClassifiedList.append(misClassifiedCount)
    if  misClassifiedCount/len(testingList) < 0.1:
        print ("\n{} % misclassified points in the test data.".format(misClassifiedCount/len(testingList)))
        break
    if iteration > 100:
        break
    for trainingData in trainingList:
        yHat = stepFunction(wi * trainingData[0] + wj * trainingData[1] + w0)
        wi += Eta * (trainingData[2] - yHat) * trainingData[0]
        wj += Eta * (trainingData[2] - yHat) * trainingData[1]
        w0 += Eta * (trainingData[2] - yHat)
    iteration += 1

runningTime = time.time() - startTime
print ("time: {}".format(runningTime))
print ("wi = {}".format(wi))
print ("wj = {}".format(wj))
print ("w0 = {}".format(w0))

plt.plot(misClassifiedList)
plt.show()