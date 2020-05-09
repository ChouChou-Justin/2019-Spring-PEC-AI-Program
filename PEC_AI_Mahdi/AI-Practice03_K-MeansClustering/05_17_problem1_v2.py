import random
import math
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

def PredictClass(testingList):
    PredictionResult=[]
    for testingData in testingList:
        yHat = stepFunction(wi * testingData[0] + wj * testingData[1] + w0)
        PredictionResult.append(yHat)
    return PredictionResult

startTime = time.time()

# read file
file = open("raw.txt", "r", encoding = "utf-8")

# 0 = not stressed, 1 = stressed
data = [(each.replace("\n", "").split("\t")) for each in file]
data = [(float(each[0]), float(each[1]), 0) if data.index(each) < 200 else (float(each[0]), float(each[1]), 1) for each in data]
xxx=[]
yyy=[]
zzz=[]
for item in data:
    xxx.append(item[0])
    yyy.append(item[1])
    zzz.append(item[2])
# the first 199 are not stressed, remaining 200 are stressed
notStressed = data[:199]
stressed = data[199:]

random.shuffle(notStressed)
random.shuffle(stressed)
trainingList = notStressed[:99] + stressed[:100]
random.shuffle(trainingList)
testingList = notStressed[99:] + stressed[100:]

learningRate = 0.0004
wi = random.random()
wj = random.random()
w0 = random.random()

misClassifiedList = []
iteration = 0
# while not isStopCriterionMet(testingList):
while True:
    misClassifiedCount = isStopCriterionMet(testingList)
    misClassifiedList.append(misClassifiedCount)
    if misClassifiedCount < 30:
        print ("converged")
        break
    if iteration > 100:
        break
    for trainingData in trainingList:
        yHat = stepFunction(wi * trainingData[0] + wj * trainingData[1] + w0)
        wi += learningRate * (trainingData[2] - yHat) * trainingData[0]
        wj += learningRate * (trainingData[2] - yHat) * trainingData[1]
        w0 += learningRate * (trainingData[2] - yHat)
    iteration += 1

runningTime = time.time() - startTime
print ("time: {}".format(runningTime))
print (wi)
print (wj)
print (w0)

plt.plot(misClassifiedList)
plt.show()

plt.scatter(xxx,yyy,c=zzz)
plt.show()

plt.scatter(xxx,yyy,c=PredictClass(data))
plt.show()
