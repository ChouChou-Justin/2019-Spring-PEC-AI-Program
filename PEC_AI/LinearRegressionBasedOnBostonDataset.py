# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 12:33:05 2019

@author: 電子二乙 106360211 周信廷 
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression



def linearTest(x, y):
    #轉成陣列
    x = np.array(x).reshape(len(x),1)
    y = np.array(y).reshape(len(y),1)
    #線性回歸
    clf = LinearRegression()
    clf.fit(x,y)
    pre = clf.predict(x)
    #畫圖
    plt.scatter(x,y,s=100)
    plt.plot(x,pre,"r-",linewidth=4)
    # x軸:TAX,y軸:MEDV
    plt.xlabel('RM')
    plt.ylabel('MEDV')
    for idx, m in enumerate(x):
        plt.plot([m,m],[y[idx],pre[idx]], 'g-')
    plt.show()
    print("係數", clf.coef_)
    print("截距", clf.intercept_)
    print(np.mean(y-pre)**2)
    print(clf.predict([[5.0]]))




#載入資料集
boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset.data, columns = boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target
#寫入 BostonDataset.csv
boston.to_csv("LinearRegressionBasedOnBostonDataset.csv")
boston_room = boston['RM']
boston_target = boston['MEDV']
x = boston_room
y = boston_target


#print(boston_dataset)
#print(boston_dataset.data)
print(boston_dataset.keys())
print(boston_dataset.feature_names)
print(boston)
print(x)
print(y)
linearTest(x, y)











