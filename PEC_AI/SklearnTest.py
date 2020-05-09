# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 02:00:54 2019

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression



def linearTest(x, y):
    #轉成陣列
    x = np.array(x).reshape(len(x),1)
    y = np.array(y).reshape(len(y),1)
    clf = LinearRegression()
    clf.fit(x,y)
    pre = clf.predict(x)
    #畫圖
    plt.scatter(x,y,s=100)
    plt.plot(x,pre,"r-",linewidth=4)
    for idx, m in enumerate(x):
        plt.plot([m,m],[y[idx],pre[idx]], 'g-')
    plt.show()
    print("係數", clf.coef_)
    print("截距", clf.intercept_)
    print(np.mean(y-pre)**2)
    print(clf.predict([[5.0]]))





hua = load_iris()
#獲取花瓣的長和寬
#x = [n[0] for n in hua.data]
#y = [n[1] for n in hua.data]
#linearTest(x, y)
#x = [1, 2, 3, 4, 5]
#y = [1, 5, 9, 12,14]
#linearTest(x, y)
x = np.random.rand(100).astype(np.float32)
y = x * 0.1 + 0.3
linearTest(x, y)









