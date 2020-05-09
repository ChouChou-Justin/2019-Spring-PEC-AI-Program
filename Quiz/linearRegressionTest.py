import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
import pandas as pd
def linearTest(x, y):
    #轉成陣列
    x = np.array(x).reshape(len(x),1)
    y = np.array(y).reshape(len(y),1)
    print(x)
    print(y)
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
   
    
def linearIris():
    hua = load_iris()
    x = [n[0] for n in hua.data]
    y = [n[1] for n in hua.data]
    linearTest(x, y)    
    
    
def main():
    linearIris()

main()    