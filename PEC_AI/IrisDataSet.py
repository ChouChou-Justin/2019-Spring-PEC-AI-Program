# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 01:58:05 2019

@author: user
"""
#匯入資料集iris
from sklearn.datasets import load_iris
#載入資料集
iris = load_iris()
#輸出資料集
print(iris.data)

#輸出真實標籤
print(iris.target)
print(len(iris.target))

#150個樣本每個樣本4個特徵
print(iris.data.shape)
print(iris.feature_names)