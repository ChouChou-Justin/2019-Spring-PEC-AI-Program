# -*- coding: utf-8 -*-
"""
Created on Sun May  5 23:01:49 2019

@author: ChouHsingTing
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing
from sklearn import tree
from sklearn import metrics
import tensorflow as tf
from keras import models
import math


def linearTest(x, y):
    #轉成陣列
    bank_X = x
    bank_y = y
    train_X, test_X, train_y, test_y = train_test_split(bank_X, bank_y, test_size = 0.5)
    x = np.array(train_X).reshape(len(train_X),1)
    y = np.array(train_y).reshape(len(train_y),1)
    tx = np.array(test_X).reshape(len(test_X),1)
    ty = np.array(test_y).reshape(len(test_y),1)

    clf = LinearRegression()
    clf.fit(x,y)
    pre = clf.predict(tx)
    for i in range(len(pre)):
        if (pre[i]) >= 0.45:
            pre[i] = math.ceil(pre[i])
        else:
            pre[i] = pre[i].round()

    #畫圖
    plt.scatter(x,y,s=10)
    plt.plot(x,pre,"r-",linewidth=4)

    plt.ylabel('pep')
    for idx, m in enumerate(x):
        plt.plot([m,m],[y[idx],pre[idx]], 'g-')
    
    plt.show()
#    print("係數", clf.coef_)
#    print("截距", clf.intercept_)
#    print(np.mean(y-pre)**2)
#    print(clf.predict([[5.0]]))
    print(ty[289:299])
    print(pre[289:299])
    accuracy = metrics.accuracy_score(ty, pre)
    print("accuracy=", accuracy)
    

       

def linearBank():
    bank_dataset = pd.read_csv('bank_data.csv')
    
    le_age = preprocessing.LabelEncoder()
    le_sex = preprocessing.LabelEncoder()
    le_region = preprocessing.LabelEncoder()
    le_income = preprocessing.LabelEncoder()
    le_married = preprocessing.LabelEncoder()
    le_children = preprocessing.LabelEncoder()
    le_car = preprocessing.LabelEncoder()
    le_save_act = preprocessing.LabelEncoder()
    le_current_act = preprocessing.LabelEncoder()
    le_mortgage = preprocessing.LabelEncoder()
    le_pep = preprocessing.LabelEncoder()
    
    bank_dataset['age_'] = le_age.fit_transform(bank_dataset['age'])
    bank_dataset['sex_'] = le_sex.fit_transform(bank_dataset['sex'])
    bank_dataset['region_'] = le_region.fit_transform(bank_dataset['region'])  
    bank_dataset['income_'] = le_income.fit_transform(bank_dataset['income'])
    bank_dataset['married_'] = le_married.fit_transform(bank_dataset['married'])
    bank_dataset['children_'] = le_children.fit_transform(bank_dataset['children'])
    bank_dataset['car_'] = le_car.fit_transform(bank_dataset['car'])
    bank_dataset['save_act_'] = le_save_act.fit_transform(bank_dataset['save_act'])
    bank_dataset['current_act_'] = le_current_act.fit_transform(bank_dataset['current_act'])
    bank_dataset['mortgage_'] = le_mortgage.fit_transform(bank_dataset['mortgage'])
    bank_dataset['pep_'] = le_pep.fit_transform(bank_dataset['pep'])
    bank_dataset_ = bank_dataset.drop(['age','sex',
                                       'region','income',
                                       'married','children',
                                       'car','save_act','current_act',
                                       'mortgage','pep',
                                       'pep_'], axis='columns')
    
    bank_age = bank_dataset_['age_']
    bank_sex = bank_dataset['sex_']
    bank_region = bank_dataset['region_']
    bank_income = bank_dataset['income_']
    bank_married = bank_dataset['children_']
    bank_children = bank_dataset['children_']
    bank_car = bank_dataset['car_']
    bank_save_act = bank_dataset['save_act_']
    bank_current_act = bank_dataset['current_act_']
    bank_mortgage = bank_dataset['mortgage_']
    bank_target = bank_dataset['pep_']
    
 
    x1 = bank_age
    x2 = bank_sex
    x3 = bank_region
    x4 = bank_income
    x5 = bank_married
    x6 = bank_children
    x7 = bank_car
    x8 = bank_save_act
    x9 = bank_current_act
    x10 = bank_mortgage
    y = bank_target
    linearTest(x1, y)    
    linearTest(x2, y) 
    linearTest(x3, y) 
    linearTest(x4, y) 
    linearTest(x5, y) 
    linearTest(x6, y) 
    linearTest(x7, y) 
    linearTest(x8, y) 
    linearTest(x9, y) 
    linearTest(x10, y) 
    
    
def main():
    linearBank()

main()    