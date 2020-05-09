# -*- coding: utf-8 -*-
"""
Created on Sun May  5 18:35:18 2019

@author: ChouHsingTing
"""

import pandas as pd
import numpy as np
import pydotplus
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree
from sklearn import metrics


def decisionTree():
    #載入資料集
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
    target = bank_dataset['pep_']
    bank_dataset_ = bank_dataset.drop(['age','sex',
                                       'region','income',
                                       'married','children',
                                       'car','save_act','current_act',
                                       'mortgage','pep',
                                       'pep_'], axis='columns')
    bank_dataset_.hist()
    bank_X = bank_dataset_
    bank_y = target
    train_X, test_X, train_y, test_y = train_test_split(bank_X, bank_y, test_size = 0.5)
    clf = tree.DecisionTreeClassifier()
    bank_clf = clf.fit(train_X, train_y)
    test_y_predicted = bank_clf.predict(test_X)
    print(test_y_predicted[289:299])
    accuracy = metrics.accuracy_score(test_y, test_y_predicted)
    print(accuracy)
    
    graph = tree.export_graphviz(bank_clf, 
                                  feature_names=bank_dataset_.columns,
                                  out_file=None,
                                  filled=True,
                                  rounded=True)
    pydot_graph = pydotplus.graph_from_dot_data(graph)
    pydot_graph.write_png('A3.png')
    pydot_graph.write_pdf('A3.pdf')

def main():
    decisionTree()
    
main()



    

   











