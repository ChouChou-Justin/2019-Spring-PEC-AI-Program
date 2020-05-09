# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 19:36:26 2019

@author: 電子二乙 106360211 周信廷
"""

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree
from sklearn import metrics
import pydotplus
import pandas as pd
import numpy as np

def decisionTree():
    treeFile = pd.read_csv('DecisionTreeExportGraphvizBasedOnTennis.csv')
    le_Day = preprocessing.LabelEncoder()
    le_Outlook = preprocessing.LabelEncoder()
    le_Temp = preprocessing.LabelEncoder()
    le_Humidity = preprocessing.LabelEncoder()
    le_Wind = preprocessing.LabelEncoder()
    le_PlayTennis = preprocessing.LabelEncoder()

    treeFile['Day_'] = le_Day.fit_transform(treeFile['Day'])
    treeFile['Outlook_'] = le_Outlook.fit_transform(treeFile['Outlook'])
    treeFile['Temp_'] = le_Temp.fit_transform(treeFile['Temp.'])  
    treeFile['Humidity_'] = le_Humidity.fit_transform(treeFile['Humidity'])
    treeFile['Wind_'] = le_Wind.fit_transform(treeFile['Wind'])
    treeFile['PlayTennis_'] = le_PlayTennis.fit_transform(treeFile['Play Tennis'])
    target = treeFile['PlayTennis_']
    treeFile_ = treeFile.drop(['Day','Outlook','Temp.','Humidity','Wind','Play Tennis','PlayTennis_'], axis='columns')

    treeFile.hist() 
    treeFile.plot.kde()
    clf = tree.DecisionTreeClassifier()
    tree_clf = clf.fit(treeFile_, target)
    graph = tree.export_graphviz(tree_clf, 
                                  feature_names=treeFile_.columns,
                                  out_file=None,
                                  filled = True,
                                  rounded=True)
    pydot_graph = pydotplus.graph_from_dot_data(graph)
    pydot_graph.write_png('DecisionTreeExportGraphvizBasedOnTennis.png')
    pydot_graph.write_pdf('DecisionTreeExportGraphvizBasedOnTennis.pdf')

def main():
    decisionTree()

main()