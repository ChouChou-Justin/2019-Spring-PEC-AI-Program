# -*- coding: utf-8 -*-
"""
Created on Mon May  6 17:08:16 2019

@author: ChouHsingTing
"""

import tensorflow as tf 
import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd  
import networkx as nx
from apyori import apriori 
#pip install apriori
from wordcloud import WordCloud 
#pip install wordcloud




def testApriori():    
    records = []      
    store_data = pd.read_csv('bank_data_n.csv', header=None)  
    #print(store_data)
    print(store_data.head())
    #perprocessing
    #convert our pandas dataframe into a list of lists
    for i in range(0, 101):  
        #records.append([str(store_data.values[i,j]) for j in range(0, 20)])  
        records.append([str(store_data.values[i,j]) for j in range(0, 12) if str(store_data.values[i,j]) != 'nan'])
        # remove NaN value
    #print(records)  
    association_rules = apriori(records, min_support=0.1, min_confidence=0.2, min_lift=2, min_length=2)  
    #min_length: at least 2 product in the rules
    association_results = list(association_rules)  
    print(len(association_results))  
    #print(association_results)  
    print(association_results[0])  
    df=pd.DataFrame(association_results)
    df.to_csv("association_results.csv")  
    print(df.head())

    for item in association_results:
        # first index of the inner list
        # Contains base item and add item
        pair = item[0] 
        items = [x for x in pair]
        print("Rule: " + items[0] + " -> " + items[1])
        #second index of the inner list
        print("Support: " + str(item[1]))
        #third index of the list located at 0th
        #of the third index of the inner list
        print("Confidence: " + str(item[2][0][2]))
        print("Lift: " + str(item[2][0][3]))
        print("=====================================")    
    
def main():
  
    testApriori()
    

main()    