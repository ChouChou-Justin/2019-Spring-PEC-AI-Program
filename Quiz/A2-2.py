# -*- coding: utf-8 -*-
"""
Created on Mon May  6 17:28:13 2019

@author: ChouHsingTing
"""

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing
from keras import models
from keras import layers
import matplotlib.pyplot as plt

df = pd.read_csv('bank_data.csv')
print(df.shape)

encode_tf = preprocessing.LabelEncoder()
df['age_n'] = encode_tf.fit_transform(df['age'])
df['sex_n'] = encode_tf.fit_transform(df['sex'])
df['region_n'] = encode_tf.fit_transform(df['region'])
df['income_n'] = encode_tf.fit_transform(df['income'])
df['married_n'] = encode_tf.fit_transform(df['married'])
df['children_n'] = encode_tf.fit_transform(df['children'])
df['car_n'] = encode_tf.fit_transform(df['car'])
df['save_act_n'] = encode_tf.fit_transform(df['save_act'])
df['current_act_n'] = encode_tf.fit_transform(df['current_act'])
df['mortgage_n'] = encode_tf.fit_transform(df['mortgage'])
df['pep_n'] = encode_tf.fit_transform(df['pep'])

print(df.shape)

df_n = df.drop(['age', 'sex', 'region', 'income', 'married', 'children', 'car', 'save_act', 'current_act', 'mortgage', 'pep'], axis=1)
label_data = df_n['pep_n']
train_data = df_n.drop(['pep_n'], axis=1)

print(label_data.shape)
print(train_data.shape)

train_x, test_x, train_y, test_y = train_test_split(train_data, label_data, test_size = 0.5)

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_shape=(train_x.shape[1],)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mse', metrics=['acc'])

train_history = model.fit(train_x, train_y, epochs=100, batch_size=16, validation_data=(test_x, test_y))


y_pred = model.predict_classes(test_x[290:300])


a = [290,291,292,293,294,295,296,297,298,299]
x=0

for i in a:
    print("The prediction for",i+1,'is',y_pred[x])
    x=x+1


plt.scatter(a, y_pred)
plt.scatter(a, test_y[290:300])
plt.show()   

 
print(y_pred)
print(test_y[290:300].shape)