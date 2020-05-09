# -*- coding: utf-8 -*-
"""
Created on Mon May 27 23:43:36 2019

@author: ChouHsingTing
"""
import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
img = plt.imread('leena.bmp')
print(img.shape)
#print('Image shape:\n', img.shape)
#print(img[0,0])
pixels = np.reshape(img, (img.shape[0]*img.shape[1], 3))/255
#print('Pixels shape:\n', pixels.shape)
som2 = MiniSom(1, 5, 3, sigma = 0.1, learning_rate = 0.2)
som2.random_weights_init(pixels)
starting_weights = som2.get_weights().copy()
#print('Initial weights:\n', starting_weights)
#print('SOM shape:\n', starting_weights.shape)
som2.train_random(pixels, 500)
#print("activition\n",som2.activation_response(pixels))
qnt = som2.quantization(pixels)
#print("q_Error\n",som2.quantization_error(pixels))
#print('Result weights\n', som2.get_weights())
clustered = np.zeros(img.shape)
for i,q in enumerate(qnt):
    clustered[np.unravel_index(i,(img.shape[0],img.shape[1]))]=q
plt.figure(figsize=(10,10))
plt.subplot(221)
plt.title('Original')
plt.imshow(img)
plt.subplot(222)
plt.title('Result')
plt.imshow(clustered)
plt.subplot(223)
plt.title('Initial Colors')
plt.imshow(starting_weights)
plt.subplot(224)
plt.title('Learnt Colors')
plt.imshow(som2.get_weights())
plt.tight_layout()
plt.show()
