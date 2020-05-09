# -*- coding: utf-8 -*-
"""
Created on Thu May  9 17:47:03 2019

@author: ChouHsingTing
"""
ans = 0
x = 0
y = 1
def SimpleFactorial(n):
    if(n > 0):
        ans = n*SimpleFactorial(n-1)
        return ans
    if(n == 0): 
        return 1
#print(SimpleFactorial(3))
def Loop(x,y):
        for x in range(x<8):
            for y in range(1,5):
                print(x,y)
                x = x+2
                y = y+1
Loop(x,y)