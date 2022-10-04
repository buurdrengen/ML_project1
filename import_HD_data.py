# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 13:51:44 2022

@author: karen / aksel 
"""

import numpy as np
import pandas as pd

filename = 'data.csv'
df = pd.read_csv(filename)
df2 = df.apply(pd.to_numeric,errors='coerce')

raw_data = df2.values  

cols = range(1, 10) 
X = raw_data[:,cols]

# print(X.dtype)
# print(np.size(X))
# print(np.shape(X))
# print(X)

# print(X[0,:],X[2,:])

# We can extract the attribute names that came from the header of the csv
attributeNames = np.asarray(df.columns[cols])
print('attributenames equals',np.size(attributeNames))

classLabels = X[:,-1] # -1 takes the last column

classNames = np.unique(classLabels)
print('classnames equals',classNames)
# We can assign each type of Iris class with a number by making a
# Python dictionary as so:
classDict = dict(zip(classNames,range(len(classNames))))


#This is the class index vector y:
y = np.array([classDict[cl] for cl in classLabels])
print(y)

N, M = X.shape

print('N and M equals',N,M)
#print(X[0,:])

C = len(classNames) 
print('C equals',C)
print(X.dtype)
