# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 09:55:05 2022

@author: ESL
"""

#tranforming data so mean = 0, STD = 1
import numpy as np
import pandas as pd
from sklearn import linear_model

filename = 'data.csv'
df = pd.read_csv(filename)
df2 = df.apply(pd.to_numeric,errors='coerce')

raw_data = df2.values  

cols = range(1, 11) 
X = raw_data[:,cols]

#Transform mean=0 og STD=1 for each column except binary data
for i in range(4):
    X[:,i] = (X[:,i]-np.mean(X[:,i]))/np.std(X[:,i])
for i in range(4):
    X[:,i+5] = (X[:,i+5]-np.mean(X[:,i+5]))/np.std(X[:,i+5])
for i in range(10):
#    print(np.mean(X[:,i]))
#    print(np.std(X[:,i]))


