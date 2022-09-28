# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 13:51:44 2022

@author: karen
"""

import numpy as np
import pandas as pd

# Load the Iris csv data using the Pandas library
filename = 'data.csv'
df = pd.read_csv(filename)
df2 = df.apply(pd.to_numeric,errors='coerce')

# Pandas returns a dataframe, (df) which could be used for handling the data.
# We will however convert the dataframe to numpy arrays for this course as 
# is also described in the table in the exercise
raw_data = df2.values  

# Notice that raw_data both contains the information we want to store in an array
# X (the sepal and petal dimensions) and the information that we wish to store 
# in y (the class labels, that is the iris species).

# We start by making the data matrix X by indexing into data.
# We know that the attributes are stored in the four columns from inspecting 
# the file.
cols = range(1, 10) 
X = raw_data[:,cols]

# Enumerate over the data to change "Absent" family history to 0 and "Present" family history to 1. 
# for i,j in enumerate(X):
#     j = 5
#     if X[i,j] == "Absent":
#         X[i,j] = 0
#     if X[i,j] == "Present":
#         X[i,j] = 1
print(X.dtype)
print(np.size(X))
print(np.shape(X))
print(X)
# Enumerate over the data to change "0" CHD to  and "Present" family history to 1. 
#X[:,10] = ['Response' if x == 1 else x for x in X[:,10]]
#X[:,10] = ['No response' if x == 0 else x for x in X[:,10]]

print(X[0,:],X[2,:])

# We can extract the attribute names that came from the header of the csv
attributeNames = np.asarray(df.columns[cols])
print('attributenames equals',np.size(attributeNames))

# Before we can store the class index, we need to convert the strings that
# specify the class of a given object to a numerical value. We start by 
# extracting the strings for each sample from the raw data loaded from the csv:
classLabels = X[:,-1] # -1 takes the last column
# Then determine which classes are in the data by finding the set of 
# unique class labels 
classNames = np.unique(classLabels)
print('classnames equals',classNames)
# We can assign each type of Iris class with a number by making a
# Python dictionary as so:
classDict = dict(zip(classNames,range(len(classNames))))
# The function zip simply "zips" togetter the classNames with an integer,
# like a zipper on a jacket. 
# For instance, you could zip a list ['A', 'B', 'C'] with ['D', 'E', 'F'] to
# get the pairs ('A','D'), ('B', 'E'), and ('C', 'F'). 
# A Python dictionary is a data object that stores pairs of a key with a value. 
# This means that when you call a dictionary with a given key, you 
# get the stored corresponding value. Try highlighting classDict and press F9.
# You'll see that the first (key, value)-pair is ('Iris-setosa', 0). 
# If you look up in the dictionary classDict with the value 'Iris-setosa', 
# you will get the value 0. Try it with classDict['Iris-setosa']

# With the dictionary, we can look up each data objects class label (the string)
# in the dictionary, and determine which numerical value that object is 
# assigned. This is the class index vector y:
y = np.array([classDict[cl] for cl in classLabels])
print(y)
# In the above, we have used the concept of "list comprehension", which
# is a compact way of performing some operations on a list or array.
# You could read the line  "For each class label (cl) in the array of 
# class labels (classLabels), use the class label (cl) as the key and look up
# in the class dictionary (classDict). Store the result for each class label
# as an element in a list (because of the brackets []). Finally, convert the 
# list to a numpy array". 
# Try running this to get a feel for the operation: 
# list = [0,1,2]
# new_list = [element+10 for element in list]

# We can determine the number of data objects and number of attributes using 
# the shape of X
N, M = X.shape



# Check the size and the change to binary data. 
print('N and M equals',N,M)
#print(X[0,:])

# Finally, the last variable that we need to have the dataset in the 
# "standard representation" for the course, is the number of classes, C:
C = len(classNames) 
print('C equals',C)
print(X.dtype)
# # Change to float: 
# for i in range(len(X)):
#     for j in range(len(X[i])):
#         X[i,j] = float(X[i,j])
#X2 = pd.to_numeric(X,errors='ignore')