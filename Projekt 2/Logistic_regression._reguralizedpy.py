## Classification #1 ## 
# 
# Author: Aksel Buur Christensen, s203947
# 
# Logistic regression reguralized 
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from toolbox_02450 import rocplot, confmatplot
from import_HD_data import * 

font_size = 15
plt.rcParams.update({'font.size': font_size})

#Using 0.99 of the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y)


# Standardize the training and set set based on training set mean and std
mu = np.mean(X_train, 0)
sigma = np.std(X_train, 0)

X_train = (X_train - mu) / sigma
# print(X_train)
X_test = (X_test - mu) / sigma
# print(X_test) 

# Fit regularized logistic regression model to training data to predict a subject. 
lambda1 = 2.4770763559917088e-05


mdl = LogisticRegression(penalty='l2', C=1/lambda1 )
    
mdl.fit(X_train, y_train)

y_train_est = mdl.predict(X_train).T
# print(y_train_est)
y_test_est = mdl.predict(X_test).T
# print(y_test_est)
    
train_error_rate = np.sum(y_train_est != y_train) / len(y_train)
test_error_rate = np.sum(y_test_est != y_test) / len(y_test)

print('For lambda equals:',lambda1)
print('The training error:',train_error_rate)
print('The test error:',test_error_rate)


import numpy as np 
a1 = np.array([1,-1.4,2.6])
a2 = np.array([1,-0.6,-1.6])
a3 = np.array([1,2.1,5.0])
a4 = np.array([1,0.7,3.8])
w1 = np.array([1.2,-2.1,3.2])
w2 = np.array([1.2,-1.7,2.9])
w3 = np.array([1.3,-1.1,2.2])
print(np.dot(a1,w1))

print(1/(1+np.exp(np.dot(a1,w1))+np.exp(np.dot(a1,w2))+np.exp(np.dot(a1,w3))))
print(1/(1+np.exp(np.dot(a2,w1))+np.exp(np.dot(a2,w2))+np.exp(np.dot(a2,w3))))
print(1/(1+np.exp(np.dot(a3,w1))+np.exp(np.dot(a3,w2))+np.exp(np.dot(a3,w3))))
print(1/(1+np.exp(np.dot(a4,w1))+np.exp(np.dot(a4,w2))+np.exp(np.dot(a4,w3))))

# b = np.ones((10,1))
# print(b)
# c = np.ones([10])
# print(c)
# d = np.arange(10,21,dtype=int)
# print(d)
# d1 = np.concatenate((d,d,d,d,d,d,d,d,d,d),axis=0)
# print(np.size(d1))