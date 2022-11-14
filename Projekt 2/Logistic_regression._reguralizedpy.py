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




#u.append(min(a1[0:2]))
# u.append(np.min(a1[2:5]))
# u.append(np.min(a1[6:9]))
#print(u)
import numpy as np 
a1 = np.array([0.243,0.365,0.763,0.458,0.245,0.378,0.256,0.287,0.393,0.374,0.532,0.489])
u = []
h_idx = []
for i, e in enumerate(a1): 
    h_idx.append(i)
    u.append(min(a1[10*i:10*i+2]))
    print(i)
    print(e)
    print(h_idx)
    print(u)
h = []

h1 = np.argmin(a1)
print(h1)
if h1 <= 2: 
    h1 = h1 
if h1/3 == 1: 
    h1 = 0
if h1%3 == 1: 
    h1 = 1
if h1%3 == 2: 
    h1 = 2
print(h1)

for j in range(0,3):
    print(j)
    b = a1[j::3]
    print(b)
    h.append(np.argmin(b))
    print(b,h)

u = []
n = np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11]])
print(n)
for j in n:
    print(j)
    u.append(min(a1[n[j,:]]))
    print(u)
# c = np.ones([10])
# print(c)
# d = np.arange(10,21,dtype=int)
# print(d)
# d1 = np.concatenate((d,d,d,d,d,d,d,d,d,d),axis=0)
# print(np.size(d1))