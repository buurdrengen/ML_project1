# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 13:16:52 2022

@author: s196140
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 09:55:05 2022
@author: karen
"""

#tranforming data so mean = 0, STD = 1
import numpy as np
import pandas as pd
from sklearn import linear_model
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid,savefig,plot)
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection

# IMPORTING DATA
filename = 'data.csv'
df = pd.read_csv(filename)
raw_data = df.values  

cols = range(1, 10) 
X = raw_data[:,cols]
y = raw_data[:,10]
attributeNames = np.asarray(df.columns[cols])

#Transforming mean=0 og STD=1
# for i in range(len(cols)):
#     X[:,i] = (X[:,i]-np.mean(X[:,i]))/np.std(X[:,i])

# K = 10 fold Crossvalidation
cvf=10

# Values of lambda
lambdas = np.power(10.,range(0,5))

CV = model_selection.KFold(cvf, shuffle=True)
M = X.shape[1]
w = np.empty((M,cvf,len(lambdas)))
train_error = np.empty((cvf,len(lambdas)))
test_error = np.empty((cvf,len(lambdas)))
f = 0
y = y.squeeze()
for train_index, test_index in CV.split(X,y):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    # Standardize the training and set set based on training set moments
    mu = np.mean(X_train[:, 1:], 0)
    sigma = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu) / sigma
    X_test[:, 1:] = (X_test[:, 1:] - mu) / sigma
    
    # precompute terms
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    for l in range(0,len(lambdas)):
        # Compute parameters for current value of lambda and current CV fold
        lambdaI = lambdas[l] * np.eye(M)
        lambdaI[0,0] = 0 # remove bias regularization
        w[:,f,l] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
        # Evaluate training and test performance
        train_error[f,l] = np.power(y_train-X_train @ w[:,f,l].T,2).mean(axis=0)
        test_error[f,l] = np.power(y_test-X_test @ w[:,f,l].T,2).mean(axis=0)

    f=f+1

opt_val_err = np.min(np.mean(test_error,axis=0))
opt_lambda = lambdas[np.argmin(np.mean(test_error,axis=0))]
train_err_vs_lambda = np.mean(train_error,axis=0)
test_err_vs_lambda = np.mean(test_error,axis=0)
mean_w_vs_lambda = np.squeeze(np.mean(w,axis=1))
    
    
#Optimal generalization error
print('Optimal value of error:' , opt_val_err)
print('Optimal value of lambda:' , opt_lambda)

figure(1)
title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
xlabel('Regularization factor')
ylabel('Squared error (10 fold crossvalidation)')
legend(['Train error','Test error'])
grid()
savefig('regression1.png')
show()

figure(2)
title('Effects of the selected attributes')
semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
xlabel('Regularization factor')
ylabel('Mean Coefficient Values')
legend(attributeNames[1:], loc='best')
grid()
savefig('weigth1.png')
