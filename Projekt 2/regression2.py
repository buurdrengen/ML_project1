# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 11:43:19 2022

@author: s196140
"""
#%% IMPORT DATA
import numpy as np
import pandas as pd
from sklearn import linear_model
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid,savefig,plot)
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate
from ANN_model import*

#%% IMPORT DATA
filename = 'data.csv'
df = pd.read_csv(filename)
raw_data = df.values  

cols = range(1, 10) 
X = raw_data[:,cols]
y = raw_data[:,10]

    
N, M = X.shape


# K1 = K2 = 10 Crossvalidation
K = 10
CV = model_selection.KFold(K, shuffle=True)

# Values of lambda
lambdas = np.power(10.,range(-5,5))

# Initialize variables
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
y_predict_nofeature = np.empty((K))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))

k=0
for train_index, test_index in CV.split(X,y):
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10    
    
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

    # Standardize outer fold based on training set
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    #BASELINE
    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]  
    
    #2 LEVEL CROSS VALDIDATION
    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Does not regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    y_prediction = X_test @ w_rlr[:,k]
    
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]
  
    # Display Table
    print('Cross validation fold {0}/{1}:'.format(k+1,K))
    print('- ANN h:                            {0}'.format(h_value[k]))
    print('- ANN Error:                        {0}'.format(errors1[k]))
    print('- Linear regression lambda:         {0}'.format(lambdas[k]))
    print('- Linear regression test error:     {0}'.format(Error_test_rlr[k]))
    print('- Baseline Test error:              {0}'.format(Error_test_nofeatures[k]))


    k+=1


