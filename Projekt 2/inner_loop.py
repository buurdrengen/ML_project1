# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 17:57:19 2022

@author: s196140
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
import pandas as pd
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats

def ANN_validate(X,y,hidden_units,cvf):
    
    M = X.shape[1]
    n_replicates = 1        # number of networks trained in each k-fold
    max_iter = 5000

    # K-fold crossvalidation
    K = cvf                 # only three folds to speed up this example
    CV = model_selection.KFold(K, shuffle=True)


    errors = [] # make a list for storing generalizaition error in each loop
    for h_index in hidden_units:
        for (k, (train_index, test_index)) in enumerate(CV.split(X,y)): 
            print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
            print('h = {0}'.format(h_index))
            # Define the model
            model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M, h_index), #M features to n_hidden_units
                                torch.nn.Tanh(),   # 1st transfer function,
                                torch.nn.Linear(h_index, 1), # n_hidden_units to 1 output neuron
                                # no final tranfer function, i.e. "linear output"
                                )
            loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
            
            # Extract training and test set for current CV fold, convert to tensors
            X_train = torch.Tensor(X[train_index,:])
            y_train = torch.Tensor(y[train_index])
            X_test = torch.Tensor(X[test_index,:])
            y_test = torch.Tensor(y[test_index])
            
            # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(model,
                                                               loss_fn,
                                                               X=X_train,
                                                               y=y_train,
                                                               n_replicates=n_replicates,
                                                               max_iter=max_iter)
            
        
            # Determine estimated class labels for test set
            y_test_est = net(X_test)
            
            # Determine errors and errors
            se = (y_test_est.float()-y_test.float())**2 # squared error
            mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
            errors.append(mse) # store error rate for current CV fold 
            #errors.append(h_index)
            
    return errors