import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn import model_selection
from sklearn import model_selection,tree
from sklearn.model_selection import train_test_split
from toolbox_02450 import rocplot, confmatplot
from import_HD_data import * 

## Tree classifier ## 

CV = model_selection.KFold(n_splits=10)

i = 0 
# Predictions 
yhat = []
y_true = []

# Lambda interval 
lambda_interval = np.logspace(-5, 5, 10)

# Error 
test_error = []

for train_index, test_index in CV.split(X,y):
    print('Crossvalidation fold: {0}/{1}'.format(i+1,10))

    # Extract training and test set for current CV fold 
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    # Standardize data 
    # mu = np.mean(X_train,0)
    # sigma = np.std(X_train,0)

    # X_train = ( X_train - mu ) / sigma
    # X_test = ( X_test - mu ) / sigma 

    # Fit classifier - tree classifier.
    dtc = tree.DecisionTreeClassifier(criterion='gini',max_depth=15)
    dtc = dtc.fit(X_train,y_train)
    y_est = dtc.predict(X_test)
    
    # Calculate the error
    de_test = np.sum(y_est != y_test) / len(y_test)

    # Save the estimate
    yhat.append(y_est)
    y_true.append(y_test) 
    test_error.append(de_test)


    i += 1 

yhat = np.concatenate(yhat)
y_true = np.concatenate(y_true) 
print('test-error:',test_error)
print('lambda:',lambda_interval)
min_error = np.min(test_error)
opt_lambda_idx = np.argmin(test_error)
opt_lambda = lambda_interval[opt_lambda_idx]
print('optimal lambda',opt_lambda)
z = y_true - yhat