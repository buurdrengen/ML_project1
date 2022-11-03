# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 13:16:52 2022

@author: s196140
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 09:55:05 2022
@author: ESL
"""

#tranforming data so mean = 0, STD = 1
import numpy as np
import pandas as pd
from sklearn import linear_model
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid,savefig)
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate

filename = 'data.csv'
df = pd.read_csv(filename)
df2 = df.apply(pd.to_numeric,errors='coerce')

raw_data = df2.values  

cols = range(1, 11) 
X = raw_data[:,cols]

#Transform mean=0 og STD=1
#Transform mean=0 og STD=1 for each column except binary data
for i in range(4):
    X[:,i] = (X[:,i]-np.mean(X[:,i]))/np.std(X[:,i])
for i in range(4):
    X[:,i+5] = (X[:,i+5]-np.mean(X[:,i+5]))/np.std(X[:,i+5])



X = (X-np.mean(X))/np.std(X)


#%%
attributeNames = np.asarray(df.columns[cols])

classLabels = X[:,-1] # -1 takes the last column

classNames = np.unique(classLabels)

# Python dictionary as so:
classDict = dict(zip(classNames,range(len(classNames))))


#This is the class index vector y:
y = np.array([classDict[cl] for cl in classLabels])


#%%
## Crossvalidation

# Values of lambda
lambdas = np.power(100.,range(-3,3))
#%%
A = rlr_validate(X,y,lambdas,cvf=10)


# opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda
figure()
title('Optimal lambda: 1e{0}'.format(np.log10(A[1])))
loglog(lambdas,A[3].T,'b.-',lambdas,A[4].T,'r.-')
xlabel('Regularization factor')
ylabel('Squared error (crossvalidation)')
legend(['Train error','Validation error'])
grid()
savefig('regression1.png')
show()
