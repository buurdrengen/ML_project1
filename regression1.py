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
from toolbox_02450 import rlr_validate

filename = 'data.csv'
df = pd.read_csv(filename)


raw_data = df.values  

cols = range(1, 10) 
X = raw_data[:,cols]
y = raw_data[:,10]

#Transform mean=0 og STD=1
for i in range(len(cols)):
    X[:,i] = (X[:,i]-np.mean(X[:,i]))/np.std(X[:,i])


## K = 10 fold Crossvalidation

# Values of lambda
lambdas = np.power(10.,range(1,5))
#lambdas = np.linspace(0.01,10,10)
A = rlr_validate(X,y,lambdas,cvf=10)

# opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda

#Optimal generalization error
print('Optimal value of error:' , A[0])
print('Optimal value of lambda:' ,A[1])

figure(1)
title('Optimal lambda: 1e{0}'.format(np.log10(A[1])))
loglog(lambdas,A[3].T,'b.-',lambdas,A[4].T,'r.-')
xlabel('Regularization factor')
ylabel('Squared error (10 fold crossvalidation)')
legend(['Train error','Test error'])
grid()
savefig('regression1.png')
show()

figure(2)
title('Effects of the selected attributes')
semilogx(lambdas,A[2].T[:,1:],'.-') # Don't plot the bias term
xlabel('Regularization factor')
ylabel('Mean Coefficient Values')
grid()
savefig('weigth1.png')
