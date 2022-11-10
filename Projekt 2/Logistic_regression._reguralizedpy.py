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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99, stratify=y)


# Standardize the training and set set based on training set mean and std
mu = np.mean(X_train, 0)
sigma = np.std(X_train, 0)

X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma

# Fit regularized logistic regression model to training data to predict a subject. 
lambda1 = 0.1 


mdl = LogisticRegression(penalty='l2', C=1/lambda1 )
    
mdl.fit(X_train, y_train)

y_train_est = mdl.predict(X_train).T
y_test_est = mdl.predict(X_test).T
    


# plt.figure(figsize=(8,8))
# #plt.plot(np.log10(lambda_interval), train_error_rate*100)
# #plt.plot(np.log10(lambda_interval), test_error_rate*100)
# #plt.plot(np.log10(opt_lambda), min_error*100, 'o')
# plt.semilogx(lambda_interval, train_error_rate*100)
# plt.semilogx(lambda_interval, test_error_rate*100)
# plt.semilogx(opt_lambda, min_error*100, 'o')
# plt.text(0.001, 3, "Minimum test error: " + str(np.round(min_error*100,2)) + ' % at ' + str(np.round(opt_lambda,2)))
# plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
# plt.ylabel('Error rate (%)')
# plt.title('Classification error')
# plt.legend(['Training error','Test error','Test minimum'],loc='upper right')
# plt.ylim([0, 50])
# plt.grid()
# plt.savefig('logregerror_plot.png')
# plt.show()


# plt.figure(figsize=(8,8))
# plt.semilogx(lambda_interval, coefficient_norm,'k')
# plt.ylabel('L2 Norm')
# plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
# plt.title('Parameter vector L2 norm')
# plt.grid()
# plt.savefig('l2norm_logreg.png')
# plt.show()    


print('Ran Logreg-regularized')