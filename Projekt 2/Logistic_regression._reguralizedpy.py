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


# Create crossvalidation partition for evaluation
# using stratification and 95 pct. split between training and test 
#K = 20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99, stratify=y)
# Try to change the test_size to e.g. 50 % and 99 % - how does that change the 
# effect of regularization? How does differetn runs of  test_size=.99 compare 
# to eachother?

# Standardize the training and set set based on training set mean and std
mu = np.mean(X_train, 0)
print(mu)
sigma = np.std(X_train, 0)
print(sigma)

X_train = (X_train - mu) / sigma
print(X_train)
X_test = (X_test - mu) / sigma
print(X_test)
# Fit regularized logistic regression model to training data to predict 
# the type of wine
lambda_interval = np.logspace(-10, 10, 50)
#print(lambda_interval)
train_error_rate = np.zeros(len(lambda_interval))
test_error_rate = np.zeros(len(lambda_interval))
coefficient_norm = np.zeros(len(lambda_interval))
for k in range(0, len(lambda_interval)):
    mdl = LogisticRegression(penalty='l2', C=1/lambda_interval[k] )
    
    mdl.fit(X_train, y_train)

    y_train_est = mdl.predict(X_train).T
    y_test_est = mdl.predict(X_test).T
    
    train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
    test_error_rate[k] = np.sum(y_test_est != y_test) / len(y_test)

    w_est = mdl.coef_[0] 
    coefficient_norm[k] = np.sqrt(np.sum(w_est**2))

min_error = np.min(test_error_rate)
opt_lambda_idx = np.argmin(test_error_rate)
opt_lambda = lambda_interval[opt_lambda_idx]
print(opt_lambda)
plt.figure(figsize=(8,8))
#plt.plot(np.log10(lambda_interval), train_error_rate*100)
#plt.plot(np.log10(lambda_interval), test_error_rate*100)
#plt.plot(np.log10(opt_lambda), min_error*100, 'o')
plt.semilogx(lambda_interval, train_error_rate*100)
plt.semilogx(lambda_interval, test_error_rate*100)
plt.semilogx(opt_lambda, min_error*100, 'o')
plt.text(0.001, 3, "Minimum test error: " + str(np.round(min_error*100,2)) + ' % at ' + str(np.round(opt_lambda,2)))
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.ylabel('Error rate (%)')
plt.title('Classification error')
plt.legend(['Training error','Test error','Test minimum'],loc='upper right')
plt.ylim([0, 50])
plt.grid()
plt.savefig('logregerror_plot.png')
plt.show()


plt.figure(figsize=(8,8))
plt.semilogx(lambda_interval, coefficient_norm,'k')
plt.ylabel('L2 Norm')
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.title('Parameter vector L2 norm')
plt.grid()
plt.savefig('l2norm_logreg.png')
plt.show()    


print('Ran Exercise 9.1.1')