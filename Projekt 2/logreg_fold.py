import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn import model_selection
from sklearn import model_selection,tree
from sklearn.linear_model import LogisticRegression
from toolbox_02450 import rocplot, confmatplot
from import_HD_data import * 
from platform import system
from os import getcwd
from toolbox_02450 import windows_graphviz_call
from matplotlib.image import imread

CV = model_selection.KFold(n_splits=10)

i = 0 
# Predictions 
yhat_log = []
y_true_log = []

# Predictions 
yhat_tree = []
y_true_tree = []

# Predictions 
yhat_base = []
y_true_base = []

# Lambda interval 
lambda_interval = np.logspace(-5, 1, 10)

# Depth interval 
depth_interval = np.arange(10,20,dtype=int)
interval = np.vstack((lambda_interval,depth_interval))

# Error 
test_error_log = []
test_error_tree = []
test_error_base = []

# From HD_data it is counted from y how many subjects has CHD (1) and how many does not have CHD (0).
sizey = np.size(y) # sizey = 462
no_CHD = np.count_nonzero(y)
CHD = sizey - no_CHD
print('chd,no-chd:',CHD,',',no_CHD)

for train_index, test_index in CV.split(X,y):
    print('Crossvalidation fold: {0}/{1}'.format(i+1,10))

    # Extract training and test set for current CV fold 
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    # Standardize data for log
    mu = np.mean(X_train,0)
    sigma = np.std(X_train,0)

    X_train_log = ( X_train - mu ) / sigma
    X_test_log = ( X_test - mu ) / sigma 

    # Fit classifier - logistic regression.
    dy_log = []
    lgr = LogisticRegression(penalty='l2',C=1/lambda_interval[i])
    lgr = lgr.fit(X_train_log,y_train)
    y_est_log = lgr.predict(X_test_log)
    dy_log.append(y_est_log)
    #print('dytest',dy_log)
    
    # Calculate the error
    de_log = np.sum(y_est_log != y_test) / len(y_test)


    # Fit classifier - TreeClassifier
    dy_tree = []
    criterion = 'gini'
    dtc = tree.DecisionTreeClassifier(criterion = criterion,max_depth=depth_interval[i])
    dtc = dtc.fit(X_train,y_train)
    y_est_tree = dtc.predict(X_test)
    dy_tree.append(y_est_tree)
    # Calculate the error
    de_tree = np.sum(y_est_tree != y_test) / len(y_test)

    # Fit classifier - Baseline
    dy_base = []
    y_est_base = np.ones((np.size(y_est_tree),1))
    dy_base.append(y_est_base)
    # Calculate the error
    de_base = np.sum(y_est_base != y_test) / len(y_test)

    # Save the estimate log
    dy_log = np.stack(dy_log, axis = 1)
    #print('dy-log',dy_log)
    yhat_log.append(dy_log)
    #print(yhat_log)
    y_true_log.append(y_test) 
    test_error_log.append(de_log)

    # Save the estimate tree
    dy_tree = np.stack(dy_tree, axis = 1)
    #print('dy-tree',dy_tree)
    yhat_tree.append(dy_tree)
    #print(yhat_tree)
    y_true_tree.append(y_test) 
    test_error_tree.append(de_tree)

    # Save the estimate tree
    dy_base = np.stack(dy_base, axis = 1)
    #print('dy-tree',dy_tree)
    yhat_base.append(dy_base)
    #print(yhat_tree)
    y_true_base.append(y_test) 
    test_error_base.append(de_base)

    i += 1 


# Log regression results:
yhat_log = np.concatenate(yhat_log)
#print('yhat:',yhat)
y_true_log = np.concatenate(y_true_log) 
print('test-error_log:',test_error_log)
print('lambda:',lambda_interval)
min_error_log = np.min(test_error_log)
opt_lambda_idx = np.argmin(test_error_log)
opt_lambda = lambda_interval[opt_lambda_idx]
print('optimal lambda',opt_lambda)
z_log = y_true_log - yhat_log

# Tree results:
yhat_tree = np.concatenate(yhat_tree)
#print('yhat:',yhat)
y_true_tree = np.concatenate(y_true_tree) 
print('test-error_tree:',test_error_tree)
print('depth:',depth_interval)
min_error = np.min(test_error_tree)
opt_depth_idx = np.argmin(test_error_tree)
opt_depth = depth_interval[opt_depth_idx]
print('optimal depth',opt_depth)
z_tree = y_true_tree - yhat_tree

# Base results:
yhat_base = np.concatenate(yhat_base)
#print('yhat:',yhat)
y_true_base = np.concatenate(y_true_base) 
print('test-error_base:',test_error_base)
# min_error = np.min(test_error_tree)
# opt_depth_idx = np.argmin(test_error_tree)
# opt_depth = depth_interval[opt_depth_idx]
# print('optimal depth',opt_depth)
z_base = y_true_base - yhat_base

# Plot log 
plt.figure(1)
plt.title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
plt.loglog(lambda_interval,test_error_log,'b.-')
plt.xlabel('Regularization factor')
plt.ylabel('Error rate (10 fold crossvalidation)')
plt.legend(['Test error'])
plt.grid()
plt.savefig('logreg_fold.png')
plt.show()

# Plot tree
plt.figure(2)
plt.title('Optimal depth')
plt.plot(depth_interval,test_error_tree,'r.-')
plt.xlabel('Regularization factor')
plt.ylabel('Error rate (10 fold crossvalidation)')
plt.legend(['Test error'])
plt.grid()
plt.savefig('tree_fold.png')
plt.show()

print('Accuracy_log:',np.count_nonzero(z_log)/np.size(z_log))
print('Accuracy_tree:',np.count_nonzero(z_tree)/np.size(z_tree))
print('Accuracy_base:',np.count_nonzero(z_base)/np.size(z_base))