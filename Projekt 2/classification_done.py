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
outerCV = model_selection.KFold(n_splits=10)
i = 0 
j = 0
# Predictions 
yhat_log = []
y_true_log = []
yhat_log_out = []
y_true_log_out = []

# Predictions 
yhat_tree = []
y_true_tree = []
y_hat_tree_out = []
y_true_tree_out = []

# Predictions 
yhat_base = []
y_true_base = []
y_hat_base_out = []
y_true_base_out = []
# Lambda interval 
lambda_interval = np.logspace(-5, 1.5, 10)
lambda1 = np.concatenate((lambda_interval,lambda_interval,lambda_interval,lambda_interval,lambda_interval,lambda_interval,lambda_interval,lambda_interval,lambda_interval,lambda_interval),axis=0)
# Depth interval 
depth_interval = np.arange(10,20,dtype=int)
depth_1 = np.concatenate((depth_interval,depth_interval,depth_interval,depth_interval,depth_interval,depth_interval,depth_interval,depth_interval,depth_interval,depth_interval),axis=0)
print(np.size(depth_interval))
print(np.size(depth_1))
# Error 
test_error_log = []
test_error_tree = []
test_error_base = []

test_error_log_out = []
test_error_tree_out = []
test_error_base_out = []

# Lambda & depth outer 
lambda_out = []
depth_out = []
# From HD_data it is counted from y how many subjects has CHD (1) and how many does not have CHD (0).
sizey = np.size(y) # sizey = 462
no_CHD = np.count_nonzero(y) #no CHD = 160
CHD = sizey - no_CHD # CHD = 302
print('chd,no-chd:',CHD,',',no_CHD)
for train_index1, test_index1 in outerCV.split(X,y):
    print('Crossvalidation fold: {0}/{1}'.format(j+1,10))
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
        lgr = LogisticRegression(penalty='l2',C=1/lambda1[i])
        lgr = lgr.fit(X_train_log,y_train)
        y_est_log = lgr.predict(X_test_log)
        dy_log.append(y_est_log)
        #print('dytest',dy_log)
        
        # Calculate the error log
        de_log = np.sum(y_est_log != y_test) / len(y_test)


        # Fit classifier - TreeClassifier
        dy_tree = []
        criterion = 'gini'
        dtc = tree.DecisionTreeClassifier(criterion = criterion,max_depth=depth_1[i])
        dtc = dtc.fit(X_train,y_train)
        y_est_tree = dtc.predict(X_test)
        dy_tree.append(y_est_tree)

        # Calculate the error tree
        de_tree = np.sum(y_est_tree != y_test) / len(y_test)

        # Fit classifier - Baseline
        dy_base = []
        y_est_base = np.ones((np.size(y_est_tree))).T
        dy_base.append(y_est_base)
        # print('yestbase',y_est_base)
        # print('ytest',y_test)
        # Calculate the error
        de_base = np.sum(y_est_base != y_test) / len(y_test)
        print(de_base)

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

        # Save the estimate base
        dy_base = np.stack(dy_base, axis = 1)
        #print('dy-tree',dy_tree)
        yhat_base.append(dy_base)
        #print(yhat_tree)
        y_true_base.append(y_test) 
        test_error_base.append(de_base)

        i += 1
    
    opt_lambda_idx = np.argmin(test_error_log)
    lambda_out.append(lambda1[opt_lambda_idx])
    
    opt_depth_idx = np.argmin(test_error_tree)
    depth_out.append(depth_1[opt_depth_idx])
    # Extract training and test set for current CV fold 
    print('lambda',lambda_out)
    print('depth',depth_out)
    X_train = X[train_index1,:]
    y_train = y[train_index1]
    X_test = X[test_index1,:]
    y_test = y[test_index1]

    # Standardize data for log
    mu = np.mean(X_train,0)
    sigma = np.std(X_train,0)

    X_train_log = ( X_train - mu ) / sigma
    X_test_log = ( X_test - mu ) / sigma 

    # Fit classifier - logistic regression.
    dy_log_out = []
    lgr = LogisticRegression(penalty='l2',C=1/lambda_interval[j])
    lgr = lgr.fit(X_train_log,y_train)
    y_est_log_out = lgr.predict(X_test_log)
    dy_log_out.append(y_est_log_out)
    #print('dytest',dy_log)
        
    # Calculate the error log
    de_log_out = np.sum(y_est_log_out != y_test) / len(y_test)

    # Fit classifier - TreeClassifier
    dy_tree_out = []
    criterion = 'gini'
    dtc = tree.DecisionTreeClassifier(criterion = criterion,max_depth=depth_out[j])
    dtc = dtc.fit(X_train,y_train)
    y_est_tree_out = dtc.predict(X_test)
    dy_tree_out.append(y_est_tree_out)

    # Calculate the error tree
    de_tree_out = np.sum(y_est_tree_out != y_test) / len(y_test)

    # Fit classifier - Baseline
    dy_base_out = []
    y_est_base_out = np.ones((np.size(y_est_tree_out))).T
    dy_base_out.append(y_est_base_out)
    print('yestbase',y_est_base_out)
    print('ytest',y_test)
    # Calculate the error
    de_base_out = np.sum(y_est_base_out != y_test) / len(y_test)
    print('error base', de_base_out)

    # Save the estimate log
    dy_log_out = np.stack(dy_log_out, axis = 1)
    #print('dy-log',dy_log)
    yhat_log_out.append(dy_log_out)
    #print(yhat_log)
    y_true_log_out.append(y_test) 
    test_error_log_out.append(de_log_out)

    # Save the estimate tree
    dy_tree_out = np.stack(dy_tree_out, axis = 1)
    #print('dy-tree',dy_tree)
    y_hat_tree_out.append(dy_tree_out)
    #print(yhat_tree)
    y_true_tree_out.append(y_test) 
    test_error_tree_out.append(de_tree_out)

    # Save the estimate base
    dy_base_out = np.stack(dy_base_out, axis = 1)
    #print('dy-tree',dy_tree)
    y_hat_base_out.append(dy_base_out)
    #print(yhat_tree)
    y_true_base_out.append(y_test) 
    test_error_base_out.append(de_base_out)

    j += 1
    
    

# Log regression results:
yhat_log = np.concatenate(yhat_log_out)
#print('yhat:',yhat)
y_true_log = np.concatenate(y_true_log_out) 
print('test-error_log:',test_error_log_out)
print('lambda:',lambda_interval)
#min_error_log = np.min(test_error_log)
opt_lambda_idx = np.argmin(test_error_log_out)
opt_lambda = lambda_out[opt_lambda_idx]
print('optimal lambda',opt_lambda)
z_log = y_true_log - yhat_log

# Tree results:
yhat_tree = np.concatenate(y_hat_tree_out)
#print('yhat:',yhat)
y_true_tree = np.concatenate(y_true_tree_out) 
print('test-error_tree:',test_error_tree_out)
print('depth:',depth_interval)
min_error = np.min(test_error_tree)
opt_depth_idx = np.argmin(test_error_tree_out)
opt_depth = depth_out[opt_depth_idx]
print('optimal depth:',opt_depth)
z_tree = y_true_tree - yhat_tree

# Base results:
yhat_base = np.concatenate(y_hat_base_out)
#print('yhat:',yhat)
y_true_base = np.concatenate(y_true_base_out) 
print('test-error_base:',test_error_base_out)
# min_error = np.min(test_error_tree)
# opt_depth_idx = np.argmin(test_error_tree)
# opt_depth = depth_interval[opt_depth_idx]
# print('optimal depth',opt_depth)
z_base = y_true_base - yhat_base

# Plot log 
plt.figure(1)
plt.title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
plt.loglog(lambda_interval,test_error_log_out,'b.-')
plt.grid()
plt.xlabel('Regularization factor')
plt.ylabel('Error rate (10 fold crossvalidation)')
plt.legend(['Test error'])
#plt.savefig('logreg_fold.png')
plt.show()

# Plot tree
plt.figure(2)
plt.title('Optimal depth')
plt.plot(depth_out,test_error_tree_out,'r.-')
plt.xlabel('Regularization factor')
plt.ylabel('Error rate (10 fold crossvalidation)')
plt.legend(['Test error'])
plt.grid()
#plt.savefig('tree_fold.png')
plt.show()

print('Accuracy_log:{0}'.format(np.count_nonzero(z_log)/np.size(z_log)))
print('Accuracy_tree:',np.count_nonzero(z_tree)/np.size(z_tree))
print('Accuracy_base:',np.count_nonzero(z_base)/np.size(z_base))