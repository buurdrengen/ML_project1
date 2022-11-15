
import numpy as np
from sklearn import model_selection
import sklearn.tree
import scipy.stats
import numpy as np, scipy.stats as st
import torch
#%%

# requires data from exercise 1.5.1
# requires data from exercise 1.5.1
from ANN_model import y_test_est, y_test
from regression2 import y_prediction, y_predict_nofeature

#%% Comparing ANN model VS Linear Regression
# perform statistical comparison of the models
# compute z with squared error.

#y_test_est = y_test_est.detach().numpy()
zA = np.abs(y_test - y_test_est[:,0]) ** 2

# compute confidence interval of model A
alpha = 0.05
CIA = st.t.interval(1-alpha, df=len(zA)-1, loc=np.mean(zA), scale=st.sem(zA))  # Confidence interval

# Compute confidence interval of z = zA-zB and p-value of Null hypothesis
zB = np.abs(y_test - y_prediction) ** 2
z = zA - zB
CI1 = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p1 = 2*st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value
print(p1)
print(CI1)

#%% Comparing ANN model VS Baseline
# perform statistical comparison of the models
# compute z with squared error.
zA = np.abs(y_test - y_test_est[:,0]) ** 2

# compute confidence interval of model A
alpha = 0.05
CIA = st.t.interval(1-alpha, df=len(zA)-1, loc=np.mean(zA), scale=st.sem(zA))  # Confidence interval

# Compute confidence interval of z = zA-zB and p-value of Null hypothesis
zB = np.abs(y_test - y_test.mean()) ** 2
z = zA - zB
CI2 = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p2 = 2*st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value
print(p2)
print(CI2)

#%% Comparing Baseline VS Linear Regression
# perform statistical comparison of the models
# compute z with squared error.
zA = np.abs(y_test - y_prediction) ** 2

# compute confidence interval of model A
alpha = 0.05
CIA = st.t.interval(1-alpha, df=len(zA)-1, loc=np.mean(zA), scale=st.sem(zA))  # Confidence interval

# Compute confidence interval of z = zA-zB and p-value of Null hypothesis
zB = np.abs(y_test - y_test.mean()) ** 2
z = zA - zB
CI3 = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p3 = 2*st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-
print(p3)
print(CI3)