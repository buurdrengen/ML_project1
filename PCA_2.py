## Principal Component Analysis #2 ## 
#
# Author: Aksel Buur Christensen, 
# With inspiration scripts by the exercise toolbox scripts provided by the 02450 team. 
# 
# Purpose: Variance explained by the different components.  
from import_HD_data import *
#from PCA_1 import * 

import matplotlib.pyplot as plt
from scipy.linalg import svd

#print(X)
print(np.size(X))
print(N)
# Subtract mean value from data
Y = X - np.ones((N,1))*X.mean(axis=0)
print(np.shape(Y))
print(np.size(Y))


# PCA by computing SVD of Y
U,S,V = np.linalg.svd(Y,full_matrices=False)

# Compute variance explained by principal componentsbvb
rho = (S*S) / (S*S).sum() 

threshold = 0.95

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

print('Ran Exercise 2.1.3')