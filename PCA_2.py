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
# AND divide by the standard deviation to standardize the data, since the attributes have very different scales/values. 
Y = (X - np.ones((N,1))*X.mean(axis=0))/ np.std(X)
print(np.shape(Y))
print(np.size(Y))

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)
print('U = ', U)
print('S = ', S)
print('V(3) = ', V[:,2])

# Compute variance explained by principal components bvb 
rho = (S*S) / (S*S).sum() 
print('rho = ', rho)

# The variance explained by the first three principal components: 
cumvar = rho[0] + rho[1] + rho[2] + rho[3]
print('cumvar = ',cumvar)

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