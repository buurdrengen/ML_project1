## Principal Component Analysis #2 ## 
#
# Author: Aksel Buur Christensen, 
# With inspiration scripts by the exercise toolbox scripts provided by the 02450 team. 
# 
# Purpose: Variance explained by the different components.  
from import_HD_data import *

import matplotlib.pyplot as plt
from scipy.linalg import svd
from mpl_toolkits.mplot3d import Axes3D

Y = (X - np.ones((N,1))*X.mean(0)) / np.std(X)
print(Y)
U,S,Vh = svd(Y,full_matrices=False)
V=Vh.T
N,M = X.shape

# We saw in 2.1.3 that the first 3 components explaiend more than 90
# percent of the variance. Let's look at their coefficients:
pcs = [0,1,2,3]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, attributeNames)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('CHD: PCA Component Coefficients')
plt.show()



# The different classes with great magnitude from all the principal components.
alcohol_data = Y[y==8,:]
sbp_data = Y[y==0,:]
age_data = Y[y==9,:]
print('V(1):')
print(V[:,0])
print('V(2):')
print(V[:,1])
print('V(3):')
print(V[:,2])

## Sbp 
print('sbp data:')
print(sbp_data[0,:])

print('SBPs projection onto PC1')
print(sbp_data[0,:]@V[:,0])
print('SBPs projection onto PC2')
print(sbp_data[0,:]@V[:,1])
print('SBPs projection onto PC3')
print(sbp_data[0,:]@V[:,2])

## Alcohol
print('alcohol_data')
print(alcohol_data[0,:])

# Based on the coefficients and the attribute values for the observation
# displayed, would you expect the projection onto PC2 to be positive or
# negative - why? Consider *both* the magnitude and sign of *both* the
# coefficient and the attribute!

# You can determine the projection by (remove comments):
print('Alcohols projection onto PC1')
print(alcohol_data[0,:]@V[:,0])
print('Alcohols projection onto PC2')
print(alcohol_data[0,:]@V[:,1])
print('Alcohols projection onto PC3')
print(alcohol_data[0,:]@V[:,2])
# Try to explain why?

## Age 
print('Age data')
print(age_data[0,:])

print('Age projection onto PC1')
print(age_data[0,:]@V[:,0])
print('Age projection onto PC2')
print(age_data[0,:]@V[:,1])
print('Age projection onto PC3')
print(age_data[0,:]@V[:,2])
