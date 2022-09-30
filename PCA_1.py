## Principal Component Analysis #1 ## 
#
# Author: Aksel Buur Christensen, 
# With inspiration scripts by the exercise toolbox scripts provided by the 02450 team. 
# 
# Purpose: Script to play around with plotting attributes to look for interesting relations.  
#
# Imports the numpy and xlrd package, then runs the ex2_1_1 code
from import_HD_data import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
# # Data attributes to be plotted
# i = 4
# j = 9

# Correlation?

print(X[:,0],X[:,1])

print(np.correlate(X[:,0],X[:,1]))

matrix = np.round(np.corrcoef(X),3)
print(matrix)


# Boxplot? 

fig = plt.figure(1)

plt.boxplot(X)
plt.show()

# Attributes normal distributed? 
# Number of bins in histogram
nbins = 20
# Plot the samples and histogram
plt.figure(figsize=(12,30))

plt.subplot(4,2,1)
plt.hist(X[:,0], bins=nbins)
x = np.linspace(X[:,0].min(), X[:,0].max(), 1000)
pdf = stats.norm.pdf(x,loc=138.33,scale=20.5)
plt.plot(x,pdf,'.',color='red')
plt.title('SBP')

plt.subplot(4,2,2)
plt.hist(X[:,1], bins=nbins)
x = np.linspace(X[:,1].min(), X[:,1].max(), 1000)
pdf = stats.norm.pdf(x,loc=3.64,scale=4.59)
plt.plot(x,pdf,'.',color='red')
plt.title('CT')

plt.subplot(4,2,3)
plt.hist(X[:,2], bins=nbins)
x = np.linspace(X[:,2].min(), X[:,2].max(), 1000)
pdf = stats.norm.pdf(x,loc=4.74,scale=2.07)
plt.plot(x,pdf,'.',color='red')
plt.title('LDL')

plt.subplot(4,2,4)
plt.hist(X[:,3], bins=nbins)
x = np.linspace(X[:,3].min(), X[:,3].max(), 1000)
pdf = stats.norm.pdf(x,loc=25.41,scale=7.78)
plt.plot(x,pdf,'.',color='red')
plt.title('BAI')

plt.subplot(4,2,5)
plt.hist(X[:,5], bins=nbins)
x = np.linspace(X[:,5].min(), X[:,5].max(), 1000)
pdf = stats.norm.pdf(x,loc=53.10,scale=9.82)
plt.plot(x,pdf,'.',color='red')
plt.title('TA')

plt.subplot(4,2,6)
plt.hist(X[:,6], bins=nbins)
x = np.linspace(X[:,6].min(), X[:,6].max(), 1000)
pdf = stats.norm.pdf(x,loc=26.04 ,scale=4.21)
plt.plot(x,pdf,'.',color='red')
plt.title('BMI')

plt.subplot(4,2,7)
plt.hist(X[:,7], bins=nbins)
x = np.linspace(X[:,7].min(), X[:,7].max(), 1000)
pdf = stats.norm.pdf(x,loc=17.04,scale=24.48)
plt.plot(x,pdf,'.',color='red')
plt.title('CAC')

plt.subplot(4,2,8)
plt.hist(X[:,8], bins=nbins)
x = np.linspace(X[:,8].min(), X[:,8].max(), 1000)
pdf = stats.norm.pdf(x,loc=42.82 ,scale=14.61)
plt.plot(x,pdf,'.',color='red')
plt.title('AGE')

plt.subplot_tool()
plt.show()



##
# Make a simple plot of the i'th attribute against the j'th attribute
# Notice that X is of matrix type (but it will also work with a numpy array)
##plot(X[:, i], X[:, j], 'o')

# %%
# Make another more fancy plot that includes legend, class labels, 
# attribute names, and a title.
#f = figure()
#title('CHD data')

#plot(X[:,7],X[:,8],'o',alpha=.3)
#show()
# for c in range(C):
#     # select indices belonging to class c:
#     class_mask = y==c
#     plot(X[class_mask,i], X[class_mask,j], 'o',alpha=.3)

# legend(classNames) # Make a line explaining it is CHD! 
# xlabel(attributeNames[i])
# ylabel(attributeNames[j])

# # Output result to screen
# show()
# print('Ran Exercise 2.1.2')
