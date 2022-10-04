## Principal Component Analysis #1 ## 
#
# Author: Aksel Buur Christensen, 
# With inspiration scripts by the exercise toolbox scripts provided by the 02450 team. 
# 
# Purpose: Script to play around with plotting attributes to look for interesting relations.  
#
# Imports the numpy and xlrd package, then runs the ex2_1_1 code
from import_HD_data import *
from scipy import stats
import matplotlib.pyplot as plt
# # Data attributes to be plotted
# i = 4
# j = 9

# Statistics for attributes
print('mean:')
print(np.mean(X[:,0]))
print(np.mean(X[:,1]))
print(np.mean(X[:,2]))
print(np.mean(X[:,3]))
print(np.mean(X[:,4]))
print(np.mean(X[:,5]))
print(np.mean(X[:,6]))
print(np.mean(X[:,7]))
print(np.mean(X[:,8]))

print('std:')
print(np.std(X[:,0]))
print(np.std(X[:,1]))
print(np.std(X[:,2]))
print(np.std(X[:,3]))
print(np.std(X[:,4]))
print(np.std(X[:,5]))
print(np.std(X[:,6]))
print(np.std(X[:,7]))
print(np.std(X[:,8]))

print('median:')
print(np.median(X[:,0]))
print(np.median(X[:,1]))
print(np.median(X[:,2]))
print(np.median(X[:,3]))
print(np.median(X[:,4]))
print(np.median(X[:,5]))
print(np.median(X[:,6]))
print(np.median(X[:,7]))
print(np.median(X[:,8]))

print('range:')
print(np.max(X[:,0])-np.min(X[:,0]))
print(np.max(X[:,1])-np.min(X[:,1]))
print(np.max(X[:,2])-np.min(X[:,2]))
print(np.max(X[:,3])-np.min(X[:,3]))
print(np.max(X[:,4])-np.min(X[:,4]))
print(np.max(X[:,5])-np.min(X[:,5]))
print(np.max(X[:,6])-np.min(X[:,6]))
print(np.max(X[:,7])-np.min(X[:,7]))
print(np.max(X[:,8])-np.min(X[:,8]))
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
# Plot the histogram
nbins = 20
f = plt.figure(figsize=(16,8))
plt.subplot(4,2,1)
plt.title('SBP')
plt.hist(X[:,0], bins=nbins, density=True)
# Over the histogram, plot the theoretical probability distribution function:
x = np.linspace(X[:,0].min(), X[:,0].max(), 1000)
pdf = stats.norm.pdf(x,loc=138.33,scale=20.50)
plt.plot(x,pdf,'.',color='red')

plt.subplot(4,2,2)
plt.title('CT')
plt.hist(X[:,1], bins=nbins, density=True)
# Over the histogram, plot the theoretical probability distribution function:
x = np.linspace(X[:,1].min(), X[:,1].max(), 1000)
pdf = stats.norm.pdf(x,loc=3.64,scale=4.59)
plt.plot(x,pdf,'.',color='red')

plt.subplot(4,2,3)
plt.title('LDL')
plt.hist(X[:,2], bins=nbins, density=True)
# Over the histogram, plot the theoretical probability distribution function:
x = np.linspace(X[:,2].min(), X[:,2].max(), 1000)
pdf = stats.norm.pdf(x,loc=4.74,scale=2.07)
plt.plot(x,pdf,'.',color='red')

plt.subplot(4,2,4)
plt.title('BAI')
plt.hist(X[:,3], bins=nbins, density=True)
# Over the histogram, plot the theoretical probability distribution function:
x = np.linspace(X[:,3].min(), X[:,3].max(), 1000)
pdf = stats.norm.pdf(x,loc=25.41,scale=7.78)
plt.plot(x,pdf,'.',color='red')

plt.subplot(4,2,5)
plt.title('TA')
plt.hist(X[:,5], bins=nbins, density=True)
# Over the histogram, plot the theoretical probability distribution function:
x = np.linspace(X[:,5].min(), X[:,5].max(), 1000)
pdf = stats.norm.pdf(x,loc=53.10,scale=9.82)
plt.plot(x,pdf,'.',color='red')

plt.subplot(4,2,6)
plt.title('BMI')
plt.hist(X[:,6], bins=nbins, density=True)
# Over the histogram, plot the theoretical probability distribution function:
x = np.linspace(X[:,6].min(), X[:,6].max(), 1000)
pdf = stats.norm.pdf(x,loc=26.04,scale=4.21)
plt.plot(x,pdf,'.',color='red')

plt.subplot(4,2,7)
plt.title('CAC')
plt.hist(X[:,7], bins=nbins, density=True)
# Over the histogram, plot the theoretical probability distribution function:
x = np.linspace(X[:,7].min(), X[:,7].max(), 1000)
pdf = stats.norm.pdf(x,loc=17.04,scale=24.48)
plt.plot(x,pdf,'.',color='red')

plt.subplot(4,2,8)
plt.title('AGE')
plt.hist(X[:,8], bins=nbins, density=True)
# Over the histogram, plot the theoretical probability distribution function:
x = np.linspace(X[:,8].min(), X[:,8].max(), 1000)
pdf = stats.norm.pdf(x,loc=42.82,scale=14.61)
plt.plot(x,pdf,'.',color='red')
plt.subplot_tool()
plt.show()
##
# Make a simple plot of the i'th attribute against the j'th attribute
# Notice that X is of matrix type (but it will also work with a numpy array)
# X = np.array(X) #Try to uncomment this line
# plot(X[:, i], X[:, j], 'o')

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
