## Principal Component Analysis #1 ## 
#
# Author: Aksel Buur Christensen, 
# With inspiration scripts by the exercise toolbox scripts provided by the 02450 team. 
# 
# Purpose: Script to play around with plotting attributes to look for interesting relations.  
#
# Imports the numpy and xlrd package, then runs the ex2_1_1 code
from import_HD_data import *

from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show

# # Data attributes to be plotted
# i = 4
# j = 9

print(X[:,0],X[:,1])

print(np.correlate(X[:,0],X[:,1]))

matrix = np.round(np.corrcoef(X),3)
print(matrix)


##
# Make a simple plot of the i'th attribute against the j'th attribute
# Notice that X is of matrix type (but it will also work with a numpy array)
# X = np.array(X) #Try to uncomment this line
# plot(X[:, i], X[:, j], 'o')

# %%
# Make another more fancy plot that includes legend, class labels, 
# attribute names, and a title.
f = figure()
title('CHD data')

plot(X[:,7],X[:,8],'o',alpha=.3)
show()
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
