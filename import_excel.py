# Test for importing Excel data file # 
# exercise 2.1.1
import numpy as np
import xlrd

# Load xls sheet with data
doc = xlrd.open_workbook("data_copy.xls").sheet_by_index(0)

# Extract attribute names (1st row, column 4 to 12)
attributeNames = doc.row_values(0, 1, 12)

# Extract class names to python list,
# then encode with integers (dict)
classLabels = doc.col_values(0, 2,462)
print('classlabels:',classLabels)

classNames = sorted(set(classLabels))
print('classnames:',classNames)

classDict = dict(zip(classNames, range(13)))

# Extract vector y, convert to NumPy array
y = np.asarray([classDict[value] for value in classLabels])

# Preallocate memory, then extract excel data to matrix X
X = np.empty((462, 12))
for i, col_id in enumerate(range(1, 12)):
    X[:, i] = np.asarray(doc.col_values(col_id, 2, 462))

# Compute values of N, M and C.
N = len(y)
M = len(attributeNames)
C = len(classNames)
print('M and N equals',N,M)
print('X equals:',X)

print('Shape of X:',np.shape(X))
print('Ran Exercise 2.1.1')