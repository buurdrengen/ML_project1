## Classification #1 ## 
# 
# Author: Aksel Buur Christensen, s203947
# 
# Logistic regression unreguralized 
from matplotlib.pylab import figure, plot, xlabel, ylabel, legend, ylim, show
import sklearn.linear_model as lm
from import_HD_data import * 

# Fit logistic regression model

model = lm.LogisticRegression()
model = model.fit(X,y)

# Classify CHD-response as "No CHD"/"CHD" (0/1) and assess probabilities
y_est = model.predict(X)
y_est_healthy_prob = model.predict_proba(X)[:,0]
print('y_estimated:',y_est)
print('y_healthy_estimated:',y_est_healthy_prob)

misses = np.sum(y_est != y)
print(f'Amount of misses: {misses} of {len(y_est)}.')
# Define a new data object (using first patient from the data.csv)
x = np.array([160, 12, 5.73, 23.11, 1, 49, 25.3, 97.2, 52]).reshape(1,-1)
# Evaluate the probability of x having CHD: 
x_class = model.predict_proba(x)[0,1]

# Evaluate classifier's misclassification rate over entire training data
misclass_rate = np.sum(y_est != y) / float(len(y_est))

# Display classification results
print('\nProbability of given sample having CHD: {0:.4f}'.format(x_class))
print('\nOverall misclassification rate: {0:.3f}'.format(misclass_rate))

f = figure();
class0_ids = np.nonzero(y==0)[0].tolist()
plot(class0_ids, y_est_healthy_prob[class0_ids], '.y')
class1_ids = np.nonzero(y==1)[0].tolist()
plot(class1_ids, y_est_healthy_prob[class1_ids], '.r')
xlabel('Data object (patient)'); ylabel('Predicted prob. of class "No-CHD"');
legend(['No CHD', 'CHD'])
ylim(-0.01,1.01)

show()

