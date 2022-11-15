## Accuracy - MCnemerar ## 
from logreg_fold import * 
from toolbox_02450.statistics import *

# Compute the Jeffreys interval
alpha = 0.05

# Logistic vs tree 
[thetahat, CI, p] = mcnemar(y_true_log, yhat_log[:,0], yhat_tree[:,0], alpha=alpha)

# Logistic vs base
[thetahatB, CIB, pB] = mcnemar(y_true_log, yhat_log[:,0], yhat_base[:,0], alpha=alpha)

# Tree vs base 
[thetahatC, CIC, pC] = mcnemar(y_true_tree, yhat_tree[:,0], yhat_base[:,0], alpha=alpha)


print("theta = theta_A-theta_B point estimate", thetahat, " CI: ", CI, "p-value", p)
print("theta = theta_A-theta_C point estimate", thetahatB, " CI: ", CIB, "p-value", pB)
print("theta = theta_B-theta_C point estimate", thetahatC, " CI: ", CIC, "p-value", pC)