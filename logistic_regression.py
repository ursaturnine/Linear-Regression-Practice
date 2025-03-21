from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np

"""
Outcomes can be either y = 0 or y = 1 for classification problems which use binary classification.

Logistic Regression will apply a probability to the x-value to classify the outcome as y = 0 or y = 1.

The following example will demonstrate that the higher the input number, the higher the likelihood the output will be 1 or 'yes'.
Similarly, it will show that the lower the input number, the higher the likelihood the output will be 0 or 'no'.

"""

x1 = np.array([0, 0.6, 1.1, 1.5, 1.8, 2.5, 3, 3.1, 3.9, 4, 4.9, 5, 5.1])

y1 = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0])

x2 = np.array([3, 3.8, 4.4, 5.2, 5.5, 6.5, 6, 6.1, 6.9, 7, 7.9, 8, 8.1])

y2 = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1])

x = np.array(
    [
        [0], [0.6], [1.1], [1.5], [1.8], [2.5], [3], [3.1], [3.9], [4], [4.9], [5], [5.1],
        [3], [3.8], [4.4], [5.2], [5.5], [6.5], [6], [6.1], [6.9], [7], [7.9], [8], [8.1] 
    ]
)

y = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1,1,1,1])


plt.plot(x1, y1, 'ro', color='blue')
plt.plot(x2, y2, 'ro', color='red')
plt.show()


model = LogisticRegression()
model.fit(x, y)


# Gradient Descent to fit model - find optimal b-parameters via using the sigmoid function

# collect b-values to input into sigmoid function
print('b0 is:', model.intercept_)
print('b1 is:', model.coef_)


# use coefficients for tuning the model to make further predictions

# sigmoid function used in maximum likelihood estimation (instead of using gradient descent to predict optimal b-value to tune the model)
def logistic(classifier: x):
    return 1/(1 + np.exp(-(model.intercept_ + model.coef_ * x)))

#  plot the logistic function
for i in range(1, 120):
    plt.plot(i /10.0 - 2, logistic(model, i/10.0), 'ro', color='green')


plt.axis([-2, 10, -.05, 2])
plt.show()

# Use model to make predictions - Supervised Learning

# the higher the input you use for 'pred', the higher the probability the predicted outcome will be a 'yes' or closer to '1'
# the lower the input you use for 'pred', the higher the probability the predicted outcome will be a 'no' or closer to '0'
pred = model.predict_proba([10])

# 'pred' will output an array showing a percentage for probability of 0 or 'no' at index 0 of the array and for yes or '1' at index 1 of the array
print("Probability: ", pred)