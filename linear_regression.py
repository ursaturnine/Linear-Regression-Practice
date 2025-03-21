import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math


# numpy handles 1 or multi-dimensional arrays
# pandas handles data
# matplotlib can create plots (we will plot the dataset; the linear regression model)
# scikit_learn will contain the linear regression models (e.g. linear regression, clustering, logistic regression etc.)

# read .csv into DataFrame
house_data = pd.read_csv("house_prices.csv")
size = house_data['sqft_living']
price = house_data['price']

# Transform Data - ML learning algos handle arrays not DataFrames - NumPy Array Shaping
x = np.array(size).reshape(-1, 1)
y = np.array(price).reshape(-1, 1)

# Use Linear Regression + fit() to train the model  
model = LinearRegression()

# Define a linear relationship between x - size and y - prices
# fit() trains model w/ gradient descent (MSE) - optimization algo!!
model.fit(x, y) 

# Evaluate model based MSE (variance) and R Value (standard deviation)
regression_model_mse = mean_squared_error(x, y)
print("MSE: ", math.sqrt(regression_model_mse))
print("R Squared Value: ", model.score(x, y))

# Get B-Values After the model fit

# this is b1 - slope of fitted linear regression line
print(model.coef_[0])

# this is b0 in our model - where y-value intercepts height parameter
print(model.intercept_[0])

# Visualize the Dataset

# Plot the Dataset
plt.scatter(x, y, color='green')
# Plot Linear Regression Line
plt.plot(x, model.predict(x), color='black')
plt.title("Linear Regression")
plt.xlabel('Size')
plt.ylabel('Price')
plt.show()


# Look at MSE and R Values to Determine if there's a linear relationship between size and prices given that there's outliers

# Make predictions with the model
# '2000' is the size, the explanatory variable, and the model will return a 'price', the dependent variable
print("Prediction by the model: ", model.predict([[20000]]))



# print(house_data)