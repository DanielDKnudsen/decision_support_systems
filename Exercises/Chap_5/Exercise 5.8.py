import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

# 8. We will now perform cross-validation on a simulated data set
#
# (a) Generate a simulated data set as follows:

#  set.seed (1)
#  y=rnorm (100)
#  x=rnorm (100)
#  y=x-2* x^2+ rnorm (100)


np.random.seed(1)
X = np.random.randn(100)
e = np.random.randn(100)
y = X - 2*X**2 + e

# np.random.seed(1)
# y = np.random.rand(100)
# X = np.random.rand(100)
#
# y = X-2*X**2+y
print(y)


# In this data set, what is n and what is p? Write out the model used to generate the data in equation form.

# (b) Create a scatterplot of X against Y . Comment on what you find.
plt.scatter(X, y)
plt.show()


# (c) Set a random seed, and then compute the LOOCV errors that
# result from fitting the following four models using least squares:

df = pd.DataFrame(np.array([np.ones(len(X)), X, X ** 2, X ** 3, X ** 4, y]).T, columns=['b0', 'x', 'x2', 'x3', 'x4', 'y'])
print(df.head())


X = df.iloc[:, :5]
y = df['y']
model = LinearRegression()
errors = np.zeros((len(X), 4))
for i in range(len(X)):
    leave_out  = ~X.index.isin([i])
    for j in range(4):
        model.fit(X.iloc[leave_out, :j+2], y[leave_out])
        errors[i, j] = (model.predict([X.iloc[i, :j+2]]) - y[i]) ** 2

# each error here is average error for linear, quadratic, cubic and quartic model.
# Looks like it stabilizes at quadratic.
print("Average error for Linear, quadratic, cubic and quartic model, respectively")
print(errors.mean(axis=0))

# Note you may find it helpful to use the data.frame() function to create a single data set containing both X and Y


# (d) Repeat (c) using another random seed, and report your results.
# Are your results the same as what you got in (c)? Why?
# again with different seed.
print("Creating new random seeds")

np.random.seed(2)
X = np.random.randn(100)
e = np.random.randn(100)
y = X - 2*X**2 + e


# np.random.seed(2)
# y = np.random.rand(100)
# X = np.random.rand(100)
#
# y = X-2*X**2+y

df = pd.DataFrame(np.array([np.ones(len(X)), X, X ** 2, X ** 3, X ** 4, y]).T, columns=['b0', 'x', 'x2', 'x3', 'x4', 'y'])
print(df.head())

X = df.iloc[:, :5]
y = df['y']
model = LinearRegression()
errors = np.zeros((len(X), 4))
for i in range(len(X)):
    leave_out  = ~X.index.isin([i])
    for j in range(4):
        model.fit(X.iloc[leave_out, :j+2], y[leave_out])
        errors[i, j] = (model.predict([X.iloc[i, :j+2]]) - y[i]) ** 2

# quite a different average error. But again stabilizes at quadratic which makes sense
print("Errors with new seeds")
print("Average error for Linear, quadratic, cubic and quartic model, respectively")
print(errors.mean(axis=0))




# (e) Which of the models in (c) had the smallest LOOCV error? Is
# this what you expected? Explain your answer.

# (f) Comment on the statistical significance of the coefficient estimates
# that results from fitting each of the models in (c) using
# least squares. Do these results agree with the conclusions drawn
# based on the cross-validation results?

#since the error doesn't improve after quadratic it's likely the standard errors for x3 and x4 would not be significant