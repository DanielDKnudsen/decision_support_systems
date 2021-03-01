#imports

import pandas as pd
import numpy as np
from Exercises import Methods
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.graphics.regressionplots import plot_leverage_resid2
import statsmodels.formula.api as smf







#3.7.1
# Describe the null hypotheses to which the p-values given in Table 3.4
# correspond. Explain what conclusions you can draw based on these
# p-values. Your explanation should be phrased in terms of sales, TV,
# radio, and newspaper, rather than in terms of the coefficients of the
# linear model.
# For advertising data


# ___________________________________________________
#           | Coeff    std.error    t-statistic     p-value
# ____________________________________________________
# intercept | 2.939      0.3119      9.42            <0.0001
# TV        | 0.046      0.0014      32.81           <0.0001
# radio     | 0.189      0.0086      21.89           <0.0001
# newspaper | -0.001     0.0059      -0.18            0.8599
# __________________________________________________________


#___answer_____
# h_0 = advertisement in tv, radio and newspaper have no effect on sales. Also the null hypothesis for radio has no effect on sales
# tv and newspaper has ads. Same goes for newspaper. the low p-values of TV and radio suggest H0 are false and can be discarded
# The high value of P for newspaper suggest h_0 is true for newspaper



# 3.7.2
# Carefully explain the differences between the KNN classifier and KNN regression methods
#

# ____answer______
# KNN regression tries to predict the value of the output variable by using a local average.
# KNN classification attempts to predict the class to which the output variable belong by computing the local probability.



# This question involves the use of simple linear regression on the Auto data set.
# (a) Use the lm() function to perform a simple linear regression with mpg as the response and horsepower as the predictor.

# Load and display summary of auto dataset
auto = pd.read_csv("Auto.csv")
auto = auto.loc[:, ~auto.columns.str.contains('^Unnamed')]
print(auto.head())


# Query NaN values and clean up
# print(auto.dtypes)
print([auto.isnull().any(axis=1)])
auto = auto.dropna()

X = np.c_[auto.horsepower]
y = np.c_[auto.mpg]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
# print(X.shape)

auto.plot(kind='scatter', x='horsepower', y='mpg')
plt.show()
model = LinearRegression().fit(X_train, y_train)

print(model.predict(X_test))



# Use the summary() function to print the results. Comment on the output.
# For example:

results = smf.ols('mpg ~ horsepower', data=auto).fit()
print(results.summary())


# i: Is there a relationship between the predictor and the response?
# Yes there is a connection between MPG and horsepower.
# Since the F-statistic is far larger than 1 and the p-value of the F-statistic
# is close to zero we can reject the null hypothesis and state there is a
# statistically significant relationship between horsepower and mpg.


# ii: How strong is the relationship between the predictor and the response?
# Just from the summary it is very strong as the t-statistic is -24 though there is still lots of variation left in the model with an r-squared of .6

# iii: Is the relationship between the predictor and the response positive or negative?
# Negative. THe more HP the automobile has the less MPG fuel effecient it is.


# iv: What is the predicted mpg associated with a horsepower of 98? What are the associated 95% confidence and prediction intervals?
print(results.conf_int())

print('The predicted mpg associated with a horsepower of 98')
print(model.predict([[98]]))

print(results.bse)

print('The associated confidence')
print(Methods.create_interval(results, 'confidence', .05, auto['horsepower'], 98))

print('the associated prediction')
print(Methods.create_interval(results, 'prediction', .05, auto['horsepower'], 98))

# (b) Plot the response and the predictor. Use the abline() function to display the least squares regression line.
sns.regplot(x='horsepower', y='mpg', data=auto, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.show()


# (c) Use the plot() function to produce diagnostic plots of the least squares regression fit. Comment on any problems you see with the fit.
plt.scatter(results.fittedvalues, results.resid);
plt.show()

fig, ax = plt.subplots(figsize=(8,6))
fig = plot_leverage_resid2(results, ax = ax)
plt.show()
# This question involves the use of multiple linear regression on the Auto data set.
# (a) Produce a scatterplot matrix which includes all of the variables in the data set.

# uncomment if plot needed
sns.pairplot(auto)
plt.show()

# (b) Compute the matrix of correlations between the variables using the function cor().
# You will need to exclude the name variable, cor() which is qualitative.

print(auto.corr())

# (c) Use the lm() function to perform a multiple linear regression
# with mpg as the response and all other variables except name as
# the predictors. Use the summary() function to print the results.
# Comment on the output. For instance:

# "cylinders","displacement","horsepower","weight","acceleration","year","origin","name"

# X = np.c_[[auto.cylinders], [auto.displacement], [auto.horsepower], [auto.weight], [auto.acceleration], [auto.year], [auto.origin]]
# y = np.c_[auto.mpg]
#
# model = LinearRegression().fit(X, y)


formula = 'mpg ~ ' + " + ".join(auto.columns[1:-1])
print(formula)
# results = smf.ols(formula, data=auto).fit()
# print(results.summary())
# i. Is there a relationship between the predictors and the response?

# ii. Which predictors appear to have a statistically significant relationship to the response?

# iii. What does the coefficient for the year variable suggest?
