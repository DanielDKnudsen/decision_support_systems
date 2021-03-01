import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import OLSInfluence
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import plot_leverage_resid2
import matplotlib.pyplot as plt
import sys

from itertools import combinations
import seaborn as sns

# Load and display summary of auto dataset
auto = pd.read_csv("Auto.csv")
auto = auto.loc[:, ~auto.columns.str.contains('^Unnamed')]
print(auto.head())

# Query NaN values and clean up
auto = auto.dropna()


# This question involves the use of multiple linear regression on the Auto data set.
# (a) Produce a scatterplot matrix which includes all of the variables in the data set.

# uncomment if plot needed
# sns.pairplot(auto)
# plt.show()

# (b) Compute the matrix of correlations between the variables using the function cor().
# You will need to exclude the name variable, cor() which is qualitative.
pd.set_option('display.max_rows', None, "display.max_columns", None)
print(auto.corr())

# (c) Use the lm() function to perform a multiple linear regression
# with mpg as the response and all other variables except name as
# the predictors. Use the summary() function to print the results.
# Comment on the output. For instance:


formula = 'mpg ~ ' + " + ".join(auto.columns[1:-1])
print(formula)

results = smf.ols(formula, data=auto).fit()
print(results.summary())
# i. Is there a relationship between the predictors and the response?
# There is a clear relationship between predictor and response. F-stat is very high

# ii. Which predictors appear to have a statistically significant relationship to the response?
# displacement, weight, year, origin are statistically significant

# iii. What does the coefficient for the year variable suggest?
# Its positive, so the higher the year the more the mpg

# (d) Use the plot() function to produce diagnostic plots of the linear
# regression fit. Comment on any problems you see with the fit.
# Do the residual plots suggest any unusually large outliers?
# Does the leverage plot identify any observations with unusually high leverage?

results_influence = OLSInfluence(results)
# looks very similar to previous problem
fig, ax = plt.subplots(2, 2, figsize=(12,10))
ax[0, 0].scatter(results.fittedvalues, results.resid)
ax[0, 0].set_ylabel("Raw Residuals")
ax[1, 0].scatter(results.fittedvalues, results_influence.resid_studentized_external)
ax[1, 0].set_ylabel("Studentized Residual")
sm.graphics.qqplot(results.resid / np.sqrt((results.resid ** 2).sum() / 390), line='45', ax=ax[0, 1])
ax[1, 1].scatter(results_influence.resid_studentized_external ** 2, results_influence.influence);
plt.show()

# Most residuals fall within 3 standard deviations and the qqplot looks relatively good until the right tail where a few observations are above 3 standard deviations indicating outliers.
fig, ax = plt.subplots(figsize=(8,6))
fig = plot_leverage_resid2(results, ax = ax)
plt.show()

# (e) Use the * and : symbols to fit linear regression models with interaction effects.
# Do any interactions appear to be statistically significant?

interactions_formula =  " + ".join([comb[0] + " * " + comb[1] for comb in combinations(auto.columns[1:-1], 2)])
formula = 'mpg ~ ' + " + ".join(auto.columns[1:-1])
formula += ' + ' + interactions_formula

print('* and : symbols')
results = smf.ols(formula, data=auto).fit()
print(results.summary())

# After adding all possible (7c2 = 21) interaction combination effects to the model only one of them is significant at the .01 level. Acceleration * origin

# (f) Try a few different transformations of the variables, such as
# log(X),√X, X^2. Comment on your findings.

# add displacement squared to model
formula += ' + np.power(displacement, 2)'
results = smf.ols(formula, data=auto).fit()
print(results.summary())

# lots of multicolinearity going on here
results = smf.ols('mpg ~ displacement + origin + np.power(displacement, 2)', data=auto).fit()
print(results.summary())

# sqrt of horsepower has higher r-squared than horsepower by itself
results = smf.ols('mpg ~ np.sqrt(horsepower)', data=auto).fit()
print(results.summary())

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,8))
# ax1.scatter(auto['horsepower'], auto['mpg'])
# ax1.set_title("Horsepower vs MPG")
# ax2.scatter(np.log(np.log(auto['horsepower'])), auto['mpg'])
# ax2.set_title("Log(Log(Horsepower)) vs MPG");
# plt.show()



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,8))
ax1.scatter(auto['horsepower'], auto['mpg'])
ax1.set_title("Horsepower vs MPG")
ax2.scatter(np.log(auto['horsepower']), auto['mpg'])
ax2.set_title("Log(Horsepower) vs MPG");
plt.show()

print('results for log horsepower')
# R-squared increases a bit more with log-log-horsepower
results = smf.ols('mpg ~ np.log((horsepower))', data=auto).fit()
print(results.summary())


