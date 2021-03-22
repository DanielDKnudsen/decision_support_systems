import Methods as meth
import pandas as pd
from sklearn.linear_model import LinearRegression

# The Wage data set contains a number of other features not exploredin this chapter,
# such as marital status (maritl), job class (jobclass), and others.
# Explore the relationships between some of these other predictors and wage,
# and use non-linear fitting techniques in order to fit flexible models to the data.
# Create plots of the results obtained, and write a summary of your findings.

Wage = meth.ReadCSV("Wage.csv")
print(Wage[['maritl', 'jobclass']].head())

X = pd.get_dummies(Wage[['maritl', 'jobclass']], drop_first=False)
y = Wage['wage']

LR = LinearRegression(fit_intercept=True)
LR.fit(X,y)

print("coeffecients for linear regression of Wage data set")
print(LR.coef_)
print("Intercepts")
print(LR.intercept_)