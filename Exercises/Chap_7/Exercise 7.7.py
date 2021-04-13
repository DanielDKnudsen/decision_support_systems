import Methods as meth
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.gam.api import GLMGam, BSplines


# The Wage data set contains a number of other features not explored in this chapter,
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

plt.figure(figsize=(8, 6))
sns.boxplot('wage', 'maritl', data=Wage)
plt.show()



plt.figure(figsize=(8, 6))
sns.boxplot('wage', 'jobclass', data=Wage)
plt.show()

# It appears a married couple makes more money on average than other groups.
# It also appears that Informational jobs are higher-wage than Industrial jobs on average.

print("coeffecients for linear regression of Wage data set")
print(LR.coef_)
print("Intercepts")
print(LR.intercept_)

results_orig = sm.OLS(y, X).fit()
print(results_orig.summary())


X = pd.get_dummies(Wage['maritl'] + ' ' + Wage['jobclass'])
y = Wage['wage']

# results = sm.OLS(y, X).fit()
# print(results.summary())

print("GLM")
gam_bs = GLMGam.from_formula('wage ~ maritl + jobclass', data=Wage)
res_bs = gam_bs.fit()
print(res_bs.summary())