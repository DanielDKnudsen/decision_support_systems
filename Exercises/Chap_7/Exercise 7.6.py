import Methods as meth
import matplotlib.pyplot as plt
from statsmodels.stats.api import anova_lm
import statsmodels.formula.api as smf
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn import model_selection
import pandas as pd
# 6. In this exercise, you will further analyze the Wage data set considered throughout this chapter.


# (a) Perform polynomial regression to predict wage using age.
# Use cross-validation to select the optimal degree d for the polynomial.
# What degree was chosen, and how does this compare to the results of hypothesis testing using ANOVA?
# Make a plot of the resulting polynomial fit to the data.

Wage = meth.ReadCSV('Wage.csv')
print(Wage.head)

y=Wage['wage']
X=Wage[['age']]

degrees = range(1,11)
LR=LinearRegression()
Scores = []
for degree in degrees:
    P_Feat = PolynomialFeatures(degree=degree,
                                             include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", P_Feat),
                         ("linear_regression", LR)])

    scores = model_selection.cross_val_score(pipeline,
                                            X, y, cv=10, scoring='neg_mean_squared_error')
    Scores.append(-np.mean(scores))


plt.plot(degrees,Scores)
plt.show()



mod1 = smf.ols('wage ~ age', data=Wage).fit()
mod2 = smf.ols('wage ~ age + np.power(age, 2)', data=Wage).fit()
mod3 = smf.ols('wage ~ age + np.power(age, 2) + np.power(age, 3)', data=Wage).fit()
mod4 = smf.ols('wage ~ age + np.power(age, 2) + np.power(age, 3) + np.power(age, 4)', data=Wage).fit()
mod5 = smf.ols('wage ~ age + np.power(age, 2) + np.power(age, 3) + np.power(age, 4) + np.power(age, 5)', data=Wage).fit()
print(anova_lm(mod1, mod2, mod3, mod4, mod5))
# Shows that degrees above 3 are not significant

P_feat = PolynomialFeatures(degree=3, include_bias=False)
LR=LinearRegression().fit(P_feat.fit_transform(X), y)


x = np.linspace(X.values.min(), X.values.max(), 1000)
plt.scatter(X, y)
plt.plot(x, LR.predict(P_feat.fit_transform(x.reshape(-1, 1))), c='r', lw=3);
plt.show()


# (b) Fit a step function to predict wage using age, and perform crossvalidation
# to choose the optimal number of cuts. Make a plot of the fit obtained.
cuts = range(1,41)
Scores = []
for cut in cuts:
    X_new = pd.get_dummies(pd.cut(X['age'], cut)).values

    linear_regression = LinearRegression(fit_intercept=False)

    scores = model_selection.cross_val_score(linear_regression, X_new, y, cv=10, scoring='neg_mean_squared_error')
    Scores.append(-np.mean(scores))

plt.plot(cuts, Scores);
plt.show()

X_new = pd.get_dummies(pd.cut(X['age'], 7)).values
linear_regression = LinearRegression(fit_intercept=False)
linear_regression.fit(X_new, y)
plt.scatter(X, y)
order = np.argsort(X['age'])
plt.plot(X['age'].values[order], linear_regression.predict(X_new[order]), c='r', lw=3);
plt.show()