import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import plot_leverage_resid2
import matplotlib.pyplot as plt

CarSeats = pd.read_csv("Carseats.csv")
CarSeats = CarSeats.loc[:, ~CarSeats.columns.str.contains('^Unnamed')]
# pd.set_option('display.max_rows', None, "display.max_columns", None)
print(CarSeats.head())

formula = 'Sales ~ Price + Urban + US'
print(formula)

results = smf.ols(formula, data=CarSeats).fit()
print(results.summary())

# (b) Provide an interpretation of each coefficient in the model.
# Be careful—some of the variables in the model are qualitative!

# Only US and Price are statistically significant in our model. There is no difference whether someone is living in an urban area or not.
# Living in the US adds 1.2 to Sales up from 13 for outside of US. For every 1 unit increase in Price a corresponding .05 decrease in sales is seen.

# (c) Write out the model in equation form, being careful to handle
# the qualitative variables properly.

# In US: Sales = 14.323 - 0.55*Price
# Not in US: Sales = 13.04 - 0.055*Price + 0.022*urban + 1.2 US

# (d) For which of the predictors can you reject the null hypothesis
# H0 : βj = 0?

# For the predictors US and Price the null hypothesis can be rejected


# (e) On the basis of your response to the previous question, fit a
# smaller model that only uses the predictors for which there is
# evidence of association with the outcome.

formula = 'Sales ~ Price + US'

print('on the basis')
results = smf.ols(formula, data=CarSeats).fit()
print(results.summary())

# (f) How well do the models in (a) and (e) fit the data?
# Since urban is nearly completely random, there is almost no difference in the two models above. R-squared is low so lots of variance remains in the model

# (g) Using the model from (e), obtain 95% confidence intervals for the coefficient(s).
print(results.conf_int(alpha=0.05))


# (h) Is there evidence of outliers or high leverage observations in the model from (e)?

plt.scatter(results.fittedvalues, results.resid)
plt.show()
# Doesn't appear to be outliers

fig, ax = plt.subplots(figsize=(8,6))
fig = plot_leverage_resid2(results, ax = ax)
plt.show()


# a few high leverage points above .025
