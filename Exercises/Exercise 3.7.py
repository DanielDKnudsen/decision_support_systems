#imports

import pandas as pd
from  sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
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
auto = pd.read_csv("D:\Repo\DSS\Exercises\Auto.csv")

auto.head()

X = auto.horsepower;
y = auto.mpg;

# sns.scatterplot(X,y)
# plt.show()


results = smf.ols('mpg ~ horsepower', data=auto).fit()
print(results.summary())


# Use the summary() function to print the results. Comment on the output.
# For example:

# i: Is there a relationship between the predictor and the response?
# ii: How strong is the relationship between the predictor and the response?
# iii: Is the relationship between the predictor and the response positive or negative?
# iv: What is the predicted mpg associated with a horsepower of 98? What are the associated 95% confidence and prediction intervals?


# (b) Plot the response and the predictor. Use the abline() function to display the least squares regression line.


# (c) Use the plot() function to produce diagnostic plots of the least squares regression fit. Comment on any problems you see with the fit.


