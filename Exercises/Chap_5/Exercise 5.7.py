import Methods
from sklearn.linear_model import LogisticRegression
import numpy as np
import statsmodels.formula.api as sm
from statsmodels.api import Logit
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
weekly = Methods.ReadCSV('Weekly.csv')


# In Sections 5.3.2 and 5.3.3, we saw that the cv.glm() function can be
# used in order to compute the LOOCV test error estimate. Alternatively,
# one could compute those quantities using just the glm() and
# predict.glm() functions, and a for loop. You will now take this approach
# in order to compute the LOOCV error for a simple logistic
# regression model on the Weekly data set. Recall that in the context
# of classification problems, the LOOCV error is given in (5.4).

# (a) Fit a logistic regressionmodel that predicts Direction using Lag1
# and Lag2.
weekly['Direction'] = np.where(weekly['Direction'] == 'Up', 1, 0)
# weekly['Direction_Up'] = (weekly['Direction'] == 'Up').astype(int)
print(weekly.head())


X = weekly[['Lag1', 'Lag2']]
y = weekly['Direction']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# print("Predictions for Direction using Lag1 and Lag2")
# print((model.predict(X_test)))

model = LogisticRegression(C=100000, tol=.0000001)
model.fit(X, y)

print("Predictions")
print(model.intercept_, model.coef_)

# accuracy
print("Accuracy")
print((model.predict(X) == y))
print("Accuracy Score")
print((model.predict(X) == y).mean())
# (b) Fit a logistic regressionmodel that predicts Direction using Lag1
# and Lag2 using all but the first observation.


model.fit(X.iloc[1:], y.iloc[1:])
print("Predictions without first observations")
print(model.intercept_, model.coef_)
print("Accuracy without first observation")
print((model.predict(X) == y).mean())

# model is different but nearly identical
# model = sm.logit('Direction ~ Lag1 + Lag2', data=weekly[1:]).fit()
# print(model.summary())
# print(model.params)
#
# print(Logit.predict(params=model))

# print(model.intercept_, model.coef_, (model.predict(X) == y).mean())


# (c) Use the model from (b) to predict the direction of the first observation.
# You can do this by predicting that the first observation
# will go up if P(Direction="Up"|Lag1, Lag2) > 0.5. Was this observation
# correctly classified?


print("Predicted:")
print(model.predict([X.iloc[0]]))
print("Actual:")
print(y[0])
#prediction was wrong


# (d) Write a for loop from i = 1 to i = n, where n is the number of
# observations in the data set, that performs each of the following
# steps:

# i. Fit a logistic regression model using all but the ith observation
# to predict Direction using Lag1 and Lag2.

# ii. Compute the posterior probability of the market moving up
# for the ith observation

# iii. Use the posterior probability for the ith observation in order
# to predict whether or not the market moves up

# iv. Determine whether or not an error was made in predicting
# the direction for the ith observation. If an error was made,
# then indicate this as a 1, and otherwise indicate it as a 0

errors = np.zeros(len(X))
for i in range(len(X)):
    leave_out  = ~X.index.isin([i])
    model.fit(X[leave_out], y[leave_out])
    if model.predict([X.iloc[i]]) != y[i]:
        errors[i] = 1

# (e) Take the average of the n numbers obtained in (d)iv in order to
# obtain the LOOCV estimate for the test error. Comment on the results.

print("Algorithm Test Error")
print(errors.mean())
