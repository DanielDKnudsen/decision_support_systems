# imports
import pandas as pd
import Methods as meth
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from random import randrange

# 10. This question should be answered using the Weekly data set, which
# is part of the ISLR package. This data is similar in nature to the
# Smarket data from this chapter’s lab, except that it contains 1, 089
# weekly returns for 21 years, from the beginning of 1990 to the end of
# 2010.

Weekly = meth.ReadCSV("Weekly.csv")
print(Weekly.head())

# (a) Produce some numerical and graphical summaries of the Weekly
# data. Do there appear to be any patterns?

print(Weekly.corr())

today = Weekly['Today']
today_perc = (100 + today) / 100
today_perc.cumprod().plot()
plt.show()

Weekly['Volume'].plot()
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot('Direction', 'Lag1', data=Weekly)
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot('Direction', 'Lag3', data=Weekly)
plt.show()

# (b) Use the full data set to perform a logistic regression with
# Direction as the response and the five lag variables plus Volume
# as predictors. Use the summary function to print the results. Do
# any of the predictors appear to be statistically significant? If so,
# which ones?

# y = Weekly['Direction']
# X = meth.DataLimiterCol(Weekly, 1, 7)
# print(X)
# # print(y)

# convert "Up" to value so model can be fitted
Weekly['Direction'] = np.where(Weekly['Direction'] == 'Up', 1, 0)

print(Weekly.dtypes)
results = sm.logit('Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume', data=Weekly).fit()
print(results.summary())

# Lag
results = sm.logit('Direction ~ Lag2', data=Weekly).fit()
print(results.summary())

# (c) Compute the confusion matrix and overall fraction of correct
# predictions. Explain what the confusion matrix is telling you
# about the types of mistakes made by logistic regression.

predictions = np.where(results.predict(Weekly) > .5, 1, 0)
print(confusion_matrix(Weekly['Direction'], predictions))

# (d) Now fit the logistic regression model using a training data period
# from 1990 to 2008, with Lag2 as the only predictor. Compute the
# confusion matrix and the overall fraction of correct predictions
# for the held out data (that is, the data from 2009 and 2010).

# split data set for period 1990 - 2008 and 2008 and onwards
year08 = Weekly.iloc[:985]
year10 = Weekly.iloc[985:]


results08 = sm.logit('Direction ~ Lag2', data=year08).fit()
print(results08.summary())

predictions = np.where(results.predict(year10) > .5, 1, 0)
print("Confusion Matrix for data held out")
print(confusion_matrix(year10['Direction'], predictions))

# (e) Repeat (d) using LDA.
# split into test and train data

year_bool = Weekly['Year'] < 2009
print(year_bool)
Weekly['ones'] = 1
X_train = Weekly[year_bool][['ones', 'Lag2']].values
X_test = Weekly[~year_bool][['ones', 'Lag2']].values
y_train = Weekly[year_bool]['Direction'].values
y_test = Weekly[~year_bool]['Direction'].values

LDA = LinearDiscriminantAnalysis()
LDA.fit(X_train, y_train)

print("Confusion matrix for LDA")
print(confusion_matrix(y_test, LDA.predict(X_test)))

# (f) Repeat (d) using QDA.

QDA = QuadraticDiscriminantAnalysis()
QDA.fit(X_train, y_train)
print("Confusion Matrix for QDA")
print(confusion_matrix(y_test, QDA.predict(X_test)))

# (g) Repeat (d) using KNN with K = 1.

KNN = KNeighborsClassifier(n_neighbors=1)
KNN.fit(X_train, y_train)
print("Confusion matrix for KNN")
print(confusion_matrix(y_test, KNN.predict(X_test)))

# (h) Which of these methods appears to provide the best results on this data?

# logistic regression and LDA work the best


# (i) Experiment with different combinations of predictors, including
# possible transformations and interactions, for each of the
# methods. Report the variables, method, and associated confusion
# matrix that appears to provide the best results on the held
# out data. Note that you should also experiment with values for
# K in the KNN classifier.

Variables = ['Lag1', 'Lag3', 'Lag5', 'Volume']

year_bool = Weekly['Year'] < 2009
print(year_bool)
Weekly['ones'] = 1
y_train = Weekly[year_bool]['Direction'].values
y_test = Weekly[~year_bool]['Direction'].values


Varlist= []
for col in Weekly.columns:
    Varlist.append(col);

Randval = randrange(0, len(Variables), 1)
# print("This is Randval = ", Randval)
print(Varlist[Randval])

for k in Variables:
    print("\n")
    if Varlist[Randval] == k:
        Randval += 1
    print("Predictors used:", k,"and", Varlist[Randval])
    X_train = Weekly[year_bool][['ones', k, Varlist[Randval]]].values
    X_test = Weekly[~year_bool][['ones', k, Varlist[Randval]]].values


    LDA = LinearDiscriminantAnalysis()
    LDA.fit(X_train, y_train)
    print("Confusion matrix for LDA")
    print(confusion_matrix(y_test, LDA.predict(X_test)))


    QDA = QuadraticDiscriminantAnalysis()
    QDA.fit(X_train, y_train)
    print("Confusion Matrix for QDA")
    print(confusion_matrix(y_test, QDA.predict(X_test)))

    for x in (1, 2, 5):
        KNN = KNeighborsClassifier(n_neighbors=x)
        KNN.fit(X_train, y_train)
        print("Confusion matrix for KNN = ", x)
        print(confusion_matrix(y_test, KNN.predict(X_test)))


print("Volume squared as transformation")
results = sm.logit('Direction ~ np.power(Volume, 2)', data=Weekly).fit()
print(results.summary())


print("Lag5 squared as transformation")
results = sm.logit('Direction ~ np.power(Lag5, 2)', data=Weekly).fit()
print(results.summary())