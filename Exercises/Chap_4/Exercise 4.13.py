import Methods
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import random
# 13. Using the Boston data set, fit classification models in order to predict
# whether a given suburb has a crime rate above or below the median.
# Explore logistic regression, LDA, and KNN models using various subsets
# of the predictors. Describe your findings.

Boston = Methods.ReadCSV('Boston.csv');
Boston['crim01'] = np.where(Boston['crim'] > Boston['crim'].median(), 1, 0)
print(Boston)

VarList = []

for col in Boston.columns:
    VarList.append(col);

print(VarList)
VarList.remove('crim01')
VarList.remove('crim')


y = Boston['crim01']



#Classification on a subset of random predictors
for i in range (2, 6):
    Subsets = random.sample(VarList, i)

    print('Performing classification with ', Subsets, ' as a predictors')

    X_train, X_test, y_train, y_test = train_test_split(Boston[Subsets], y, random_state=1)

    LR = LogisticRegression(C=1)
    LR.fit(X_train, y_train)

    # LDA = LinearDiscriminantAnalysis()
    # LDA.fit(X_train, y_train)

    print("Accuracy Test score for LR")
    print(accuracy_score(y_test, LR.predict(X_test)), '\n')

    QDA = QuadraticDiscriminantAnalysis()
    QDA.fit(X_train, y_train)
    print("Accuracy Test score QDA")
    print(accuracy_score(y_test, QDA.predict(X_test)), '\n')

    for x in (1, 2, 5):
        KNN = KNeighborsClassifier(n_neighbors=x)
        KNN.fit(X_train, y_train)
        print("Accuracy Test score KNN=", x)
        print(accuracy_score(y_test, KNN.predict(X_test)), '\n')