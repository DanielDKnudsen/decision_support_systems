import pandas as pd
from sklearn.model_selection import train_test_split
from numpy import arange
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn import preprocessing
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn import model_selection
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import explained_variance_score, max_error, r2_score, mean_squared_error, mean_absolute_error

def evaluate_predictions(y_pred, y):
    result = "Explained variance: {}".format(explained_variance_score(y, y_pred))
    result += "\nMean Squared Error: {}".format(mean_squared_error(y, y_pred))
    result += "\nMax Error: {}".format(max_error(y, y_pred))
    result += "\nR2 score: {}".format(r2_score(y, y_pred))
    result += "\nMean Absolute Error: {}".format(mean_absolute_error(y, y_pred))
    return result


college = pd.read_csv('college.csv')
college['Private'].value_counts()

college['private_yes'] = (college['Private'] == 'Yes') * 1

X = college.iloc[:, 3:]
y = college['Apps']

x = X.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
X = pd.DataFrame(x_scaled)

y_array = y.values #returns a numpy array
y_scaled = min_max_scaler.fit_transform(y_array.reshape(-1, 1))
y = pd.DataFrame(y_scaled)


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=13)

## Linear model ##
print("Linear regression: ")
lr = LinearRegression()
lr.fit(X_train, y_train)

print("Results for Linear:")
print(evaluate_predictions(y_test, lr.predict(X_test)))

## RIDGE ##
print("Ridge: ")
model = Ridge()
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid
grid = dict()
grid['alpha'] = arange(0, 1, 0.01)
# define search
search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# perform the search
results = search.fit(X_train, y_train)
# summarize
print('MAE: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)

ridge_model = Ridge(alpha=0.28)
ridge_model.fit(X_train, y_train)

print("Results for Ridge:")
print(evaluate_predictions(y_test, ridge_model.predict(X_test)))

## LASSO ##

print("Lasso: ")
model = Lasso()
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define grid
grid = dict()
grid['alpha'] = arange(0.01, 1, 0.01)
# define search
search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# perform the search
results = search.fit(X_train, y_train)
# summarize
print('MAE: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)

lasso_model = Lasso(alpha=0.01)
result = lasso_model.fit(X_train, y_train)

print(result.coef_.tolist())

print("Results for Lasso:")
print(evaluate_predictions(y_test, lasso_model.predict(X_test)))

## PCR ##
print("PCR: ")
pca = PCA()
X_reduced_train = pca.fit_transform(X_train)

# 10-fold CV, with shuffle
n = len(X_reduced_train)
kf_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)

regr = LinearRegression()
mse = []

# Calculate MSE with only the intercept (no principal components in regression)
score = -1 * model_selection.cross_val_score(regr, np.ones((n, 1)), y_train.values.ravel(), cv=kf_10,
                                             scoring='neg_mean_squared_error').mean()
mse.append(score)

# Calculate MSE using CV for the 16 principle components, adding one component at the time.
for i in np.arange(1, 17):
    score = -1 * model_selection.cross_val_score(regr, X_reduced_train[:, :i], y_train.values.ravel(), cv=kf_10,
                                                 scoring='neg_mean_squared_error').mean()
    mse.append(score)

# Plot results
plt.plot(mse, '-v')
plt.xlabel('Number of principal components in regression')
plt.ylabel('MSE')
plt.title('PCR - Applications')
plt.xlim(xmin=-1)
plt.show()

X_reduced_test = pca.transform(X_test)[:,:7]

# Train regression model on training data
regr = LinearRegression()
regr.fit(X_reduced_train[:,:7], y_train)

print("Results for PCR:")
print(evaluate_predictions(y_test, regr.predict(X_reduced_test)))

## PLS ##
print("PLS: ")

n = len(X_train)

# 10-fold CV, with shuffle
kf_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)

mse = []

for i in np.arange(1, 17):
    pls = PLSRegression(n_components=i)
    score = model_selection.cross_val_score(pls, X_train, y_train, cv=kf_10, scoring='neg_mean_squared_error').mean()
    mse.append(-score)

# Plot results
plt.plot(np.arange(1, 17), np.array(mse), '-v')
plt.xlabel('Number of principal components in regression')
plt.ylabel('MSE')
plt.title('PLS - Applications')
plt.xlim(xmin=-1)
plt.show()

pls = PLSRegression(n_components=10)
pls.fit(X_train, y_train)

print("Results for PLS:")
print(evaluate_predictions(y_test, pls.predict(X_test)))

print("Done")