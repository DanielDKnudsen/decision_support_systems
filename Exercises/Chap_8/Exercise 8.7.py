# In the lab, we applied random forests to the Boston data using mtry=6  and using ntree=25 and ntree=500.
# Create a plot displaying the test error resulting from random forests on this data set for a more comprehensive
# range of values for mtry and ntree.
# You can model your plot after Figure 8.10. Describe the results obtained.
#%%
import pandas as pd
import numpy as np
import itertools
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import datetime
import os
import multiprocessing as mp
from sklearn.metrics import mean_squared_error

# %%
df = pd.read_csv('Boston.csv')
y = np.array(df['crim'])
df = df.drop('crim', axis=1).drop('Unnamed: 0', axis=1)
X = np.array(df[df.columns])
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# %%
n_models = list(range(1, 500, 25))
n_max_features = list(range(1, 14, 3))

trees = []
features = []
results = []

for i in itertools.product(n_models, n_max_features):
    n_model = i[0]
    n_max_feature = i[1]
    name = 'trees_' + str(n_model) + "_" + 'features_' + str(n_max_feature)
    clf = RandomForestRegressor(n_jobs=-1, n_estimators=n_model, max_features=n_max_feature, random_state=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'{mse=}')
    trees.append(n_model)
    features.append(n_max_feature)
    results.append(mse)

# %%
tree_data_points = [trees[i] for i in range(0, len(trees), 5)]
feat_1 = [results[i] for i in range(0, len(results), 5)]
feat_4 = [results[i] for i in range(1, len(results), 5)]
feat_7 = [results[i] for i in range(2, len(results), 5)]
feat_10 = [results[i] for i in range(3, len(results), 5)]
feat_13 = [results[i] for i in range(4, len(results), 5)]
# %%
import matplotlib.pyplot as plt
# plotting the line 1 points 
plt.plot(tree_data_points, feat_1, label = "1 feature")
plt.plot(tree_data_points, feat_4, label = "4 features")
plt.plot(tree_data_points, feat_7, label = "7 features")
plt.plot(tree_data_points, feat_10, label = "10 features")
plt.plot(tree_data_points, feat_13, label = "13 features")
plt.xlabel('Amount of trees in model')
# Set the y axis label of the current axis.
plt.ylabel('Mean Squared Error')
# Set a title of the current axes.
plt.title('Comparison of Random Forest Regressor varying amount of trees and max features')
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()
# %%
