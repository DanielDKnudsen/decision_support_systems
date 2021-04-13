import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from sklearn.metrics import explained_variance_score, max_error, r2_score, mean_squared_error, mean_absolute_error

def create_interval(ols_result, interval_type, alpha, x_values, conf_x):
    if interval_type == 'confidence':
        add_one = 0
    elif interval_type == 'prediction':
        add_one = 1
    else:
        print("Choose interval_type as confidence or prediction")
        return
    n = len(x_values)
    t_value = stats.t.ppf(1 - alpha / 2, df = n - 2)
    sy = np.sqrt((ols_result.resid ** 2).sum() / (n - 2))
    numerator = (conf_x - x_values.mean()) ** 2
    denominator = ((x_values - x_values.mean()) ** 2).sum()
    interval = t_value * sy * np.sqrt(add_one + 1 / n + numerator / denominator)
    prediction = ols_result.params[0] + ols_result.params[1] * conf_x
    return (prediction - interval, prediction + interval)

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')

def ReadCSV(fileName):
    returnFile = pd.read_csv(fileName)
    returnFile = returnFile.loc[:, ~returnFile.columns.str.contains('^Unnamed')]
    return returnFile

def DataLimiterCol(Dataset,Start, stop):
    return Dataset.iloc[:, Start:stop]

def DataLimiterrow(Dataset,Start, stop):
    return Dataset.iloc[Start:stop]

def evaluate_predictions(y_pred, y):
    result = "Explained variance: {}".format(explained_variance_score(y, y_pred))
    result += "\nMean Squared Error: {}".format(mean_squared_error(y, y_pred))
    result += "\nMax Error: {}".format(max_error(y, y_pred))
    result += "\nR2 score: {}".format(r2_score(y, y_pred))
    result += "\nMean Absolute Error: {}".format(mean_absolute_error(y, y_pred))
    return result
