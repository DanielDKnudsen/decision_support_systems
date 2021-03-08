import numpy as np
import random
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import itertools

X = np.random.randint(5,size=(1000,20))
B = np.random.ranf(size=(20,1))
B[0] = 0
B[3] = 0
B[17] = 0
epsilon = np.random.ranf(size=(1000,1))
Y = X.dot(B) + epsilon

train_x = pd.DataFrame(X[:900,:])
train_y = pd.DataFrame(Y[:900,:])
test_x = pd.DataFrame(X[900:,:])
test_y = pd.DataFrame(Y[900:,:])


def fit_linear_reg(X,Y):
    #Fit linear regression model and return RSS and R squared values
    model_k = LinearRegression(fit_intercept = True)
    model_k.fit(X,Y)
    RSS = mean_squared_error(Y,model_k.predict(X)) * len(Y)
    R_squared = model_k.score(X,Y)
    return RSS, R_squared

RSS_list, R_squared_list, feature_list = [],[], []
numb_features = []


for p in range(1,len(train_x.columns)):

    for combo in itertools.combinations(train_x.columns,p):
        tmp_result = fit_linear_reg(train_x[list(combo)],train_y)  
        RSS_list.append(tmp_result[0])                  
        R_squared_list.append(tmp_result[1])
        feature_list.append(combo)
        numb_features.append(len(combo))  

df = pd.DataFrame({'numb_features': numb_features,'RSS': RSS_list, 'R_squared':R_squared_list,'features':feature_list})

print(df.shape)

print("test",test_x.shape)
print("train",train_x.shape)
