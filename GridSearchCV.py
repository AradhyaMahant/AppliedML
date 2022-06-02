#%%
import pandas as pd 
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#%%
X,Y = make_regression(n_samples=1000,n_informative=3,noise =5,n_features = 1)
plt.scatter(X, Y, color='red')
plt.xlabel("X")
plt.ylabel("y")
plt.show()
#%%
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#%%
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg = reg.fit(X_train , y_train)
#%%

from sklearn.model_selection import GridSearchCV
tuned_parameters=[{'fit_intercept': ['True'], 'normalize': ['True']},
    {'fit_intercept': ['false'], 'normalize': ['True']}]
#%%
score =  'r2'
model = GridSearchCV(LinearRegression(),tuned_parameters,scoring=score)
model.fit(X_train,y_train)
print(model.best_params_)
model.cv_results_

#%%