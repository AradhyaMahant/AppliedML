import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier 
from sklearn.datasets import make_classification
#%%
X, y = make_classification(n_samples=1000, n_features=4,
                            n_informative=2, n_redundant=0,  
                            random_state=0, shuffle=False)


clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)
print(clf.predict([[0, 0, 0, 0]]))
print(f"accuracy score is {clf.score(X,y)*100}%")
#%%
Xi, yi = make_classification(n_samples=1000, n_features=4,
                            n_informative=2, n_redundant=2, 
                            random_state=1, shuffle=False)

rforest = RandomForestClassifier(max_depth=2, random_state=0)
rforest.fit(Xi, yi)
print(rforest.predict([[0, 0, 0, 0]]))
print(f"accuracy score is {rforest.score(X,y)*100}%")

#%%

Xc, yc= make_classification(n_samples=1000, n_features=4,
                            n_informative=2, n_redundant=0,  
                            random_state=0, shuffle=False)


cl = RandomForestClassifier(n_estimators=200,max_depth=8,
                             random_state=3)
cl.fit(Xc, yc)
print(cl.predict([[0, 0, 0, 0]]))
print(f"accuracy score is {cl.score(Xc,yc)*100}%")

#%%

Xd, yd= make_classification(n_samples=1000, n_features=4,
                            n_informative=2, n_redundant=0,  
                            random_state=0, shuffle=False)


cld = RandomForestClassifier(n_estimators=200,max_depth=8,
                            min_samples_split = 5, max_features=3,
                             random_state=3)
cld.fit(Xd, yd)
print(cld.predict([[0, 0, 0, 0]]))
print(f"accuracy score is {cld.score(Xd,yd)*100}%")
#%%
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

max_features = ['auto', 'sqrt']

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

min_samples_split = [2, 5, 10]

min_samples_leaf = [1, 2, 4]

bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

print(random_grid)

from sklearn.model_selection import RandomizedSearchCV

rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf,
                               param_distributions = random_grid,
                               n_iter = 100, cv = 3, verbose=2,
                               random_state=42, n_jobs = -1)

rf_random.fit(Xi, yi)

