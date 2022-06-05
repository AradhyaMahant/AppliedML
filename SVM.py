import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

# Creating Data (We create 40 separable points)

X, y = make_blobs(n_samples=40, centers=2, random_state=6)

#Creating Model
clf = svm.SVC(kernel="linear", C=1000)
clf.fit(X, y)
#%%
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2, random_state=0)
print('Shape of X_train : ' , X_train.shape)
print('Shape of y_train : ' , y_train.shape)
print('Shape of X_test : ' , X_test.shape)
print('Shape of y_test : ' , y_test.shape)

# Compiling 
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
print('Score : ' , clf.score(X_test, y_test))

clf = svm.SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(clf, X, y, cv=5)
print(scores)
print("%0.2f accuracy " % (scores.mean()))

from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

model = SVC()

param_grid = {'C' : [0.1, 1, 10, 100, 1000],
           'gamma' : [1, 0.1, 0.01, 0.001, 0.0001],
           'kernel' : ['rbf']}

grid = GridSearchCV(SVC() , param_grid , refit = 'True', verbose = 3)

grid.fit(X,y)
print('\n\n')
print(grid.best_params_)
print(grid.best_estimator_)
