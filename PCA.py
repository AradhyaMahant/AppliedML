# -*- coding: utf-8 -*-import matplotlib.pyplot as plt

"""
Created on Fri Sep 17 09:59:52 2021

@author: Sumit
"""
from sklearn.datasets import make_classification
x,y=make_classification(n_samples=100,n_features=10,n_informative=3)
x1=x
y1=y
"""
from sklearn import decomposition
pca=decomposition.PCA(n_components=3)
pca.fit(x)
x=pca.transform(x)
"""
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=50)

from sklearn.linear_model import LogisticRegression
Lr = LogisticRegression()
Lr.fit(x_train,y_train)

y_predict = Lr.predict(x_test)
print(y_predict)

from sklearn.metrics import confusion_matrix,accuracy_score

Confusion_Matrix = confusion_matrix(y_test,y_predict)
print("Confusion_Matrix: ",Confusion_Matrix)
accuracy = accuracy_score(y_test,y_predict)
accuracy


from sklearn import decomposition
pca=decomposition.PCA(n_components=1)
pca.fit(x1)
x1=pca.transform(x1)
x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,test_size=0.2,random_state=50)
Lr1 = LogisticRegression()
Lr1.fit(x1_train,y1_train)
y1_predict = Lr1.predict(x1_test)
print(y1_predict)
Confusion_Matrix1 = confusion_matrix(y1_test,y1_predict)
print("Confusion_Matrix: ",Confusion_Matrix1)
accuracy1 = accuracy_score(y1_test,y1_predict)
accuracy1