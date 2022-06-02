from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

#create data
X,Y = make_classification(n_samples=1000,n_features = 5,n_classes = 2)

#visualizing data
print("Shape of X is: \n",X.shape)
print("Shape of Y is: \n",Y.shape,"\n\n")

#model building
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2,random_state = 1)


from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()

reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)


#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

print(cm)
