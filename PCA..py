import sklearn
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



# creating dataset
X , Y = sklearn.datasets.make_regression(n_samples=100,n_informative=3,n_features=10)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=1)

#LinearRegression Object
lr = LinearRegression()
lr.fit(X_train, y_train)

#y Prediction
y_pred = lr.predict(X_test)


# R2 score
score = metrics.r2_score(y_test, y_pred)
print("R2 score is :",score)


#PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca.fit(X)
X1 = pca.transform(X)



# Splitting dataset into training and testing set 
X_train, X_test, y_train, y_test = train_test_split(X1, Y, test_size=0.25, random_state=1)

#LinearRegression Object
lr = LinearRegression()
lr.fit(X_train, y_train)

# y Prediction
y_pred = lr.predict(X_test)


# R2 score
score = metrics.r2_score(y_test, y_pred)

print("R2 score with pca is :",score)

