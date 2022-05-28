from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

#create data
X,y = make_regression(n_samples = 100, n_features = 5 , noise = 5, random_state = 5)

print("The shape of X is ",X.shape)
print("\n The shape of y is  ",y.shape)

plt.plot(y)
#%%

plt.figure()
plt.scatter(X[:,0], y, c = 'r')
plt.scatter(X[:,1], y, c = 'g')
plt.scatter(X[:,2], y, c = 'b')
plt.scatter(X[:,3], y, c = 'black')
plt.scatter(X[:,4], y, c = 'yellow')

plt.xlabel('X Value',fontsize = 16)
plt.ylabel('y Value',fontsize = 16)
plt.title("Dataset",fontsize = 20)
plt.show()
#%% 
#ploynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 2)
X = poly.fit_transform(X,y)
#%%
#train test split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 20)

#%%RidgeCV

from sklearn.linear_model import RidgeCV
model = RidgeCV(cv=5)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
score = model.score(X_test,y_test)
print("CV Score is",score)

#%%
temp=abs(score)
from sklearn import metrics
print("Mean Squared Logarithmic Error Loss is",metrics.mean_squared_log_error(abs(y_test),abs(y_pred)))
print("Mean Squared Error is",metrics.mean_squared_error(y_test,y_pred,squared = False))
print("Coefficient of determination regression score function:-\n",metrics.r2_score(y_test, y_pred))

plt.figure()
plt.scatter(X_test[:,0], y_pred)
plt.scatter(X_test[:,1], y_pred)
plt.scatter(X_test[:,2], y_pred)
plt.scatter(X_test[:,3], y_pred)
plt.scatter(X_test[:,4], y_pred)
plt.xlabel('X Value',fontsize = 16)
plt.ylabel('y Value',fontsize = 16)
plt.title("Polynomial Model",fontsize = 20)
plt.show()

#%%

