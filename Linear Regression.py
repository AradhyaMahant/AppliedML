import pandas as pd 
import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

X,Y = make_regression(n_samples=10000,noise =10,n_features = 1)

#calculating mean of x and y
n= len(X)
mean_x =np.mean(X);
mean_y =np.mean(Y);


#calculating slope(m)
numerator = 0
denominator = 0

for i in range(n):
    numerator += (X[i]-mean_x)*(Y[i]-mean_y)
    denominator += (X[i]-mean_x)**2
    
m = numerator/denominator
c = mean_y - (m * mean_x)
print(m,c)


max_x = np.max(X) + 100
min_x = np.min(X) - 100

#calculating line values x and y
x = np.linspace(min_x,max_x, 1000)
y = c  + m * x 

#plotting line
plt.plot(x,y, color = 'red' , label = 'Regression Line')
#plotting scatter points
plt.scatter(X,Y, color= 'blue', label = 'Scatter plot')

plt.xlabel('Head Size(cm^3)')
plt.ylabel('Brain Weight(grams)')
plt.legend()
plt.show()


#calculating the best fit check( r squared method)
num = 0
dem = 0

for i in range(n):
    y_pred = (m*X[i]) + c
    num += (y_pred - mean_y)**2
    dem += (Y[i] - mean_y)**2
    
R =  (num/dem)
print(R)