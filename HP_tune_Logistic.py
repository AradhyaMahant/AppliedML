from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X,Y = make_classification(n_samples=1000,n_features = 10, n_classes = 2)
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)


from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()

reg.fit(X_train,y_train)

from sklearn.model_selection import GridSearchCV
tuned_parameters=[{'fit_intercept': ['True'], 'normalize': ['True']},
    {'fit_intercept': ['false'], 'normalize': ['True']}]

scores =  ['r2','mse']
for score in scores:
    model = GridSearchCV(LogisticRegression(),tuned_parameters,scoring=score)
    model.fit(X_train,y_train)
        
    print(model.best_params_)
    model.cv_results_
