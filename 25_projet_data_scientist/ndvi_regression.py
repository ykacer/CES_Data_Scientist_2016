# Author : Youcef KACER <youcef.kacer@gmail.com>
# License: BSD 3 clause

from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import ShuffleSplit
from sklearn import grid_search

from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

print("Formatting data")

data = pd.read_csv('France/ndvi_features.csv')
variables = [v for v in data.columns if v not in ['name','population','surface','densite']]
y = data['population'].as_matrix()/data['surface'].as_matrix(); 
X = data[variables].as_matrix(); 

print("Classification bench")
cv = ShuffleSplit(y.size,n_iter=5,test_size=0.3) # cross-validation set
results = [];
verbose = 2

print("* Linear Regression")
cl = linear_model.LinearRegression()
param_grid = {'fit_intercept':[True,False]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(X,y)
results.append(['Linear Regression',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,""])

print("* Ridge Regression")
cl = linear_model.Ridge()
param_grid = {'fit_intercept':[True,False],'alpha':[0.001,0.01,0.1,1.0,10.0,100.0,1000.0]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(X,y)
results.append(['Ridge Regression',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,""])

print("* Lasso Regression")
cl = linear_model.Lasso()
param_grid = {'fit_intercept':[True,False],'alpha':[0.01,0.1,1.0,10.0,100.0]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(X,y)
results.append(['Lasso Regression',g1rid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,""])

print("* ElasticNet Regression")
cl = linear_model.ElasticNet()
param_grid = {'fit_intercept':[True,False],'alpha':[0.01,0.1,1.0,10.0,100.0],'l1_ratio':[0.25,0.5,0.75]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(X,y)
results.append(['ElasticNet Regression',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,""])

print("* Ransac Regression")
base_estimator = linear_model.LinearRegression(fit_intercept=True)
cl = linear_model.RANSACRegressor(base_estimator=base_estimator,min_samples=0.8,loss='squared_loss')
cl.fit(X,y)
best_score_ = cl.score(X[cl.inlier_mask_,:],y[cl.inlier_mask_])
results.append(['Ransac Regression',[],[],best_score_,[],None,""])

print("* Support Vector Regression")
cl = SVR(kernel='linear',verbose=True)
param_grid = {'C':[10.0]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(X,y)
info = "percentage of support vectors : "+1.0*len(grid.best_estimator_.support_)/y.size+"%\n"
results.append(['Support Vector Regression',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,info])

print("* Random Forest Regressor")
cl = RandomForestRegressor(max_depth=11,min_samples_split=40,min_samples_leaf=20,max_features=33,random_state=0)
param_grid = {'n_estimators':[60]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(X,y)
results.append(['Random Forest Regression',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,""])

print("* Gradient Boosting Regressor")
cl = GradientBoostingRegressor(n_estimators=400,max_depth=17,min_samples_split=30,min_samples_leaf=30,max_features=19,subsample=0.75,random_state=0)
param_grid = {'learning_rate':[0.01]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(X,y)
results.append(['Gradient Boosting Regression',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,""])

print("* Multi Layer Perceptron Regressor")
cl = MLPRegressor(hidden_layer_sizes=(1000),verbose=True)
param_grid = {'learning_rate_init':[0.0001]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(X,y)
results.append(['Multi Layer Perceptron Regression',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,""])

