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
from sklearn.ensemble import RandomForestRegressor

def write_submission(csv_file,id,loss):
    dtype = [('id','int32'), ('loss','float32')]
    values = np.zeros(loss.size,dtype=dtype)
    values['id'] = id
    values['loss'] = loss
    pd.DataFrame(values).to_csv(csv_file,sep=',',index=False)

print("Formatting data")

data = pd.read_csv('France/ndvi_features.csv')
variables = [v for v in data.columns if v not in ['name','population','surface','densite']]
y = data['population'].as_matrix()/data['surface'].as_matrix(); 
X = data[variables]; 

print("Classification bench")
cv = ShuffleSplit(y.size,n_iter=5,test_size=0.3) # cross-validation set
results = [];
verbose = 2

print("* Linear Regression")
cl = linear_model.LinearRegression()
param_grid = {'fit_intercept':[True,False]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(X,y)
#ytest = grid.best_estimator_.predict(X_test)
results.append(['Linear Regression',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,""])
#write_submission('sample_submission_linear_regression.csv',id_test,ytest)

print("* Ridge Regression")
cl = linear_model.Ridge()
param_grid = {'fit_intercept':[True,False],'alpha':[0.001,0.01,0.1,1.0,10.0,100.0,1000.0]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(X,y)
#ytest = grid.best_estimator_.predict(X_test)
results.append(['Ridge Regression',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,""])
#write_submission('sample_submission_ridge_regression.csv',id_test,ytest)

print("* Lasso Regression")
cl = linear_model.Lasso()
param_grid = {'fit_intercept':[True,False],'alpha':[0.01,0.1,1.0,10.0,100.0]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(X,y)
#ytest = grid.best_estimator_.predict(X_test)
results.append(['Lasso Regression',g1rid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,""])
#write_submission('sample_submission_lasso_regression.csv',id_test,ytest)

print("* ElasticNet Regression")
cl = linear_model.ElasticNet()
param_grid = {'fit_intercept':[True,False],'alpha':[0.01,0.1,1.0,10.0,100.0],'l1_ratio':[0.25,0.5,0.75]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(X,y)
#ytest = grid.best_estimator_.predict(X_test)
results.append(['ElasticNet Regression',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,""])
#write_submission('sample_submission_elasticnet_regression.csv',id_test,ytest)

print("* Ransac Regression")
base_estimator = linear_model.LinearRegression(fit_intercept=True)
cl = linear_model.RANSACRegressor(base_estimator=base_estimator,min_samples=0.8,loss='squared_loss')
cl.fit(X,y)
#ytest = cl.estimator_.predict(X_test)
best_score_ = cl.score(X[cl.inlier_mask_,:],y[cl.inlier_mask_])
results.append(['Ransac Regression',[],[],best_score_,[],None,""])
#write_submission('sample_submission_ransac_regression.csv',id_test,ytest)

print("* Support Vector Regression")
cl = SVR(kernel='linear',verbose=True)
param_grid = {'C':[10.0]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(X,y)
#ytest = grid.best_estimator_.predict(X_test)
info = "percentage of support vectors : "+1.0*len(grid.best_estimator_.support_)/y.size+"%\n"
results.append(['Support Vector Regression',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,info])
#write_submission('sample_submission_support_regression.csv',id_test,ytest)

print("* Random Forest Regressor")
cl = RandomForestRegressor(max_depth=11,min_samples_split=40,min_samples_leaf=20,max_features=33,random_state=0)
param_grid = {'n_estimators':60}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(X,y)
#ytest = grid.best_estimator_.predict(X_test)
results.append(['Random Forest Regression',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,""])
#write_submission('sample_submission_random_forest_regression.csv',id_test,ytest)
