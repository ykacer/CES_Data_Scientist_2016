#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author : Youcef KACER <youcef.kacer@gmail.com>
# License: BSD 3 clause

from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import ShuffleSplit
from sklearn.cross_validation import StratifiedKFold
from sklearn import grid_search

from sklearn.preprocessing import StandardScaler  
from sklearn import linear_model
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.externals import joblib

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier,KerasRegressor
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD,Adagrad,RMSprop,Adam,Nadam
from keras.layers.convolutional import Convolution1D
from keras.layers.pooling import MaxPooling1D

import os
import re
print("Formatting data")

data = pd.read_csv('France/ndvi_features.csv',encoding='utf-8',na_values='NaN')
data.dropna(how='any',inplace=True)
data = data[data.SURFACE != 0]
variables = [v for v in data.columns if v.isdigit()]

if u'PMUN13' in data.columns:
	p = data['PMUN13'].as_matrix()
elif u'PMUN14' in data.columns:
	p = data['PMUN14'].as_matrix()
elif u'PMUN15' in data.columns:
	p = data['PMUN15'].as_matrix()
elif u'PMUN16' in data.columns:
	p = data['PMUN16'].as_matrix()

s = data['SURFACE'].as_matrix(); 
y = p/s
yl = np.log(1+y)

X = data[variables].as_matrix(); 
scaler = StandardScaler()  
scaler.fit(X)  
Xsc = scaler.transform(X)  

print("Classification bench")
cv = ShuffleSplit(y.size,n_iter=5,test_size=0.3)
results = [];
verbose = 2

print("* Linear Regression")
cl = linear_model.LinearRegression()
param_grid = {'fit_intercept':[True,False]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(X,y)
results.append(['Linear Regression',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,grid.get_params(),grid.best_estimator_,""])

print("* Ridge Regression")
cl = linear_model.Ridge()
param_grid = {'fit_intercept':[True,False],'alpha':[0.001,0.01,0.1,1.0,10.0,100.0,1000.0]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(X,y)
results.append(['Ridge Regression',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,grid.get_params(),grid.best_estimator_,""])

print("* Lasso Regression")
cl = linear_model.Lasso()
param_grid = {'fit_intercept':[True,False],'alpha':[0.01,0.1,1.0,10.0,100.0]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(X,y)
results.append(['Lasso Regression',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,grid.get_params(),grid.best_estimator_,""])

print("* ElasticNet Regression")
cl = linear_model.ElasticNet()
param_grid = {'fit_intercept':[True,False],'alpha':[0.01,0.1,1.0,10.0,100.0],'l1_ratio':[0.25,0.5,0.75]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(X,y)
results.append(['ElasticNet Regression',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,grid.get_params(),grid.best_estimator_,""])

print("* Ransac Regression")
base_estimator = linear_model.LinearRegression(fit_intercept=True)
cl = linear_model.RANSACRegressor(base_estimator=base_estimator,min_samples=0.8,loss='squared_loss')
cl.fit(X,y)
best_score_ = cl.score(X[cl.inlier_mask_,:],y[cl.inlier_mask_])
results.append(['Ransac Regression',[],[],best_score_,[],None,""])

print("* Support Vector Regression")
cl = LinearSVR(loss='squared_epsilon_insensitive',dual=False,verbose=False,random_state=0)
param_grid = {'C':[0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0],'epsilon':[0,0.1,0.2,0.5,1,2,4]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(X,y)
info = ""
#info = "percentage of support vectors : "+1.0*len(grid.best_estimator_.support_)/y.size+"%\n"
results.append(['Support Vector Regression',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,grid.get_params(),grid.best_estimator_,info])

print("* Random Forest Regressor")
cl = RandomForestRegressor(max_depth=13,min_samples_split=10,min_samples_leaf=10,max_features=17,random_state=0)
param_grid = {'n_estimators':[160]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(X,y)
results.append(['Random Forest Regression',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,grid.get_params(),grid.best_estimator_,""])

print("* Gradient Boosting Regressor")
cl = GradientBoostingRegressor(learning_rate=0.5,n_estimators=40,min_samples_split=200,min_samples_leaf=110,max_depth=8,subsample=0.8,random_state=0)
param_grid = {'max_features':range(15,40,5)}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(Xsc,yl)
results.append(['Gradient Boosting Regression',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,grid.get_params(),grid.best_estimator_,""])

print("* Multi Layer Perceptron Regressor")
cl = MLPRegressor(hidden_layer_sizes=(2048,),alpha=0.001,solver='sgd',learning_rate='adaptive',tol=0.0001,early_stopping=True,verbose=True)
param_grid = {'learning_rate_init':[0.00001]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(Xsc,yl)
results.append(['Multi Layer Perceptron Regression',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,grid.get_params(),grid.best_estimator_,""])

cl = MLPRegressor(hidden_layer_sizes=(2000,500),alpha=0.0001,solver='sgd',power_t=0.2,learning_rate='invscaling',tol=0.0001,early_stopping=True,verbose=True)
param_grid = {'learning_rate_init':[0.000005]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(Xsc,yl)
results.append(['Multi Layer Perceptron Regression',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,grid.get_params(),grid.best_estimator_,""])

cl = MLPRegressor(hidden_layer_sizes=(1500),solver='lbfgs',tol=0.0001,verbose=True)
param_grid = {'learning_rate_init':[0.001]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(Xsc,yl)
results.append(['Multi Layer Perceptron Regression',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,grid.get_params(),grid.best_estimator_,""])

print("* Neural Network Regression (Keras)")
def make_nn(optimizer='adam'):
    model = Sequential()
    model.add(Dense(output_dim=1024,input_dim=512,init='uniform',activation='relu'))
    model.add(Dense(512, init='uniform', activation='relu'))
    model.add(Dense(1, init='normal', activation='softmax'))
    #opt = SGD(lr=0.01, decay=1e-1, momentum=0.9, nesterov=True)
    #opt = Adagrad()
    #opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #opt = RMSprop()
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error'])
    return model

cl = KerasRegressor(make_nn, nb_epoch=100)
optimizers = [SGD(lr=0.000000001, clipnorm=1.)]
param_grid = {'batch_size':[10000],'optimizer':optimizers}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(Xsc,y)
results.append(['Neural Network Regression (Keras)',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,grid.get_params(),grid.best_estimator_,""])

print("* Convolutional Neural Network Regression (Keras)")
def make_cnn(optimizer='adam'):
    model = Sequential()
    
    model.add(Convolution1D(32,9,subsample_length=4,border_mode='same',activation='relu',input_shape=(1,512)))
    model.add(MaxPooling1D(pool_length=2, stride=None, border_mode='valid'))
    
    model.add(Convolution1D(32,3,border_mode='same',activation='relu'))
    model.add(Convolution1D(32,3,border_mode='same',activation='relu'))
    model.add(MaxPooling1D(pool_length=2, stride=None, border_mode='valid'))
    
    model.add(Convolution1D(64,3,border_mode='same',activation='relu'))
    model.add(Convolution1D(64,3,border_mode='same',activation='relu'))
    model.add(MaxPooling1D(pool_length=2, stride=None, border_mode='valid'))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(1))
    return model

cl = KerasRegressor(make_cnn, nb_epoch=100)
optimizers = [SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)]
param_grid = {'batch_size':[100],'optimizer':optimizers}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(Xsc,y)
results.append(['Convolutional Neural Network Regression (Keras)',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,grid.get_params(),grid.best_estimator_,""])

try:
	os.mkdir('model_regression')
except:
	pass

for res in results:
    name = re.sub(res[0],u' ',u'_');
    model = res[6];
    joblib.dump(model,u'model_regression/'+name+u'.pkl')
