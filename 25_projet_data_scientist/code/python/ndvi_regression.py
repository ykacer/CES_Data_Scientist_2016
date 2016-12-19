#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author : Youcef KACER <youcef.kacer@gmail.com>
# License: BSD 3 clause

from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

from sklearn.metrics import r2_score
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import ShuffleSplit
from sklearn.cross_validation import StratifiedKFold
from sklearn import grid_search
from sklearn.decomposition import PCA

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

from keras.layers import Dense, Dropout, Activation, Flatten

import os
import re
import codecs

print("Formatting data")

data = pd.read_csv('France/ndvi_features.csv',encoding='utf-8',na_values='NaN')
data.dropna(how='any',inplace=True)
data = data[data.SURFACE != 0]
variables = [v for v in data.columns if v.isdigit()]

if u'PMUN13' in data.columns:
	data = data[data.PMUN13 !=0]
	p = data['PMUN13'].as_matrix()
elif u'PMUN14' in data.columns:
	data = data[data.PMUN14 !=0]
	p = data['PMUN14'].as_matrix()
elif u'PMUN15' in data.columns:
	data = data[data.PMUN15 !=0]
	p = data['PMUN15'].as_matrix()
elif u'PMUN16' in data.columns:
	data = data[data.PMUN16 !=0]
	p = data['PMUN16'].as_matrix()

s = data['SURFACE'].as_matrix(); 
y = p/s
yl = np.log(1+y)

X = data[variables].as_matrix(); 

# reduction
scaler = StandardScaler()  
scaler.fit(X)  
Xsc = scaler.transform(X)  
joblib.dump(scaler,u'model_regression/Scaler_classification.pkl')

# projection
n_components = 608;
pca = PCA(n_components=n_components,random_state=0);
Xpca = pca.fit_transform(Xsc);
joblib.dump(pca,u'model_regression/PCA_regression.pkl')
fpca = codecs.open(u'model_regression/PCA_regression.txt',u'w',u'utf8')
fpca.write(u'PCA explained variance : '+str(pca.explained_variance_ratio_.sum()*100)+u'%\n')
fpca.close()

print(u'*** PCA explained variance : '+str(pca.explained_variance_ratio_.sum()*100)+u'%\n')

print("Classification bench")
cv = ShuffleSplit(y.size,n_iter=5,test_size=0.30,random_state=0)
results = [];
verbose = 2

print("* Linear Regression")
cl = linear_model.LinearRegression()
param_grid = {'fit_intercept':[False]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(Xpca,y)
ypred = grid.best_estimator_.predict(Xpca);
score = r2_score(y,ypred)
results.append(['Linear Regression',grid.grid_scores_,grid.scorer_,grid.best_score_,score,grid.best_params_,grid.get_params(),grid.best_estimator_,""])

print("* Ridge Regression")
cl = linear_model.Ridge()
param_grid = {'fit_intercept':[True,False],'alpha':[0.001,0.01,0.1,1.0,10.0,100.0,1000.0]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(Xpca,y)
ypred = grid.best_estimator_.predict(Xpca);
score = r2_score(y,ypred)
results.append(['Ridge Regression',grid.grid_scores_,grid.scorer_,grid.best_score_,score,grid.best_params_,grid.get_params(),grid.best_estimator_,""])

print("* Lasso Regression")
cl = linear_model.Lasso()
param_grid = {'fit_intercept':[True,False],'alpha':[0.01,0.1,1.0,10.0,100.0]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(Xpca,y)
ypred = grid.best_estimator_.predict(Xpca);
score = r2_score(y,ypred)
results.append(['Lasso Regression',grid.grid_scores_,grid.scorer_,grid.best_score_,score,grid.best_params_,grid.get_params(),grid.best_estimator_,""])

print("* ElasticNet Regression")
cl = linear_model.ElasticNet()
param_grid = {'fit_intercept':[True,False],'alpha':[0.01,0.1,1.0,10.0,100.0],'l1_ratio':[0.25,0.5,0.75]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(Xpca,y)
ypred = grid.best_estimator_.predict(Xpca);
score = r2_score(y,ypred)
results.append(['ElasticNet Regression',grid.grid_scores_,grid.scorer_,grid.best_score_,score,grid.best_params_,grid.get_params(),grid.best_estimator_,""])

print("* Ransac Regression")
base_estimator = linear_model.LinearRegression(fit_intercept=True)
cl = linear_model.RANSACRegressor(base_estimator=base_estimator,min_samples=0.8,loss='squared_loss')
cl.fit(Xpca,y)
ypred = cl.predict(Xpca);
score = r2_score(y,ypred)
best_score_ = cl.score(X[cl.inlier_mask_,:],y[cl.inlier_mask_])
results.append(['Ransac Regression',[],[],best_score_,score,[],None,""])

print("* Support Vector Regression")
cl = LinearSVR(loss='squared_epsilon_insensitive',dual=False,verbose=False,random_state=0)
param_grid = {'C':[0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0],'epsilon':[0,0.1,0.2,0.5,1,2,4]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(Xpca,y)
ypred = grid.best_estimator_.predict(Xpca);
score = r2_score(y,ypred)
info = ""
#info = "percentage of support vectors : "+1.0*len(grid.best_estimator_.support_)/y.size+"%\n"
results.append(['Support Vector Regression',grid.grid_scores_,grid.scorer_,grid.best_score_,score,grid.best_params_,grid.get_params(),grid.best_estimator_,info])

print("* Random Forest Regressor")
cl = RandomForestRegressor(max_depth=13,min_samples_split=10,min_samples_leaf=10,max_features=17,random_state=0)
param_grid = {'n_estimators':[160]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(Xpca,y)
ypred = grid.best_estimator_.predict(Xpca);
score = r2_score(y,ypred)
results.append(['Random Forest Regression',grid.grid_scores_,grid.scorer_,grid.best_score_,score,grid.best_params_,grid.get_params(),grid.best_estimator_,""])

print("* Gradient Boosting Regressor")
cl = GradientBoostingRegressor(learning_rate=0.5,n_estimators=40,min_samples_split=200,min_samples_leaf=110,max_depth=8,subsample=0.8,random_state=0)
param_grid = {'max_features':range(15,40,5)}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(Xpca,y)
ypred = grid.best_estimator_.predict(Xpca);
score = r2_score(y,ypred)
results.append(['Gradient Boosting Regression',grid.grid_scores_,grid.scorer_,grid.best_score_,score,grid.best_params_,grid.get_params(),grid.best_estimator_,""])

print("* Multi Layer Perceptron Regressor")
cl = MLPRegressor(hidden_layer_sizes=(1000,),alpha=0.001,solver='sgd',learning_rate='adaptive',tol=0.0001,early_stopping=True,verbose=True)
param_grid = {'learning_rate_init':[0.000001]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(Xpca,y)
ypred = grid.best_estimator_.predict(Xpca);
score = r2_score(y,ypred)
results.append(['Multi Layer Perceptron Regression',grid.grid_scores_,grid.scorer_,grid.best_score_,score,grid.best_params_,grid.get_params(),grid.best_estimator_,""])

cl = MLPRegressor(hidden_layer_sizes=(1000,1000,600))
param_grid = {'learning_rate_init':[0.001]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(Xpca,y)
ypred = grid.best_estimator_.predict(Xpca);
score = r2_score(y,ypred)
results.append(['Multi Layer Perceptron Regression',grid.grid_scores_,grid.scorer_,grid.best_score_,score,grid.best_params_,grid.get_params(),grid.best_estimator_,""])

cl = MLPRegressor(hidden_layer_sizes=(1000),solver='lbfgs',tol=0.0001,verbose=True)
param_grid = {'learning_rate_init':[0.001]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(Xpca,y)
ypred = grid.best_estimator_.predict(Xpca);
score = r2_score(y,ypred)
results.append(['Multi Layer Perceptron Regression',grid.grid_scores_,grid.scorer_,grid.best_score_,score,grid.best_params_,grid.get_params(),grid.best_estimator_,""])

print("* Neural Network Regression Keras")
def make_nn(optimizer='adam'):
    model = Sequential()
    model.add(Dense(output_dim=1024,input_dim=n_components,init='uniform',activation='relu'))
    model.add(Dense(512, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='relu'))
    model.compile(loss='mean_absolute_error', optimizer=optimizer, metrics=['mean_absolute_error'])
    return model

cl = KerasRegressor(make_nn, nb_epoch=200)
#optimizers = [Nadam(),Adam(),Adagrad(),SGD(lr=0.001, decay=1e-3, momentum=0.9, nesterov=True),RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)]
#optimizers = [RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)]
optimizers = [SGD(lr=0.005, decay=1e-3, momentum=0.9, nesterov=True)]
param_grid = {'batch_size':[100],'optimizer':optimizers}
#param_grid = {'batch_size':[100]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(Xpca,y)
ypred = grid.best_estimator_.predict(Xpca);
score = r2_score(y,ypred)
results.append(['TensorFlow Neural Network Regression',grid.grid_scores_,grid.scorer_,grid.best_score,score,grid.best_params_,grid.get_params(),grid.best_estimator_,""])

print("* Convolutional Neural Network Regression (Keras)")
def make_cnn(loss='mean_absolute_error',optimizer='adam'):
    model = Sequential()
    model.add(Convolution1D(1,19,subsample_length=19,border_mode='same',input_shape=(n_components,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Convolution1D(32, 3, border_mode='same',input_shape=(1,32)))
    model.add(Activation('relu'))
    model.add(Convolution1D(32, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Dropout(0.25))
    model.add(Convolution1D(64, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution1D(64, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('relu'))
    model.compile(loss=loss, optimizer=optimizer, metrics=['mean_absolute_error'])
    return model

cl = KerasRegressor(make_cnn, nb_epoch=200)
optimizers = [SGD(lr=0.001, decay=1e-5, momentum=0.9, nesterov=True)]
param_grid = {'batch_size':[200],'optimizer':optimizers}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(Xpca.reshape(Xpca.shape + (1,)),y)
ypred = grid.best_estimator_.predict(Xpca.reshape(Xpca.shape + (1,)));
score = r2_score(y,ypred);
results.append(['TensorFlow Convolutional Neural Network Regression', grid.grid_scores_,grid.scorer_,score,score, grid.best_params_, grid.get_params(),grid.best_estimator_,""])

try:
	os.mkdir('model_regression')
except:
	pass

for res in results:
    name = re.sub(u' ',u'_',res[0])
    model = res[7];
    folder_model = u'model_regression/'+name+'/'
    try:
        os.mkdir(folder_model)
    except:
        pass
    try:
        joblib.dump(model,folder_model+name+u'.pkl')
    except:
         model.model.save(folder_model+name+u'.h5')
    f = codecs.open(folder_model+name+u'_report.txt','w','utf8')
    f.write(res[-1]+u'\n\n')
    f.write(u'score cv : '+str(res[3])+u'\n\n')
    f.write(u'error cv : '+str((1-res[3]))+u'\n\n')
    f.write(u'score total : '+str(res[4])+u'\n\n')
    f.write(u'error total : '+str((1-res[4]))+u'\n\n')
    f.write(u'grid scores : \n')
    for gs in res[1]:
        f.write(u'\t'+str(gs)+u'\n')
    f.write(u'\n')
    f.write(u'best params : \n')
    f.write(str(res[5])+u'\n\n')
    f.write(u'best estimator : \n')
    f.write(str(res[6]))
    f.write(u'\n')
