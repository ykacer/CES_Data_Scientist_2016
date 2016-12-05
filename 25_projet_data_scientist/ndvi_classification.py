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
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from xgboost.sklearn import XGBClassifier

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils.np_utils import to_categorical

import re
import os
import codecs

print("Formatting data")

categorization = [500,2000,5000,10000,13000]
nc = len(categorization)
target_names = []
target_names.append(u'0 - '+str(categorization[0])+u' habs/km²')
for i in np.arange(nc-1):
    c1 = str(categorization[i])
    c2 = str(categorization[i+1])
    target_names.append(c1+u' - '+c2+u' habs/km²')

target_names.append(u'> '+str(categorization[-1])+u' habs/km²')

data = pd.read_csv('France/ndvi_features.csv',dtype={'SURFACE':np.float64})
variables = [v for v in data.columns if v.isdigit()]
if 'PMUN13' in data.columns:
    population = data['PMUN13'].as_matrix()
elif 'PMUN14' in data.columns:
    population = data['PMUN14'].as_matrix()
elif 'PMUN15' in data.columns:
    population = data['PMUN15'].as_matrix()
elif 'PMUN16' in data.columns:
    population = data['PMUN16'].as_matrix()

df = df[df.SURFACE != 0]
surface = data['SURFACE'].as_matrix()
densite = population/surface;

X = data[variables].as_matrix(); 
y = -1*np.ones(X.shape[0])

categorization_r = list(reversed(categorization))
for i,n in enumerate(densite):
    if n>categorization_r[0]:
        y[i] = 5
    elif n>categorization_r[1]:
        y[i] = 4
    elif n>categorization_r[2]:
        y[i] = 3
    elif n>categorization_r[3]:
        y[i] = 2
    elif n>categorization_r[4]:
        y[i] = 1
    else:
        y[i] = 0

for i in np.arange(nc+1):
    print("categorie "+str(i)+": "+str((y==i).sum())+" samples")

print("Classification bench")
cv = StratifiedKFold(y,n_folds=5,random_state=0) # cross-validation set
results = [];
verbose = 5

print("* Support Vector Classification")
cl = SVC(kernel='linear',class_weight='balanced',random_state=0,verbose=True)
param_grid = {'C':[1.0]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(X,y)
info = "percentage of support vectors : "+1.0*len(grid.best_estimator_.support_)/y.size+"%\n"
info = info + np.array_str(metrics.confusion_matrix(grid.best_estimator_.predictX), y)
results.append(['Support Vector Classification',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,grid.get_params(),grid.best_estimator_,info])

print("* Random Forest Classification")
cl = RandomForestClassifier(n_estimators=40,max_depth=15,min_samples_split=20,min_samples_leaf=20,max_features=35,random_state=0)
param_grid = {'max_features':[15]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(X,y)
info = np.array_str(metrics.confusion_matrix(grid.best_estimator_.predict(X), y))
results.append(['Random Forest Classification',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,grid.get_params(),grid.best_estimator_,info])

print("* Gradient Boosting Classification")
cl = GradientBoostingClassifier(learning_rate=0.45, n_estimators=40, min_samples_split=80,min_samples_leaf=20,max_depth=32,max_features=32, subsample=0.8, random_state=0)
param_grid = {'n_estimators':[40]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(X,y)
info = np.array_str(metrics.confusion_matrix(grid.best_estimator_.predict(X), y))
results.append(['Gradient Boosting Classification',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,grid.get_params(),grid.best_estimator_,info])

print("* Extreme Gradient Boosting Classification")
cl = XGBClassifier(learning_rate =0.1,n_estimators=1000,max_depth=5,min_child_weight=1,gamma=0,subsample=0.8,colsample_bytree=0.8, objective='binary:logistic',nthread=4, scale_pos_weight=1, seed=0,silent=False)
param_grid = {'learning_rate':[0.1]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(X,y)
info = np.array_str(metrics.confusion_matrix(grid.best_estimator_.predict(X),y))
results.append(['Extreme Gradient Boosting Classification',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,grid.get_params(),grid.best_estimator_,info])

print("* Neural Network Classifier")
scaler = StandardScaler()  
scaler.fit(X)  
Xsc = scaler.transform(X)  
cl = MLPClassifier(activation='logistic', batch_size='auto',learning_rate='constant',learning_rate_init=0.0001, power_t=0.5, max_iter=50,random_state=0,tol=0.0001,momentum=0.9,nesterovs_momentum=True,early_stopping=False,verbose=True)
param_grid = {'hidden_layer_sizes':[(2048,)],'solver':['lbfgs'],'alpha':[0.001,0.01,0.1,1.0,10.,100.0]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(Xsc,y)
info = np.array_str(metrics.confusion_matrix(grid.best_estimator_.predict(X), y))
results.append(['Neural Network Classification',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,grid.get_params(),grid.best_estimator_,info])

print("* TensorFlow Neural Network Classification")
def make_model():
    model = Sequential()
    model.add(Dense(output_dim=1204,input_dim=512,init='uniform',activation='relu'))
    model.add(Dense(512, init='uniform', activation='relu'))
    model.add(Dense(nc+1, init='uniform', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

cl = KerasClassifier(make_model, nb_epoch=200)
param_grid = {'batch_size':[100]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(Xsc,to_categorical(y,nc+1))
info = np.array_str(metrics.confusion_matrix(grid.best_estimator_.predict(X),to_categorical(y,nc+1)))
results.append(['Neural Network Classification TensorFlow',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,grid.get_params(),grid.best_estimator_,info])

try:
    os.mkdir('model_classification')
except:
    pass

for res in [results[-1]]:
    name = re.sub(u' ','_',res[0])
    model = res[6];
    joblib.dump(model,u'model_classification/'+name+u'.pkl')
    f = codecs.open(u'model_classification/'+name+u'_report.txt','w','utf8')
    f.write(res[-1]+u'\n\n')
    f.write("error cv : "+str((1-res[3])*100)+u'%\n\n')
    f.write(metrics.classification_report(y, model.predict(X), labels=np.arange(nc+1).tolist(), target_names=target_names,digits=3))
    f.close()


	
