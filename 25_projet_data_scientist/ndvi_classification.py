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
from sklearn import grid_search
from sklearn import metrics
from sklearn.externals import joblib

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

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
cv = ShuffleSplit(y.size,n_iter=5,test_size=0.3) # cross-validation set
results = [];
verbose = 2

print("* Support Vector Classification")
cl = SVC(kernel='linear',class_weight='balanced',verbose=True)
param_grid = {'C':[10.0]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(X,y)
info = "percentage of support vectors : "+1.0*len(grid.best_estimator_.support_)/y.size+"%\n"
info = info + np.array_str(metrics.confusion_matrix(grid.best_estimator_.predictX), y))
results.append(['Support Vector Classification',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,grid.get_params(),grid.best_estimator_,info])

print("* Random Forest Classification")
cl = RandomForestClassifier(max_depth=11,min_samples_split=40,min_samples_leaf=20,max_features=33,random_state=0)
param_grid = {'n_estimators':[60]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(X,y)
info = np.array_str(metrics.confusion_matrix(grid.best_estimator_.predict(X), y))
results.append(['Random Forest Classification',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,grid.get_params(),grid.best_estimator_,info])

print("* Gradient Boosting Classification")
cl = GradientBoostingClassifier(n_estimators=70, max_depth=21, min_samples_split=210, min_samples_leaf=60, max_features=15, subsample=0.8, random_state=10)
param_grid = {'learning_rate':[0.1]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(X,y)
info = np.array_str(metrics.confusion_matrix(grid.best_estimator_.predict(X), y))
results.append(['Gradient Boosting Classification',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,grid.get_params(),grid.best_estimator_,info])

print("* Neural Network Classification")
def make_model():
    model = Sequential()
    model.add(Dense(output_dim=1204,input_dim=512,init='uniform',activation='relu'))
    model.add(Dense(512, init='uniform', activation='relu'))
    model.add(Dense(6, init='uniform', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

cl = KerasClassifier(make_model, nb_epoch=200)
param_grid = {'batch_size':[100]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(X,to_categorical(y,6))
results.append(['Neural Network Classification',scores,best_score_,best_params_,""])

try:
    os.mkdir('model_classification')
except:
    pass

for res in results:
    name = re.sub(u' ','_',res[0])
    model = res[6];
    joblib.dump(model,u'model_classification/'+name+u'.pkl')
    f = codecs.open(u'model_classification/'+name+u'_report.txt','w','utf8')
    f.write(res[-1]+u'\n\n')
    f.write("error cv : "+str(1-res[3])+u'\n\n')
    f.write(metrics.classification_report(y, model.predict(X), labels=np.arange(nc+1).tolist(), target_names=target_names,digits=3))
    f.close()


	
