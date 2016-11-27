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

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils.np_utils import to_categorical

print("Formatting data")

data = pd.read_csv('France/ndvi_features.csv')
variables = [v for v in data.columns if v not in ['name','population','surface','densite']]
densite = data['population'].as_matrix()/data['surface'].as_matrix(); 
X = data[variables].as_matrix(); 

y = -1*np.ones(X.shape[0])
for i,n in enumerate(densite):
    if n>13000:
        y[i] = 5
    elif n>10000:
        y[i] = 4
    elif n>5000:
        y[i] = 3
    elif n>2000:
        y[i] = 2
    elif n>500:
        y[i] = 1
    else:
        y[i] = 0

for i in np.arange(5):
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
info = info + np.array_str(metrics.confusion_matrix(grid.best_estimator_.fit(X), y))
results.append(['Support Vector Classification',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,info])

print("* Random Forest Classification")
cl = RandomForestClassifier(max_depth=11,min_samples_split=40,min_samples_leaf=20,max_features=33,random_state=0)
param_grid = {'n_estimators':[60]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(X,y)
info = np.array_str(metrics.confusion_matrix(grid.best_estimator_.fit(X), y))
results.append(['Random Forest Classification',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,info])

print("* Gradient Boosting Classification")
cl = GradientBoostingClassifier(n_estimators=70, max_depth=21, min_samples_split=210, min_samples_leaf=60, max_features=15, subsample=0.8, random_state=10)
param_grid = {'learning_rate':[0.1]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(X,y)
info = np.array_str(metrics.confusion_matrix(grid.best_estimator_.fit(X), y))
results.append(['Gradient Boosting Classification',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,info])

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
#
