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

def write_submission(csv_file,id,loss):
    dtype = [('id','int32'), ('loss','float32')]
    values = np.zeros(loss.size,dtype=dtype)
    values['id'] = id
    values['loss'] = loss
    pd.DataFrame(values).to_csv(csv_file,sep=',',index=False)

print("Formatting data")

data = pd.read_csv('France/ndvi_features.csv')
variables = [v for v in data.columns if v not in ['name','population','surface','densite']]
densite = data['population'].as_matrix()/data['surface'].as_matrix(); 
X = data[variables]; 

y = -1*np.ones(X.shape[0])
for i,n in enumerate(densite):
    if n>13000:
        y[i] = 0
    elif n>5000:
        y[i] = 1
    elif n>2000:
        y[i] = 2
    elif n>500:
        y[i] = 3
    else:
        y[i] = 4

for i in np.arange(5):
    print("categorie "+str(i)+": "+str((y==i).sum())+" samples")

print("Classification bench")
cv = ShuffleSplit(y.size,n_iter=5,test_size=0.3) # cross-validation set
results = [];
verbose = 2

print("* Support Vector Classification")
cl = SVC(kernel='linear',verbose=True)
param_grid = {'C':[10.0]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(X,y)
info = "percentage of support vectors : "+1.0*len(grid.best_estimator_.support_)/y.size+"%\n"
results.append(['Support Vector Classification',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,info])

print("* Random Forest Classification")
cl = RandomForestClassifier(max_depth=11,min_samples_split=40,min_samples_leaf=20,max_features=33,random_state=0)
param_grid = {'n_estimators':[60]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(X,y)
results.append(['Random Forest Classification',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,""])

print("* Gradient Boosting Classification")
cl = GradientBoostingClassifier(n_estimators=400,max_depth=17,min_samples_split=30,min_samples_leaf=30,max_features=19,subsample=0.75,random_state=0)
param_grid = {'learning_rate':[0.01]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(X,y)
results.append(['Gradient Boosting Classification',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,""])
#
