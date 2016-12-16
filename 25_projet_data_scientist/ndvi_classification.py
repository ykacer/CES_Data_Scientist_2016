#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author : Youcef KACER <youcef.kacer@gmail.com>
# License: BSD 3 clause

from __future__ import print_function
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, label_binarize

from sklearn.metrics import accuracy_score,roc_curve,auc
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import ShuffleSplit
from sklearn.cross_validation import StratifiedKFold
from sklearn import grid_search
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler,label_binarize
from sklearn.decomposition import PCA

from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from xgboost.sklearn import XGBClassifier

from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import AllKNN

from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

from imblearn.combine import SMOTETomek
from imblearn.combine import SMOTEENN

from imblearn.ensemble import EasyEnsemble
from imblearn.ensemble import BalanceCascade

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD,Adagrad,RMSprop,Adam,Nadam
from keras.layers.convolutional import Convolution1D
from keras.layers.pooling import MaxPooling1D


from scipy import interp

import matplotlib.pyplot as plt

from itertools import cycle
import re
import os
import codecs

print("*** Formatting data")

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
data = data[data.SURFACE != 0]
variables = [v for v in data.columns if v.isdigit()]
if 'PMUN13' in data.columns:
    data = data[data.PMUN13 != 0]
    population = data['PMUN13'].as_matrix()
elif 'PMUN14' in data.columns:
    data = data[data.PMUN14 != 0]
    population = data['PMUN14'].as_matrix()
elif 'PMUN15' in data.columns:
    data = data[data.PMUN15 != 0]
    population = data['PMUN15'].as_matrix()
elif 'PMUN16' in data.columns:
    data = data[data.PMUN16 != 0]
    population = data['PMUN16'].as_matrix()

surface = data['SURFACE'].as_matrix()
densite = population/surface;

# reduction
X = data[variables].as_matrix(); 
scaler = StandardScaler()
scaler.fit(X)
joblib.dump(scaler,u'model_classification/Scaler_classification.pkl')
Xsc = scaler.transform(X)

# projection
n_components = 608;
pca = PCA(n_components=n_components);
Xpca = pca.fit_transform(Xsc);
joblib.dump(pca,u'model_classification/PCA_classification.pkl')
fpca = codecs.open(u'model_classification/PCA_classification.txt',u'w',u'utf8')
fpca.write(u'PCA explained variance : '+str(pca.explained_variance_ratio_.sum()*100)+u'%\n')
fpca.close()

print(u'*** PCA explained variance : '+str(pca.explained_variance_ratio_.sum()))

# density categorization
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

n_cl = nc+1

print('*** Classe Distribution :')
for i in np.arange(nc+1):
    print("categorie "+str(i)+": "+str((y==i).sum())+" samples")

#ROS = RandomOverSampler(ratio=0.5)
#Xo,yo = ROS.fit_sample(Xpca,y)

Xo = Xpca;
yo = y;
for i in np.arange(nc):
    SMO = SMOTE(ratio=0.5,random_state=0);
    Xo,yo = SMO.fit_sample(Xo,yo)

print('*** Classe Distribution (after oversampling):')
for i in np.arange(nc+1):
    print("categorie "+str(i)+": "+str((yo==i).sum())+" samples")

print("*** Classification bench")
cv = StratifiedKFold(y,n_folds=5,random_state=0) # cross-validation set
cvo = StratifiedKFold(yo,n_folds=5,random_state=1) # cross-validation set

results = [];
verbose = 5

def compute_mean_score(y,yp,nc):
    mean_scores_per_category = np.zeros(nc+1)
    for ic in np.arange(nc+1):
	ni = (y==ic).sum()
	pi = ((yp==ic) & (y==ic)).sum()
        if ni!=0:
            mean_scores_per_category[ic] = 100.0*pi/ni
        else:
            mean_scores_per_category[ic] = -1
    return mean_scores_per_category

print("* Support Vector Classification")
n_samples = X.shape[0]
classes = np.arange(nc+1).tolist()
C = 0.001
weights = 1.0*n_samples / (n_cl * np.bincount(y.astype(np.int64)))
class_weight = dict(zip(classes,weights))
class_weight[1] = 2.0*class_weight[1]
class_weight[2] = 2.0*class_weight[2]
class_weight[3] = 2.0*class_weight[3]
cl = LinearSVC(C=C,dual=False,random_state=0,verbose=False)
param_grid = {'penalty':['l2'],'class_weight':[class_weight]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(Xpca,y)
#info = "percentage of support vectors : "+1.0*len(grid.best_estimator_.support_)/y.size+"%\n"
info=''
info = info + np.array_str(metrics.confusion_matrix(y,grid.best_estimator_.predict(Xpca)))
info = info+u"\n\n"
mean_scores = compute_mean_score(y,grid.best_estimator_.predict(Xpca),nc)
for i in np.arange(nc+1):
    info = info+target_names[i]+': '+str(100-mean_scores[i])+"%\n"

info = info+u"\n\n"
info = info+u'mean error per class : '+str(100-mean_scores.mean())+"%\n\n"
results.append(['Support Vector Classification',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,grid.get_params(),grid.best_estimator_,info])


print("* Support Vector Classification-oversampling")
n_sampleso = Xo.shape[0]
classeso = np.arange(nc+1).tolist()
weightso = 1.0*n_sampleso / (n_cl * np.bincount(yo.astype(np.int64)))
class_weighto = dict(zip(classeso,weightso))
class_weighto[0] = 1.1*class_weighto[0]
class_weighto[1] = 1.6*class_weighto[1]

class_weighto1 = dict(zip(classeso,weightso))
class_weighto1[0] = 1.3*class_weighto1[0]
class_weighto1[1] = 1.6*class_weighto1[1]

class_weighto2 = dict(zip(classeso,weightso))
class_weighto2[0] = 1.6*class_weighto2[0]
class_weighto2[1] = 1.6*class_weighto2[1]

Co = 0.1
cl = LinearSVC(C=Co,dual=False,random_state=0,verbose=False)
param_grid = {'penalty':['l2'],'class_weight':[class_weighto1]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cvo,verbose=verbose)
grid.fit(Xo,yo)
info=''
info = info + np.array_str(metrics.confusion_matrix(yo,grid.best_estimator_.predict(Xo)))
info = info+u"\n\n"
mean_scores = compute_mean_score(yo,grid.best_estimator_.predict(Xo),nc)
for i in np.arange(nc+1):
    info = info+target_names[i]+': '+str(100-mean_scores[i])+"%\n"

info = info+u"\n\n"
info = info+u'mean error per class : '+str(100-mean_scores.mean())+"%\n\n"
info = info + np.array_str(metrics.confusion_matrix(y,grid.best_estimator_.predict(Xpca)))
info = info+u"\n\n"
mean_scores = compute_mean_score(y,grid.best_estimator_.predict(Xpca),nc)
for i in np.arange(nc+1):
    info = info+target_names[i]+': '+str(100-mean_scores[i])+"%\n"

info = info+u"\n\n"
info = info+u'mean error per class : '+str(100-mean_scores.mean())+"%\n\n"
results.append(['Support Vector Classification-oversampling',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,grid.get_params(),grid.best_estimator_,info])


print("* Random Forest Classification")
cl = RandomForestClassifier(n_estimators=40,max_depth=15,min_samples_split=20,min_samples_leaf=20,max_features=35,random_state=0)
param_grid = {'max_features':[15]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(Xpca,y)
info = np.array_str(metrics.confusion_matrix(y,grid.best_estimator_.predict(Xpca)))
info = info+u'\n\n'
mean_scores = compute_mean_score(y,grid.best_estimator_.predict(Xpca),nc)
for i in np.arange(nc+1):
    info = info+target_names[i]+': '+str(100-mean_scores[i])+'%\n'
info = info+u'\n\n'
info = info+u'mean error per class : '+str(100-mean_scores.mean())+'%\n\n'
results.append(['Random Forest Classification',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,grid.get_params(),grid.best_estimator_,info])

print("* Gradient Boosting Classification")
cl = GradientBoostingClassifier(learning_rate=0.45, n_estimators=40, min_samples_split=80,min_samples_leaf=20,max_depth=32,max_features=32, subsample=0.8, random_state=0)
param_grid = {'n_estimators':[40]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(Xpca,y)
info = np.array_str(metrics.confusion_matrix(y,grid.best_estimator_.predict(Xpca)))
info = info+u"\n\n"
mean_scores = compute_mean_score(y,grid.best_estimator_.predict(Xpca),nc)
for i in np.arange(nc+1):
    info = info+target_names[i]+': '+str(100-mean_scores[i])+u"%\n"

info = info+u"\n\n"
info = info+u'mean error per class : '+str(100-mean_scores.mean())+'%\n\n'
results.append(['Gradient Boosting Classification',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,grid.get_params(),grid.best_estimator_,info])

print("* Extreme Gradient Boosting Classification")
cl = XGBClassifier(learning_rate =0.1,n_estimators=1000,max_depth=5,min_child_weight=1,gamma=0,subsample=0.8,colsample_bytree=0.8, objective='binary:logistic',nthread=4, scale_pos_weight=1, seed=0,silent=False)
param_grid = {'learning_rate':[0.1]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(Xpca,y)
info = np.array_str(metrics.confusion_matrix(y,grid.best_estimator_.predict(Xpca)))
info = info+u"\n\n";
mean_scores = compute_mean_score(y,grid.best_estimator_.predict(Xpca),nc)
for i in np.arange(nc+1):
    info = info+target_names[i]+": "+str(mean_scores[i])+"%\n";

info = info+u"\n\n"
info = info+u'mean error per class : '+str(mean_scores.mean())+"%\n\n"
results.append(['Extreme Gradient Boosting Classification',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,grid.get_params(),grid.best_estimator_,info])

print("* Neural Network Classification-oversampling")
cl = MLPClassifier(activation='logistic', batch_size='auto',learning_rate='constant', power_t=0.5, max_iter=200,random_state=0,tol=0.0001,momentum=0.9,nesterovs_momentum=True,early_stopping=False,verbose=True)
param_grid = {'hidden_layer_sizes':[(2*n_components,)],'solver':['sgd'],'alpha':[0.001],'learning_rate_init':[0.001]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cvo,verbose=verbose)
grid.fit(Xo,yo)
info = np.array_str(metrics.confusion_matrix(yo,grid.best_estimator_.predict(Xo)))
info = info+u'\n\n'
mean_scores = compute_mean_score(yo,grid.best_estimator_.predict(Xo),nc)
for i in np.arange(nc+1):
    info = info+target_names[i]+': '+str(100-mean_scores[i])+'%\n'

info = info+u'\n\n'
info = info+u'mean error per class : '+str(100-mean_scores.mean())+'%\n\n'
info = info+np.array_str(metrics.confusion_matrix(y,grid.best_estimator_.predict(Xpca)))
info = info+u'\n\n'
mean_scores = compute_mean_score(y,grid.best_estimator_.predict(Xpca),nc)
for i in np.arange(nc+1):
    info = info+target_names[i]+': '+str(100-mean_scores[i])+'%\n'

info = info+u'\n\n'
info = info+u'mean error per class : '+str(100-mean_scores.mean())+'%\n\n'
results.append(['Neural Network Classification-oversampling',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,grid.get_params(),grid.best_estimator_,info])

print("* TensorFlow Neural Network Classification-oversampling")
def make_model():
    model = Sequential()
    model.add(Dense(output_dim=1204,input_dim=n_components,init='uniform',activation='relu'))
    model.add(Dense(512, init='uniform', activation='relu'))
    model.add(Dense(nc+1, init='uniform', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

cl = KerasClassifier(make_model, nb_epoch=10)
param_grid = {'batch_size':[100]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cvo,verbose=verbose)
grid.fit(Xo,to_categorical(yo,n_cl))
info = np.array_str(metrics.confusion_matrix(yo,grid.best_estimator_.predict(Xo)))
info = info+u'\n\n'
mean_scores = compute_mean_score(yo,grid.best_estimator_.predict(Xo),nc)
for i in np.arange(nc+1):
    info = info+target_names[i]+': '+str(100-mean_scores[i])+'%\n'
info = info+u'\n\n'
info = info+u'mean error per class : '+str(100-mean_scores.mean())+'%\n\n'
info = info+np.array_str(metrics.confusion_matrix(y,grid.best_estimator_.predict(Xpca)))
info = info+u'\n\n'
mean_scores = compute_mean_score(y,grid.best_estimator_.predict(Xpca),nc)
for i in np.arange(nc+1):
    info = info+target_names[i]+': '+str(100-mean_scores[i])+'%\n'
info = info+u'\n\n'
info = info+u'mean error per class : '+str(100-mean_scores.mean())+'%\n\n'
results.append(['Neural Network Classification TensorFlow-oversampling',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,grid.get_params(),grid.best_estimator_,info])

print("* TensorFlow Convolutional Neural Network Classification")
def make_cnn(loss='categorical_crossentropy',optimizer='adam'):
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
    model.add(Dense(n_cl))
    model.add(Activation('softmax'))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model

cl = KerasClassifier(make_cnn, nb_epoch=20)
n_samples = X.shape[0]
classes = np.arange(nc+1).tolist()
weights = 1.0*n_samples / (n_cl * np.bincount(y.astype(np.int64)))
weights = np.log(1+weights)
class_weight = dict(zip(classes,weights))
class_weight[2] = 2.0*class_weight[2] 
class_weight[3] = 2.0*class_weight[3] 
class_weight[4] = 2.0*class_weight[4] 
class_weight[5] = 2.0*class_weight[5] 
#optimizers = [Nadam(),Adam(),Adagrad(),SGD(lr=0.001, decay=1e-3, momentum=0.9, nesterov=True),RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)]
#optimizers = [SGD(lr=0.001, decay=1e-3, momentum=0.9, nesterov=True)]
#optimizers = [Adam(lr=0.005)]
#param_grid = {'batch_size':[100],'optimizer':optimizers}
param_grid = {'batch_size':[100]}
grid = grid_search.GridSearchCV(cl,param_grid,fit_params={'class_weight':class_weight},cv=cv,verbose=verbose)

grid.fit(Xpca.reshape(Xpca.shape + (1,)),to_categorical(y,n_cl))

info = np.array_str(metrics.confusion_matrix(y,grid.best_estimator_.predict(Xpca.reshape(Xpca.shape + (1,)))))
info = info+u'\n\n'
mean_scores = compute_mean_score(y,grid.best_estimator_.predict(Xpca.reshape(Xpca.shape + (1,))),nc)
for i in np.arange(nc+1):
    info = info+target_names[i]+': '+str(100-mean_scores[i])+'%\n'

info = info+u'\n\n'
info = info+u'mean error per class : '+str(100-mean_scores.mean())+'%\n\n'
results.append(['Convolutional Neural Network Classification TensorFlow',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,grid.get_params(),grid.best_estimator_,info])


print("* Nearest Neighboors-oversampling")
cl = KNeighborsClassifier()
param_grid = {'n_neighbors':[5],'weights':['distance'],'algorithm':['auto'],'leaf_size':[10],'p':[3]}
grid = grid_search.GridSearchCV(cl,param_grid,cv=cv,verbose=verbose)
grid.fit(Xo,yo)
ypred = grid.best_estimator_.predict(Xo)
info = np.array_str(metrics.confusion_matrix(yo,ypred))
info = info+u"\n\n"
mean_scores = compute_mean_score(yo,ypred,nc)
for i in np.arange(nc+1):
    info = info+target_names[i]+': '+str(100-mean_scores[i])+"%\n"

info = info+u"\n\n"
info = info+u'mean error per class : '+str(100-mean_scores.mean())+"%\n\n"
info = info+u"\n\n"
ypred = grid.best_estimator_.predict(Xpca)
info = info+np.array_str(metrics.confusion_matrix(y,ypred))
info = info+u"\n\n"
mean_scores = compute_mean_score(y,ypred,nc)
for i in np.arange(nc+1):
    info = info+target_names[i]+': '+str(100-mean_scores[i])+"%\n"

info = info+u"\n\n"
info = info+u'mean error per class : '+str(100-mean_scores.mean())+"%\n\n"
results.append(['Nearest Neighboors Classification',grid.grid_scores_,grid.scorer_,grid.best_score_,grid.best_params_,grid.get_params(),grid.best_estimator_,info])

colors = [[49,140,231],
 [255,255,153],
 [255,232,139],
 [246,209,125],
 [241,185,111],
 [236,162,97],
 [232,139,83],
 [227,116,70],
 [227,116,70],
 [218,70,42],
 [213,46,28],
 [209,23,14],
 [204,0,0],
 [170,0,0],
 [136,0,0],
 [102,0,0],
 [92,0,20],
 [82,0,41],
 [71,0,61],
 [61,0,82],
 [51,0,102]]

try:
    os.mkdir('model_classification')
except:
    pass

for res in [results[-1]]:
    name = re.sub(u' ','_',res[0])
    model = res[6];
    folder_model = u'model_classification/'+name+'/'
    try:
        os.mkdir(folder_model)
    except:
        pass
    try:
        joblib.dump(model,folder_model+name+u'.pkl')
    except:
        try:
            model.model.save(folder_model+name+u'.h5')
        except:
            with open(folder_model+name+u'.pkl','wb') as fid:
                cPickle.dump(model,fid)
    f = codecs.open(folder_model+name+u'_report.txt','w','utf8')
    f.write(res[-1]+u'\n\n')
    f.write(u'score cv : '+str(res[3]*100)+u'%\n\n')
    f.write(u'error cv : '+str((1-res[3])*100)+u'%\n\n')
    f.write(u'grid scores : \n')
    for gs in res[1]:
        f.write(u'\t'+str(gs)+u'\n')
    f.write(u'\n')
    f.write(u'best params : \n')
    f.write(str(res[4]))
    f.write(u'\n')
    try:
        yhat = model.predict(Xpca)
    except:
        yhat = model.predict(Xpca.reshape(Xpca.shape + (1,)))
    f.write(metrics.classification_report(y, yhat, labels=np.arange(nc+1).tolist(), target_names=target_names,digits=3))
    f.close()
    # Compute ROC curve and ROC area for each class
    if 'decision_functon' in dir(model):
        yp = model.decision_function(Xpca)
    elif 'predict_proba' in dir(model):
	try:
        	yp = model.predict_proba(Xpca)
	except:
		yp =  model.predict_proba(Xpca.reshape(Xpca.shape + (1,)))
    else:
        yp = label_binarize(model.predict(Xpca), classes=np.arange(nc+1))
    yb = label_binarize(y, classes=np.arange(nc+1))
    n_classes = yb.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(yb[:, i], yp[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i]) 
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(yb.ravel(), yp.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))   
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # Plot mean ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='blue', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='green', linewidth=4)
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves')
    plt.legend(loc="lower right",fontsize = 'medium')
    plt.savefig(folder_model+name+u'_roc_mean.png',dpi=1000)
    #plt.show()
    # Plot all ROC curves
    plt.figure()
    for i in range(n_classes):
            ci = np.asarray(colors[1:][int(1.0*i/(nc+1)*len(colors))])/255.0
            plt.plot(fpr[i], tpr[i], color=ci, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves')
    plt.legend(loc="lower right",fontsize = 'medium')
    plt.savefig(folder_model+name+u'_roc.png',dpi=1000)
    #plt.show()
