#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Author : Youcef KACER <youcef.kacer@gmail.com>
# License: BSD 3 clause

from __future__ import print_function
import pandas as pd
import numpy as np

from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler,label_binarize
from sklearn.metrics import accuracy_score,roc_curve,auc

from keras.models import load_model

from scipy import interp

import matplotlib.pyplot as plt

import sys
import os

categorization = [500,2000,5000,10000,13000]
nc = len(categorization)
target_names = []
target_names.append(u'0 - '+str(categorization[0])+u' habs/km²')
for i in np.arange(nc-1):
    c1 = str(categorization[i])
    c2 = str(categorization[i+1])
    target_names.append(c1+u' - '+c2+u' habs/km²')
target_names.append(u'> '+str(categorization[-1])+u' habs/km²')

file_test = sys.argv[1]
folder = os.path.dirname(file_test);
model = sys.argv[2]
data = pd.read_csv(file_test,encoding='utf-8',na_values='NaN')
data.dropna(how='any',inplace=True)
data = data[data.SURFACE != 0]
print(str(data.shape[0])+" cities")

variables = [v for v in data.columns if v.isdigit()]
if 'PMUN13' in data.columns:
    year = u'13'
    data = data[data.PMUN13 != 0]
    population = data['PMUN13'].as_matrix()
elif 'PMUN14' in data.columns:
    year = u'14'
    data = data[data.PMUN14 != 0]
    population = data['PMUN14'].as_matrix()
elif 'PMUN15' in data.columns:
    year = u'15'
    data = data[data.PMUN15 != 0]
    population = data['PMUN15'].as_matrix()
elif 'PMUN16' in data.columns:
    year = u'16'
    data = data[data.PMUN16 != 0]
    population = data['PMUN16'].as_matrix()

X = data[variables].as_matrix(); 
surface = data['SURFACE'].as_matrix()
densite = population/surface;
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

pca = joblib.load(u'model_classification/PCA_classification.pkl')
scaler = joblib.load(u'model_classification/Scaler_classification.pkl')
Xsc = scaler.transform(X)
X = pca.transform(Xsc)

if model[-3:] == 'pkl':
    clf = joblib.load(model)
    yp = clf.predict(X)
elif model[-2:] == 'h5':
    clf = load_model(model)
    try:
        yp = np.argmax(clf.predict(X),axis=1)
    except:
        yp = np.argmax(clf.predict(X.reshape(X.shape + (1,))),axis=1)

data['CLASSIFICATION'] = yp
if model[-3:] == 'pkl':
    error = 100*(1-clf.score(X,y))
elif model[-2:] == 'h5':
    try:
        error = 100*(1-clf.evaluate(X,label_binarize(y,np.arange(nc+1)))[1])
    except:
        error = 100*(1-clf.evaluate(X.reshape(X.shape + (1,)),label_binarize(y,np.arange(nc+1)))[1])
data['ERROR'] = error
print("error : "+str(error)+"%")

try:
	os.mkdir(folder+u'/test/')
except:
	pass
print(model)
folder_model = folder+u'/test/'+os.path.basename(model)[:str.rfind(os.path.basename(model),'.')]+'/'
try:
	os.mkdir(folder_model)
except:
	pass

file_test_prediction = folder_model+os.path.basename(model)[:str.rfind(os.path.basename(model),'.')]+u'_prediction.csv'
data.to_csv(file_test_prediction,encoding='utf-8')

#print(u'~/anaconda2/bin/python density_plot.py '+file_test_prediction+u' '+year)
os.system(u'~/anaconda2/bin/python code/python/plot/density_plot.py '+file_test_prediction+u' '+year+u' '+os.path.basename(model)[:str.rfind(os.path.basename(model),'.')]+u'.png')
#os.system(u'python density_plot.py '+file_test_prediction+u' '+year+u' '+os.path.basename(model)[:str.rfind(os.path.basename(model),'.')]+u'.png')

compute_roc = True

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

if compute_roc:
    # Compute ROC curve and ROC area for each class
    
    if 'decision_function' in dir(clf):
        yp = clf.decision_function(X)
    elif 'predict_proba' in dir(clf):
        try:
            yp = clf.predict_proba(X)
        except:
            yp = clf.predict_proba(X.reshape(X.shape + (1,)))
    else:
        try:
            yp = label_binarize(clf.predict(X), classes=np.arange(nc+1))
        except:
            yp = label_binarize(clf.predict(X.reshape(X.shape + (1,))), classes=np.arange(nc+1))
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
    plt.savefig(folder_model+os.path.basename(model)[:str.rfind(os.path.basename(model),'.')]+u'_roc_mean.png',dpi=1000)
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
    plt.savefig(folder_model+os.path.basename(model)[:str.rfind(os.path.basename(model),'.')]+u'_roc.png',dpi=1000)
    #plt.show()

