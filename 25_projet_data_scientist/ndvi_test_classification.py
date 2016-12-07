#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Author : Youcef KACER <youcef.kacer@gmail.com>
# License: BSD 3 clause

from __future__ import print_function
import pandas as pd
import numpy as np

from sklearn.externals import joblib

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
X = pca.transform(X)
clf = joblib.load(model)
yp = clf.predict(X)
data['CLASSIFICATION'] = yp
error = 100*(1-clf.score(X,y))
data['ERROR'] = error
print("error : "+str(error)+"%")

try:
	os.mkdir(folder+u'/test')
except:
	pass

file_test_prediction = folder+u'/test/'+os.path.basename(model)[:-4]+u'_prediction.csv'
data.to_csv(file_test_prediction,encoding='utf-8')


print(u'~/anaconda2/bin/python density_plot.py '+file_test_prediction+u' '+year)
os.system(u'~/anaconda2/bin/python density_plot.py '+file_test_prediction+u' '+year+u' '+os.path.basename(model)[:-4]+u'.png')
print(u'~/anaconda2/bin/python density_plot.py '+file_test_prediction+u' '+year)

