#Author : Youcef KACER <youcef.kacer@gmail.com>
# License: BSD 3 clause

from __future__ import print_function
import pandas as pd
import numpy as np

from sklearn.externals import joblib

import sys
import os


file_test = sys.argv[1]
folder = os.path.dirname(file_test);
model = sys.argv[2]
data = pd.read_csv(file_test,encoding='utf-8',na_values='NaN')
data.dropna(how='any',inplace=True)
data = data[data.SURFACE != 0]
print(str(data.shape[0])+" cities")

variables = [v for v in data.columns if v not in ['INDEX','LIBMIN','PMUN','PMUN13','PMUN14','PMUN15','PMUN16','LAT','LONG','SURFACE','DENSITE']]
y = data['DENSITE'].as_matrix()
X = data[variables].as_matrix();
clf = joblib.load(model)
yp = clf.predict(X)
data['PREDICTION'] = yp
error = 1-clf.score(X,y)
print("error : "+str(error))

try:
	os.mkdir(folder+u'/test')
except:
	pass

file_test_prediction = folder+u'/test/'+os.path.basename(model)[:-4]+u'_prediction.csv'
data.to_csv(file_test_prediction,encoding='utf-8')

if u'PMUN13' in data.columns:
	year = u'13'
elif u'PMUN14' in data.columns:
	year = u'14'
elif u'PMUN15' in data.columns:
	year = u'15'
elif u'PMUN16' in data.columns:
	year = u'16'
if u'PMUN13' in data.columns:
	year = u'13'
print(u'~/anaconda2/bin/python density_plot.py '+file_test_prediction+u' '+year)
os.system(u'~/anaconda2/bin/python density_plot.py '+file_test_prediction+u' '+year+u' '+os.path.basename(model)[:-4]+u'.png')


