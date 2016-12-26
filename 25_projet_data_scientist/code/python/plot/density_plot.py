#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import codecs
from itertools import izip 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
import matplotlib as mpl
c = mcolors.ColorConverter().to_rgb

from sklearn import metrics

from pyproj import Proj

#cities_file = 'Pays-Bas/test/Random_Forest_Regression_prediction.csv'
#year = u'14'
#model = 'Random_Forest_classification'
#narg = 4

#cities_file = 'Pays-Bas/test/Gradient_Boosting_Classification_prediction.csv'
#year = u'14'
#model = 'Gradient_Boosting_Classification'
#narg = 4

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

debugging = 0

cities_file = sys.argv[1]
year = sys.argv[2]
model = sys.argv[-1]
narg = len(sys.argv)

folder = os.path.dirname(cities_file)
if narg==4:
	folder  = folder + u'/'+model[:-4]+u'/'
	try:
		os.mkdir(folder)
	except:
		pass
	density_regression = folder+u'/'+model[:-4]+u'.png'
	log = codecs.open(folder+u'/'+model[:-4]+u'_log.txt','w','utf-8')
else:
	log = None

df = pd.read_csv(cities_file,encoding='utf-8')
df.dropna(how='any',inplace=True)
df = df[df.SURFACE != 0]

if year==u'13':
    df = df[df.PMUN13 != 0]
elif year==u'14':
    df = df[df.PMUN14 != 0]
elif year==u'15':
    df = df[df.PMUN15 != 0]
elif year==u'16':
    df = df[df.PMUN16 != 0]

print str(df.shape[0])+" cities"
if log:
    log.write(str(df.shape[0])+" cities\n")

name = df['LIBMIN'].tolist()
lg = df[u'LONG'].as_matrix()
lt = df[u'LAT'].as_matrix()
x = np.array([Proj(init="EPSG:3857")(longi,lat)[0] for lat,longi in izip(lt,lg)]).astype(np.int)/1000.0
y = np.array([Proj(init="EPSG:3857")(longi,lat)[1] for lat,longi in izip(lt,lg)]).astype(np.int)/1000.0
s = df[u'SURFACE'].as_matrix().astype(np.float64)
p = df[u'PMUN'+year].as_matrix().astype(np.float64)

if 'PREDICTION' in df.columns:
    yr = df['PREDICTION'].as_matrix()
    print "prediction max : ",yr.max(),u"habs/km²"
    print "prediction min : ",yr.min(),u"habs/km²"
    if log:
	log.write("prediction max : "+str(yr.max())+u"habs/km²\n")
        log.write("prediction min : "+str(yr.min())+u"habs/km²\n")

d = p/s
density = folder+u'/density.png'
density_cat = folder+u'/density_ground_truth.png'

print "density max : ",d.max(),u"habs/km²"
print "density min : ",d.min(),u"habs/km²"
if log:
    log.write("density max : "+str((p/s).max())+u"habs/km²\n")
    log.write("density min : "+str((p/s).min())+u"habs/km²\n")

if 'COVERING' in df.columns:
    covering = df['COVERING'].as_matrix()

dpi = 500
size_pt = 50
if (y.max()-y.min())>1000:
    size_pt = 5

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

colors_error = [[0,0,255],
[28,28,255],
[57,57,255],
[85,85,255],
[113,113,255],
[142,142,255],
[170,170,255],
[198,198,255],
[227,227,255],
[0,0,0],
#[255,51,153],
#[255,51,153],
#[255,51,153],
#[255,51,153],
[227,255,227],
[198,255,198],
[170,255,170],
[142,255,142],
[113,255,113],
[85,255,85],
[57,255,57],
[28,255,28],
[0,255,0]]

colors_covering = [[210,210,210],
[180,180,180],
[150,150,150],
[120,120,120],
[90,90,90],
[70,70,70],
[60,60,60],
[50,50,50],
[40,40,40],
[30,30,30]]


# plot density
densities = np.asarray([-100000,0,5,15,30,50,80,110,150,250,500,1000,2000,4000,8000,15000,20000,25000,30000,35000,40000,45000]).astype(np.float64)
fig = plt.figure()
ax = plt.subplot(111)
legend_labels = []
legend_scatters = []
N =0 
for i,cl in enumerate(colors):
	ci = np.asarray(cl)/255.0
	d1 = densities[i]
	d2 = densities[i+1]
	xi = x[(d>=d1) & (d<d2)]
	yi = y[(d>=d1) & (d<d2)]
	di = d[(d>=d1) & (d<d2)]
	si = s[(d>=d1) & (d<d2)]
	ni = di.shape[0]
	N = N+ni
	sci = plt.scatter(xi,yi,s=size_pt,marker='o',facecolor=[ci,]*10,edgecolor=[ci,]*3)
	legend_scatters.append(sci)
	legend_labels.append(str(int(d1))+u' - '+str(int(d2))+u' habs/km²')

print u'******** N : '+str(N)+' points'
plt.xlabel('x Web Mercator (km)')
plt.ylabel('y Web Mercator (km)')
ax.legend(legend_scatters,legend_labels,loc='center left', bbox_to_anchor=(0.83, 0.5),fontsize = 'xx-small')
plt.savefig(density,dpi=dpi)
if debugging:
	plt.show()

# plot cat density
categorization = [0,500,2000,5000,10000,13000]
nc = len(categorization)
dc = -1*np.ones(d.shape)
categorization_r = list(reversed(categorization))
for i,n in enumerate(d):
        if n>categorization_r[0]:
            dc[i] = 5
        elif n>categorization_r[1]:
            dc[i] = 4
        elif n>categorization_r[2]:
            dc[i] = 3
        elif n>categorization_r[3]:
            dc[i] = 2
        elif n>categorization_r[4]:
            dc[i] = 1
        else:
            dc[i] = 0

fig = plt.figure()
ax = plt.subplot(111)
legend_labels = []
legend_scatters = []
N = 0
for i in np.arange(nc-1):
    ci = np.asarray(colors[1:][int(1.0*i/nc*len(colors))])/255.0
    e1 = categorization[i]
    e2 = categorization[i+1]
    xi = x[dc==i]
    yi = y[dc==i]
    di = d[dc==i]
    si = s[dc==i]
    ni = di.shape[0]
    N = N+ni
    sci = plt.scatter(xi,yi,s=size_pt,marker='o',facecolor=[ci,]*10,edgecolor=[ci,]*3)
    legend_scatters.append(sci)
    legend_labels.append(str(int(e1))+u' - '+str(int(e2))+u' habs/km²')

xi = x[dc==(nc-1)]
yi = y[dc==(nc-1)]
di = d[dc==(nc-1)]
si = s[dc==(nc-1)]
ni = di.shape[0]
N = N+ni
ci = np.asarray(colors[-1])/255.0
sci = plt.scatter(xi,yi,s=size_pt,marker='o',facecolor=[ci,]*10,edgecolor=[ci,]*3)
legend_scatters.append(sci)
legend_labels.append(u'> '+str(categorization[-1])+u' habs/km²')
print u'******** N : '+str(N)+' points'
plt.xlabel('x Web Mercator (km)')
plt.ylabel('y Web Mercator (km)')
ax.legend(legend_scatters,legend_labels,loc='center left', bbox_to_anchor=(0.83, 0.5),fontsize = 'xx-small')
plt.savefig(density_cat,dpi=dpi)
if debugging:
	plt.show()
plt.clf()

if 'PREDICTION' in df.columns:
	# plot density prediction
	fig = plt.figure()
	ax = plt.subplot(111)
	legend_labels = []
	legend_scatters = []
	N =0
	for i,cl in enumerate(colors):
        	ci = np.asarray(cl)/255.0
        	d1 = densities[i]
        	d2 = densities[i+1]
        	xi = x[(yr>=d1) & (yr<d2)]
        	yi = y[(yr>=d1) & (yr<d2)]
        	di = d[(yr>=d1) & (yr>d2)]
        	si = s[(yr>=d1) & (yr<d2)]
        	ni = xi.shape[0]
       		N = N+ni
        	sci = plt.scatter(xi,yi,s=size_pt,marker='o',facecolor=[ci,]*10,edgecolor=[ci,]*3)
        	legend_scatters.append(sci)
        	legend_labels.append(str(int(d1))+u' - '+str(int(d2))+u' habs/km²')
	
	print u'******** N : '+str(N)+' points'
	plt.xlabel('x Web Mercator (km)')
	plt.ylabel('y Web Mercator (km)')
	ax.legend(legend_scatters,legend_labels,loc='center left', bbox_to_anchor=(0.83, 0.5),fontsize = 'xx-small')
	density_prediction = density[:-4]+u'_prediction.png'
	plt.savefig(density_prediction,dpi=1000)
	if debugging:
		plt.show()
	plt.clf()
	# plot density prediction error
	error = (yr-d)/d*100
	imin = np.argmin(error)
	imax = np.argmax(error)
	print "error max :",error.max(),"% for city ",name[imax],"(truth:",(p/s)[imax],u"habs/km²,predicted:",d[imax],u"habs/km²)"
	print "error min :",error.min(),"% for city ",name[imin],"(truth:",(p/s)[imin],u"habs/km²,predicted:",d[imin],u"habs/km²)"
	if log:
		log.write("error max :"+str(error.max())+"% for city "+name[imax]+"(truth:"+str((p/s)[imax])+u"habs/km²,predicted:"+str(d[imax])+u"habs/km²)\n")
		log.write("error min :"+str(error.min())+"% for city "+name[imin]+"(truth:"+str((p/s)[imin])+u"habs/km²,predicted:"+str(d[imin])+u"habs/km²)\n")
	densities_error = np.asarray([-10000000,-100,-50,-30,-20,-10,-5,-3,-2,-1,1,2,3,5,10,20,30,50,100,10000000]).astype(np.float64)
	# plot density errors
	fig = plt.figure()
	ax = plt.subplot(111)
	legend_labels = []
	legend_scatters = []
	N = 0
	for i,cl in enumerate(colors_error):
		ci = np.asarray(cl)/255.0
		e1 = densities_error[i]
		e2 = densities_error[i+1]
		xi = x[(error>=e1) & (error<e2)]
		yi = y[(error>=e1) & (error<e2)]
		di = d[(error>=e1) & (error<e2)]
		si = s[(error>=e1) & (error<e2)]
		ni = di.shape[0]
		N = N+ni
		sci = plt.scatter(xi,yi,s=size_pt,marker='o',facecolor=[ci,]*10,edgecolor=[ci,]*3)
		legend_scatters.append(sci)
		if i==0:
			legend_labels.append(u'< '+str(int(e2))+u' %')
		elif i==len(colors_error)-1:
			legend_labels.append(u'> '+str(int(e1))+u' %')
		else:
			legend_labels.append(str(int(e1))+u' à '+str(int(e2))+u' %')
		
	print u'******** N : '+str(N)+' points'
	plt.xlabel('x Web Mercator (km)')
	plt.ylabel('y Web Mercator (km)')
	ax.legend(legend_scatters,legend_labels,loc='center left', bbox_to_anchor=(0.83, 0.5),fontsize = 'small')
	density_error = density[:-4]+u'_error.png'
	plt.savefig(density_error,dpi=dpi)
	if debugging:
		plt.show()
	plt.clf()
	density_error_histo = density[:-4]+u'_error_histo.png'
	fig = plt.figure();
	h,b = np.histogram(error,range(-100,100,1))
	plt.bar(b[:-1],h)
	plt.xlabel('error bins (%)')
	plt.savefig(density_error_histo,dpi=dpi)
	if debugging:
		plt.show()
	plt.clf()

if 'CLASSIFICATION' in df.columns:
    #plot classification density
    yc = df['CLASSIFICATION'].as_matrix().astype(np.int64)
    categorization = [0,500,2000,5000,10000,13000]
    nc = len(categorization)
    for i in np.arange(nc):
        print("categorie "+str(i)+": "+str((yc==i).sum())+" predicted samples")
    target_names = []
    for i in np.arange(nc-1):
        c1 = str(categorization[i])
        c2 = str(categorization[i+1])
        target_names.append(c1+u' - '+c2+u' habs/km²')
    
    target_names.append(u'> '+str(categorization[-1])+u' habs/km²')
    fig = plt.figure()
    ax = plt.subplot(111)
    legend_labels = []
    legend_scatters = []
    N = 0
    for i in np.arange(nc-1):
        ci = np.asarray(colors[1:][int(1.0*i/nc*len(colors))])/255.0
	e1 = categorization[i]
	e2 = categorization[i+1]
	xi = x[yc==i]
	yi = y[yc==i]
	yi = y[yc==i]
	di = d[yc==i]
	ni = di.shape[0]
	N = N+ni
	sci = plt.scatter(xi,yi,s=size_pt,marker='o',facecolor=[ci,]*30,edgecolor=[ci,]*3)
	legend_scatters.append(sci)
	legend_labels.append(str(int(e1))+u' - '+str(int(e2))+u' habs/km²')
    xi = x[yc==(nc-1)]
    yi = y[yc==(nc-1)]
    di = d[yc==(nc-1)]
    si = s[yc==(nc-1)]
    ni = di.shape[0]
    N = N+ni
    ci = np.asarray(colors[-1])/255.0
    sci = plt.scatter(xi,yi,s=size_pt,marker='o',facecolor=[ci,]*10,edgecolor=[ci,]*3)
    legend_scatters.append(sci)
    legend_labels.append(u'> '+str(categorization[-1])+u' habs/km²')
    print u'******** N : '+str(N)+' points'
    plt.xlabel('x Web Mercator (km)')
    plt.ylabel('y Web Mercator (km)')
    ax.legend(legend_scatters,legend_labels,loc='center left', bbox_to_anchor=(0.83, 0.5),fontsize = 'xx-small')
    density_classification = density[:-4]+u'_classification.png'
    plt.savefig(density_classification,dpi=dpi)
    if debugging:
	plt.show()
    plt.clf()
    log.write(metrics.classification_report(dc, yc, labels=np.arange(nc).tolist(), target_names=target_names,digits=3))
    info = '\n\n'
    mean_scores = compute_mean_score(dc,yc,nc-1)
    for i in np.arange(nc):
        info = info+target_names[i]+': '+str(100-mean_scores[i])+'%\n'
    info = info+u'\n\n'
    info = info+u'mean error per class : '+str(100-mean_scores.mean())+'%\n\n'
    log.write(info)
    log.write(np.array_str(metrics.confusion_matrix(dc, yc, labels=np.arange(nc).tolist())))
    # plot density classification error
    errorc = yc-dc
    fig = plt.figure()
    ax = plt.subplot(111)
    legend_labels = []
    legend_scatters = []
    N = 0
    for i in range(0,nc):
        ci = np.asarray(colors_error[int(1.0*(i+nc-1)/(2*(nc-1)+1)*len(colors_error))])/255.0
        xi = x[errorc==i]
        yi = y[errorc==i]
        di = d[errorc==i]
        si = s[errorc==i]
        ni = di.shape[0]
        N = N+ni
        sci = plt.scatter(xi,yi,s=size_pt,marker='o',facecolor=[ci,]*10,edgecolor=[ci,]*3)
        legend_scatters.append(sci)
        legend_labels.append(str(int(i)))
    for i in list(reversed(range(-(nc-1),0))):
        ci = np.asarray(colors_error[int(1.0*(i+nc-1)/(2*(nc-1)+1)*len(colors_error))])/255.0
        xi = x[errorc==i]
        yi = y[errorc==i]
        di = d[errorc==i]
        si = s[errorc==i]
        ni = di.shape[0]
        N = N+ni
        sci = plt.scatter(xi,yi,s=size_pt,marker='o',facecolor=[ci,]*10,edgecolor=[ci,]*3)
        legend_scatters.append(sci)
        legend_labels.append(str(int(i)))

    print u'******** N : '+str(N)+' points'
    plt.xlabel('x Web Mercator (km)')
    plt.ylabel('y Web Mercator (km)')
    ax.legend(legend_scatters,legend_labels,loc='center left', bbox_to_anchor=(0.85, 0.5),fontsize = 'small')
    density_classification_error = density[:-4]+'_classification_error.png'
    plt.savefig(density_classification_error,dpi=500)
    if debugging:
	plt.show()
    plt.clf()
    

if 'COVERING' in df.columns:
	cloud_covering = np.asarray([0,1,2,5,10,13,20,25,30,50,100.01]).astype(np.float64)
	# plot density errors
	fig = plt.figure()
	ax = plt.subplot(111)
	legend_labels = []
	legend_scatters = []
	N = 0
	for i,cl in enumerate(colors_covering):
		ci = np.asarray(cl)/255.0
		e1 = cloud_covering[i]
		e2 = cloud_covering[i+1]
		xi = x[(covering>=e1) & (covering<e2)]
		yi = y[(covering>=e1) & (covering<e2)]
		di = d[(covering>=e1) & (covering<e2)]
		si = s[(covering>=e1) & (covering<e2)]
		ni = di.shape[0]
		N = N+ni
		sci = plt.scatter(xi,yi,s=size_pt,marker='o',facecolor=[ci,]*10,edgecolor=[ci,]*3)
		legend_scatters.append(sci)
		legend_labels.append(str(int(e1))+u' à '+str(int(e2))+u' %')
	
	print u'******** N : '+str(N)+' points'
	plt.xlabel('x Web Mercator (km)')
	plt.ylabel('y Web Mercator (km)')
	ax.legend(legend_scatters,legend_labels,loc='center left', bbox_to_anchor=(0.85, 0.5),fontsize = 'small')
	density_covering = density[:-4]+u'_covering.png'
	plt.savefig(density_covering,dpi=dpi)
	if debugging:
		plt.show()
	plt.clf()
	density_covering_histo = density[:-4]+u'_covering_histo.png'
	fig = plt.figure();
	h,b = np.histogram(covering,range(0,100,1))
	plt.bar(b[:-1],h)
	plt.xlabel('cloud covering bins (%)')
	plt.savefig(density_covering_histo,dpi=dpi)
	if debugging:
		plt.show()
	plt.clf()
	density_error_covering = density[:-4]+u'_error_covering.png'
	if 'PREDICTION' in df.columns:
		fig = plt.figure();
		plt.scatter(covering,error);
		plt.xlabel('cloud covering (%)')
		plt.ylabel('error (%)')
		plt.savefig(density_error_covering,dpi=dpi)
		if debugging:
			plt.show()
		plt.clf()
		

if log:
	log.close()


