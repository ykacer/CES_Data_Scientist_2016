#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from itertools import izip 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors
import matplotlib as mpl
c = mcolors.ColorConverter().to_rgb

from pyproj import Proj

cities_file = sys.argv[1]
folder = os.path.dirname(cities_file)
year = sys.argv[2]

if len(sys.argv)==4:
	density_imagename = folder+u'/'+sys.argv[3]

df = pd.read_csv(cities_file,encoding='utf-8')
df.dropna(how='any',inplace=True)
df = df[df.SURFACE != 0]
print str(df.shape[0])+" cities"

if year==u'13':
    df = df[df.PMUN13 != 0]

if year==u'14':
    df = df[df.PMUN14 != 0]

if year==u'15':
    df = df[df.PMUN15 != 0]

if year==u'16':
    df = df[df.PMUN16 != 0]

lg = df[u'LONG'].as_matrix()
lt = df[u'LAT'].as_matrix()
x = np.array([Proj(init="EPSG:3857")(longi,lat)[0] for lat,longi in izip(lt,lg)]).astype(np.int)/1000.0
y = np.array([Proj(init="EPSG:3857")(longi,lat)[1] for lat,longi in izip(lt,lg)]).astype(np.int)/1000.0
s = df[u'SURFACE'].as_matrix().astype(np.float64)
p = df[u'PMUN'+year].as_matrix().astype(np.float64)


if 'PREDICTION' in df.columns:
    d = df['PREDICTION'].as_matrix()
else:
    d = p/s
    density_imagename = folder+u'/density.png'

dpi = 1000
size_pt = 50
if (y.max()-y.min())>1000:
    size_pt = 5

def make_colormap(seq):
    """ Return a LinearSegmentedColormap
        seq: a sequence of floats and RGB-tuples. The floats should be increasing
        and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

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


# plot density
density = np.asarray([-100000,0,5,15,30,50,80,110,150,250,500,1000,2000,4000,8000,15000,20000,25000,30000,35000,40000,45000]).astype(np.float64)
frontiers = density
f = (frontiers-np.min(frontiers))/(np.max(frontiers)-np.min(frontiers))
color_sequence = []
color_sequence.append(np.asarray(colors[0])/255.0)
for i,cl in enumerate(colors):
    color_sequence.append(f[i])
    color_sequence.append(np.asarray(cl)/255.0)
    color_sequence.append(np.asarray(cl)/255.0)

color_sequence.append(f[-1])
color_sequence.append(np.asarray(colors[-1])/255.0)

cmap_density = make_colormap(color_sequence)

fig = plt.figure()
ax = plt.subplot(111)
legend_labels = []
legend_scatters = []
N =0 
for i,cl in enumerate(colors):
	ci = np.asarray(cl)/255.0
	d1 = density[i]
	d2 = density[i+1]
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
ax.legend(legend_scatters,legend_labels,loc='center left', bbox_to_anchor=(0.85, 0.5),fontsize = 'x-small')
plt.savefig(density_imagename,dpi=1000)
plt.show()

if 'PREDICTION' in df.columns:
	error = (d-p/s)/(p/s)*100
	density_error = np.asarray([-10000000,-100,-50,-30,-20,-10,-5,-3,-2,-1,1,2,3,5,10,20,30,50,100,10000000]).astype(np.float64)
	# plot density errors
	frontiers = density_error
	f = (frontiers-np.min(frontiers))/(np.max(frontiers)-np.min(frontiers))
	color_sequence = []
	color_sequence.append(np.asarray(colors_error[0])/255.0)
	for i,cl in enumerate(colors_error):
    		color_sequence.append(f[i])
    		color_sequence.append(np.asarray(cl)/255.0)
    		color_sequence.append(np.asarray(cl)/255.0)

	color_sequence.append(f[-1])
	color_sequence.append(np.asarray(colors_error[-1])/255.0)

	cmap_density_error = make_colormap(color_sequence)

	fig = plt.figure()
	ax = plt.subplot(111)
	legend_labels = []
	legend_scatters = []
	N = 0
	for i,cl in enumerate(colors_error):
		ci = np.asarray(cl)/255.0
		e1 = density_error[i]
		e2 = density_error[i+1]
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
	ax.legend(legend_scatters,legend_labels,loc='center left', bbox_to_anchor=(0.85, 0.5),fontsize = 'x-small')

	density_error_imagename = density_imagename[:-4]+u'_error.png'
	plt.savefig(density_error_imagename,dpi=dpi)
	plt.show()

	density_error_histoname = density_imagename[:-4]+u'_error_histo.png'
	fig = plt.figure();
	h,b = np.histogram(error,range(-100,100,1))
	plt.bar(b[:-1],h)
	plt.xlabel('error bins (%)')
	plt.savefig(density_error_histoname,dpi=dpi)
	plt.show()
