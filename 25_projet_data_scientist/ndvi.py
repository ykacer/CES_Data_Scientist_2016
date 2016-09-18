import pandas as pd
import numpy as np
import os
import sys
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from itertools import izip
import utm
from pyproj import Proj
from os.path import expanduser
home = expanduser("~")

import matplotlib.colors as mcolors
c = mcolors.ColorConverter().to_rgb

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

# put natural limits for NDVI values
frontiers = [-1,-0.1,0.0,0.1,0.3,0.6,0.9,1.0]
f = (frontiers-np.min(frontiers))/(np.max(frontiers)-np.min(frontiers))
dark_green = np.asarray([47,79,47])/255.0;
light_green = np.asarray([34,139,34])/255.0;
light_brown = np.asarray([205,133,63])/255.0;
blue = c('blue');
cyan = c('cyan');
red = c('red');
dark_red = np.asarray([139,0,0])/255.0;
yellow = c('yellow');

# generate custom color map using limits defined below
cmap_ndvi = make_colormap([blue,f[0],blue,cyan,f[2],red,yellow,f[4],yellow,light_brown,f[5],light_brown,light_green,f[6],light_green,dark_green,f[7],dark_green])
#cmap_ndvi = []

# take USGS csv file containing metadata for datasets
file = sys.argv[1]

# take coordinates for cropping
x1 = int(sys.argv[2])
y1 = int(sys.argv[3])
x2 = int(sys.argv[4])
y2 = int(sys.argv[5])

# name of cropping
name = sys.argv[6];

# loop over datasets, download it if not yet present, construct NDVI using bands 4 and 5. record it using customed colormap
folder = os.path.dirname(file)
image_names = list(pd.read_csv(file,sep=',',header=0,usecols=[52])['Display ID'])
image_months = [month[5:7] for month in list(pd.read_csv(file,sep=',',header=0,usecols=[18])['Date Acquired'])]

for ID,month in izip(image_names,image_months):
	print "processing "+folder+'/'+ID+'...'
	if os.path.isdir(folder+'/'+ID) == False:
		os.system('/usr/local/bin/landsat download '+ID+' --bands 45 --dest '+folder)
		if os.path.isfile(folder+'/'+ID+'.tar.bz') == True:
			os.mkdir(folder+'/'+ID)
 			os.system('tar xfvj '+folder+'/'+ID+'.tar.bz -C '+folder+'/'+ID)
			os.system('rm '+folder+'/'+ID+'.tar.bz')

        if os.path.isdir(folder+'/'+name) == False:
                os.mkdir(folder+'/'+name)
	
        os.system('/usr/local/bin/landsat process '+folder+'/'+ID+' --ndvigrey')
	ndvi_grey = cv2.imread(home+'/landsat/processed/'+ID+'/'+ID+'_NDVI.TIF',-1);
        if cmap_ndvi == []:
                cdict = {}
	        os.system('/usr/local/bin/landsat process '+folder+'/'+ID+' --ndvi')
		ndvi = cv2.imread(home+'/landsat/processed/'+ID+'/'+ID+'_NDVI.TIF',-1);
                for j in np.arange(ndvi_grey.shape[0]):
                    for i in np.arange(ndvi_grey.shape[1]):
                        cdict[ndvi_grey[j,i]] = ndvi[j,i,:]/255.0
                print "creating colormap for ndvi image"
                print len(cdict.keys())
                seq = [];
                for i in np.arange(0,1.01,0.1):
                    if int(255*i) in cdict.keys():
                        seq.append(cdict[int(255*i)])
                        seq.append(i);
                        seq.append(cdict[int(255*i)])
                cmap_ndvi = make_colormap(seq)

	zoom = ndvi_grey[y1:y2,x1:x2];
        fig = plt.figure(); 
        cplt=plt.imshow(zoom,cmap=cmap_ndvi,vmin=0,vmax=255); 
        cbar = fig.colorbar(cplt, ticks=[int(255*fr) for fr in f]);
        cbar.ax.set_yticklabels([str(fr) for fr in frontiers])
        plt.savefig(folder+'/'+name+'/ndvi_'+month+'_map.png')
        cv2.imwrite(folder+'/'+name+'/ndvi_'+month+'.png',np.delete(255*cmap_ndvi(zoom),3,2)[:,:,::-1]);
