# coding: utf-8

import pandas as pd
import numpy as np
import sys
import os
import re
import shutil
import commands
import codecs
from itertools import izip
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import utm
from pyproj import Proj
import gdal
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
frontiers = [-1,-0.1,-0.03,0.1,0.3,0.5,0.9,1.0]
f = (frontiers-np.min(frontiers))/(np.max(frontiers)-np.min(frontiers))
dark_green = np.asarray([47,79,47])/255.0;
light_green = np.asarray([34,139,34])/255.0;
light_brown = np.asarray([205,133,63])/255.0;
dark_brown = np.asarray([139,69,19])/255.0;

blue = c('blue');
cyan = c('cyan');
red = c('red');
dark_red = np.asarray([139,0,0])/255.0;
yellow = c('yellow');

# generate custom color map using limits defined below
#cmap_ndvi = make_colormap([blue,f[0],blue,cyan,f[2],red,yellow,f[4],yellow,light_brown,f[5],light_brown,light_green,f[6],light_green,dark_green,f[7],dark_green])
cmap_ndvi = make_colormap([blue,f[0],blue,cyan,f[2],red,yellow,f[4],yellow,light_green,f[5],light_green,dark_green,f[6],dark_green,dark_brown,f[7],dark_brown])
#cmap_ndvi = []


data_file = sys.argv[1]
usgs_file = sys.argv[2]
year = sys.argv[3]

nbins = 1024;
verbose = 0;
record = 0;

folder = os.path.dirname(data_file);
features_file = folder+"/ndvi_features.csv"
features = codecs.open(features_file, 'w', 'utf-8')
features.write('LIBMIN,'+','.join(str(i) for i in np.arange(nbins).tolist())+u',LAT,LONG,DENSITE,PMUN'+year+',SURFACE,COVERING\n')

data = pd.read_csv(data_file,encoding='utf-8',na_values="NaN",keep_default_na=False)
data.dropna(how="any", inplace=True);
print data.shape
cities = {}
cities['latitude'] = np.array(data[u'LAT'])
cities['longitude'] = np.array(data[u'LONG'])
cities['nom'] = list(data[u'LIBMIN'])
cities['surface'] = list(data[u'SURFACE'])
cities['population'] = np.array(data[u'PMUN'+year])
cities['densite'] =  cities['population']/cities['surface']
cities['surface'] = np.array(data[u'SURFACE'])

data = pd.read_csv(usgs_file,sep=',',header=0,encoding='utf-8',usecols=np.arange(42,53))
images = {}
images['Clat'] = np.array(data['Center Latitude dec']);  
images['Clong'] = np.array(data['Center Longitude dec']);  
images['NWlat'] = np.array(data['NW Corner Lat dec']);
images['NWlong'] = np.array(data['NW Corner Long dec']);
images['NElat'] = np.array(data['NE Corner Lat dec']);
images['NElong'] = np.array(data['NE Corner Long dec']);
images['SElat'] = np.array(data['SE Corner Lat dec']);
images['SElong'] = np.array(data['SE Corner Long dec']);
images['SWlat'] = np.array(data['SW Corner Lat dec']);
images['SWlong'] = np.array(data['SW Corner Long dec']);
images['ID'] = list(data['Display ID']);

lat_min = np.min([images['NWlat'].min(),images['SWlat'].min(),images['NElat'].min(),images['SElat'].min()])
lat_max = np.max([images['NWlat'].max(),images['SWlat'].max(),images['NElat'].max(),images['SElat'].max()])

long_min = np.min([images['NWlong'].min(),images['SWlong'].min(),images['NElong'].min(),images['SElong'].min()])
long_max = np.max([images['NWlong'].max(),images['SWlong'].max(),images['NElong'].max(),images['SElong'].max()])

folder_cities = folder+'/cities'
if os.path.isdir(folder_cities):
	shutil.rmtree(folder_cities)
os.mkdir(folder_cities)

IDprev = ""
for (name,lt,lg,de,p,s) in izip(cities['nom'],cities['latitude'],cities['longitude'],cities['densite'],cities['population'],cities['surface']):
	name = re.sub('\/','-',name)
	if (lt<lat_min) | (lt>lat_max) | (lg<long_min) | (lg>long_max):
		print "Warning : "+name+" not in any of the provided Landsat datasets..."
		continue;

	nearest_dataset = np.argmin((lt-images['Clat'])**2+(lg-images['Clong'])**2)
	ID = images['ID'][nearest_dataset]

	#print "\nprocessing "+folder+'/'+ID+' for city '+name+'...'
	if os.path.isdir(folder+'/'+ID) == False:
		os.system('/usr/local/bin/landsat download '+ID+' --bands 2345 --dest '+folder)
		if os.path.isfile(folder+'/'+ID+'.tar.bz') == True:
			os.mkdir(folder+'/'+ID)
			os.system('tar xfvj '+folder+'/'+ID+'.tar.bz -C '+folder+'/'+ID)
			os.system('rm '+folder+'/'+ID+'.tar.bz')

	if os.path.isdir(folder_cities+'/'+name) == False:
		os.mkdir(folder_cities+'/'+name)
	
	if os.path.isfile(folder+'/'+ID+'_NDVI.TIF')==False | os.path.isfile(folder+'/'+ID+'_RGB.TIF')==False:
		os.popen("gdalwarp -t_srs EPSG:3857 "+folder+'/'+ID+'/'+ID+'_B2.TIF '+folder+'/'+ID+'/'+ID+'_proj_B2.TIF ');
		os.popen("gdalwarp -t_srs EPSG:3857 "+folder+'/'+ID+'/'+ID+'_B3.TIF '+folder+'/'+ID+'/'+ID+'_proj_B3.TIF ');
		os.popen("gdalwarp -t_srs EPSG:3857 "+folder+'/'+ID+'/'+ID+'_B4.TIF '+folder+'/'+ID+'/'+ID+'_proj_B4.TIF ');
		os.popen("gdalwarp -t_srs EPSG:3857 "+folder+'/'+ID+'/'+ID+'_B5.TIF '+folder+'/'+ID+'/'+ID+'_proj_B5.TIF ');

	if os.path.isfile(folder+'/'+ID+'_NDVI.TIF')==False:
	    os.system('/usr/bin/python ndvi_computation.py '+folder+'/'+ID+'/'+ID+'_proj_B4.TIF '+folder+'/'+ID+'/'+ID+'_proj_B5.TIF '+folder+'/'+ID+'_NDVI.TIF')

	if os.path.isfile(folder+'/'+ID+'_RGB.TIF')==False:
	    for band in [2,3,4]:
		os.system('gdal_contrast_stretch -ndv 0 -linear-stretch 70 30 '+folder+'/'+ID+'/'+ID+'_proj_B'+str(band)+'.TIF '+folder+'/'+ID+'_B'+str(band)+'_8.TIF');
	    os.system('gdal_merge_simple -in '+folder+'/'+ID+'_B4_8.TIF -in '+folder+'/'+ID+'_B3_8.TIF -in '+folder+'/'+ID+'_B2_8.TIF  -out '+folder+'/'+ID+'_RGB.TIF')
	    for band in [2,3,4]:
		os.system('rm '+folder+'/'+ID+'_B'+str(band)+'_8.TIF');
	    for band in [2,3,4,5]:
		os.system('rm '+folder+'/'+ID+'/'+ID+'_proj_B'+str(band)+'_8.TIF');

	if os.path.isfile(folder+'/'+ID+'_QUALITY.TIF')==False:
            os.popen('gdalwarp -t_srs EPSG:3857 '+ folder+'/'+ID+'/'+ID+'_BQA.TIF '+folder+"/"+ID+"_QUALITY.TIF");

	if ID != IDprev:
		ndvi_grey = (cv2.imread(folder+'/'+ID+'_NDVI.TIF',-1).astype(np.float32))/(2**15-1)-1;
		ndvi_grey[ndvi_grey>1]=1
		quality = cv2.imread(folder+'/'+ID+'_QUALITY.TIF',-1);
	if verbose:
		print "ndvi min  :",np.min(ndvi_grey)
		print "ndvi max  :",np.max(ndvi_grey)
	if cmap_ndvi == []:
		cdict = {}
		os.system('/usr/local/bin/landsat process '+folder+'/'+ID+' --ndvi')
		if ID != IDprev:
			ndvi = cv2.imread(folder+'/'+ID+'_NDVI.TIF',-1);
		if verbose:
			print ndvi_grey.shape
			print ndvi.shape
		for j in np.arange(ndvi_grey.shape[0]):
		    for i in np.arange(ndvi_grey.shape[1]):
		        cdict[ndvi_grey[j,i]] = ndvi[j,i,:]/255.0
		if verbose:
			print "creating colormap for ndvi image"
			print len(cdict.keys())
		seq = [];
		for i in np.arange(0,1.01,0.1):
		    if int(255*i) in cdict.keys():
		        seq.append(cdict[int(255*i)])
		        seq.append(i);
		        seq.append(cdict[int(255*i)])
		cmap_ndvi = make_colormap(seq)

	if record:
		## FORM RGB ZOOM OF THE CITY
		if ID != IDprev:
			rgb = cv2.imread(folder+'/'+ID+'_RGB.TIF',-1);
		geo = gdal.Open(folder+"/"+ID+"_RGB.TIF")
		geo_t = geo.GetGeoTransform()
		ul_x = geo_t[0]
		ul_y = geo_t[3]
		br_x = geo_t[0] + geo_t[1]*geo.RasterXSize
		br_y = geo_t[3] + geo_t[5]*geo.RasterYSize
		if verbose:
			print "ul_y : ",ul_y
			print "br_y : ",br_y
			print "ul_x : ",ul_x
			print "br_x : ",br_x
		[c_x,c_y]=Proj("+init=EPSG:3857 +units=m +no_defs")(lg,lt)
		if verbose:
			print "c_y : ",c_y
			print "c_x : ",c_x
		d = np.sqrt(s)*1000 # window size in meters 
		if verbose:
			print "d : ",d
			print "h,w rgb : ",rgb.shape
		x1 = int(float(c_x-d/2-ul_x)/(br_x-ul_x)*rgb.shape[1])
		y1 = int(float(c_y-d/2-ul_y)/(br_y-ul_y)*rgb.shape[0])
		x2 = int(float(c_x+d/2-ul_x)/(br_x-ul_x)*rgb.shape[1])
		y2 = int(float(c_y+d/2-ul_y)/(br_y-ul_y)*rgb.shape[0])
		if verbose:
			print "x1 : ",x1
			print "x2 : ",x2
			print "y1 : ",y1
			print "y2 : ",y2
		if (x1<0) | (y1<0) | (x2>=rgb.shape[1]) | (y2>=rgb.shape[0]):
		    IDprev = ID
		    print "Warning : "+name+" doesn't fit into the nearest dataset..."
		    continue;

		zoom_rgb = rgb[y2:y1,x1:x2,:];
		if verbose:
			print "zoom rgb shape : ",zoom_rgb.shape
		cv2.imwrite((folder_cities+'/'+name+'/'+name+'_rgb.png').encode("utf-8"),zoom_rgb);

	## FORM NDVI ZOOM OF THE CITY
	geo = gdal.Open(folder+"/"+ID+"_NDVI.TIF")
	geo_t = geo.GetGeoTransform()
	ul_x = geo_t[0]
	ul_y = geo_t[3]
	br_x = geo_t[0] + geo_t[1]*geo.RasterXSize
	br_y = geo_t[3] + geo_t[5]*geo.RasterYSize
	if verbose:
		print "ul_y : ",ul_y
		print "br_y : ",br_y
		print "ul_x : ",ul_x
		print "br_x : ",br_x
	[c_x,c_y]=Proj("+init=EPSG:3857 +units=m +no_defs")(lg,lt)
	if verbose:
		print "c_y : ",c_y
		print "c_x : ",c_x
	d = np.sqrt(s)*1000 # window size in meters 
	if verbose:
		print "d : ",d
		print "h,w ndvi : ",ndvi_grey.shape
	x1 = int(float(c_x-d/2-ul_x)/(br_x-ul_x)*ndvi_grey.shape[1])
	y1 = int(float(c_y-d/2-ul_y)/(br_y-ul_y)*ndvi_grey.shape[0])
	x2 = int(float(c_x+d/2-ul_x)/(br_x-ul_x)*ndvi_grey.shape[1])
	y2 = int(float(c_y+d/2-ul_y)/(br_y-ul_y)*ndvi_grey.shape[0])
	if verbose:
		print "x1 : ",x1
		print "x2 : ",x2
		print "y1 : ",y1
		print "y2 : ",y2
	zoom = ndvi_grey[y2:y1,x1:x2];
	if verbose:
		print "zoom ndvi shape :",zoom.shape
	
	if record:
		fig = plt.figure(); 
		cplt=plt.imshow(zoom,cmap=cmap_ndvi,vmin=-1,vmax=1); 
		cbar = fig.colorbar(cplt, ticks=frontiers);
		cbar.ax.set_yticklabels([str(fr) for fr in frontiers])
		plt.savefig(folder_cities+'/'+name+'/'+name+'_ndvi_colormap.png')
		zoom_rescaled = 255*(zoom-(-1))/(1-(-1));
		cv2.imwrite((folder_cities+'/'+name+'/'+name+'_ndvi.png').encode("utf-8"),(255*cmap_ndvi(zoom_rescaled.astype(np.uint8))).astype(np.uint8)[:,:,:3][:,:,::-1]);
		plt.gcf().clear()

		fig = plt.figure();
	bins = np.linspace(-1,1,nbins+1);
	histo = np.histogram(zoom,bins)[0]
	if record:
		plt.plot(bins[:-1],histo)
		plt.savefig(folder_cities+'/'+name+'/'+name+'_ndvi_histo.png')
		plt.gcf().clear()

	## FORM CLOUD COVERING ZOOM OF THE CITY
        geo = gdal.Open(folder+"/"+ID+"_QUALITY.TIF")
        geo_t = geo.GetGeoTransform()
        ul_x = geo_t[0]
        ul_y = geo_t[3]
        br_x = geo_t[0] + geo_t[1]*geo.RasterXSize
        br_y = geo_t[3] + geo_t[5]*geo.RasterYSize
	if verbose:
        	print "ul_y : ",ul_y
        	print "br_y : ",br_y
        	print "ul_x : ",ul_x
        	print "br_x : ",br_x
        [c_x,c_y]=Proj("+init=EPSG:3857 +units=m +no_defs")(lg,lt)
	if verbose:
        	print "c_y : ",c_y
       		print "c_x : ",c_x
        d = np.sqrt(s)*1000 # window size in meters 
	if verbose:
        	print "d : ",d
        	print "h,w quality : ",quality.shape

        x1 = int(float(c_x-d/2-ul_x)/(br_x-ul_x)*quality.shape[1])
        y1 = int(float(c_y-d/2-ul_y)/(br_y-ul_y)*quality.shape[0])
        x2 = int(float(c_x+d/2-ul_x)/(br_x-ul_x)*quality.shape[1])
        y2 = int(float(c_y+d/2-ul_y)/(br_y-ul_y)*quality.shape[0])
        if ( (x1<0) | (x2<0) | (y1<0) | (y2<0) | (x2>=quality.shape[1]) | (y2>=quality.shape[0]) ):
                print name+" not in "+ID+"..."
                continue
	if verbose:
        	print "x1 : ",x1
        	print "x2 : ",x2
        	print "y1 : ",y1
        	print "y2 : ",y2

        zoom_quality = quality[y2:y1,x1:x2];
	if verbose:
        	print "zoom quality shape : ",zoom_quality.shape

        zoom_quality_cloud = np.zeros_like(zoom_quality).astype(np.float64)
        zoom_quality_cloud[zoom_quality==61440]=1.0
        zoom_quality_cloud[zoom_quality==59424]=1.0
        zoom_quality_cloud[zoom_quality==57344]=1.0
        zoom_quality_cloud[zoom_quality==56320]=1.0
        zoom_quality_cloud[zoom_quality==53248]=1.0
	q = 100.0*zoom_quality_cloud.mean()
	if record:
		cv2.imwrite((folder_cities+'/'+name+'/'+name+'_quality.png').encode("utf-8"),65535.0*zoom_quality_cloud);

	features.write(unicode(name)+u','+u",".join(unicode(str(i)) for i in histo.tolist())+u','+unicode(str(lt))+u','+unicode(str(lg))+u','+unicode(str(de))+u','+unicode(str(p))+u','+unicode(str(s))+u','+unicode(str(q))+u'\n')
	IDprev = ID
features.close()

