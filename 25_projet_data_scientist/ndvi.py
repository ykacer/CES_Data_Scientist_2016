# coding: utf-8

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

# take USGS csv file containing metadata for datasets
file = sys.argv[1]

# take coordinates for cropping
c_lat = float(sys.argv[2])
c_long = float(sys.argv[3])
surface = float(sys.argv[4])
densite = float(sys.argv[5])

# name of cropping
name = sys.argv[6];

print u"\nCurrently studying "+name
print u"\t- latitude : "+str(c_lat)+u"°"
print u"\t- longitude : "+str(c_long)+u"°"
print u"\t- surface  : "+str(surface)+u" km²"
print u"\t- density  : "+str(densite)+u" habs/km²"

# loop over datasets, download it if not yet present, construct NDVI using bands 4 and 5. record it using customed colormap
folder = os.path.dirname(file)
image_names = list(pd.read_csv(file,sep=',',header=0,usecols=[52])['Display ID'])
image_months = [month[5:7] for month in list(pd.read_csv(file,sep=',',header=0,usecols=[18])['Date Acquired'])]
nbins = 256
histo_per_month = {}

for ID,month in izip(image_names,image_months):
	print "\nprocessing "+folder+'/'+ID+'.........................................'
	if os.path.isdir(folder+'/'+ID) == False:
		os.system('/usr/local/bin/landsat download '+ID+' --bands 2345 --dest '+folder)
		if os.path.isfile(folder+'/'+ID+'.tar.bz') == True:
			os.mkdir(folder+'/'+ID)
 			os.system('tar xfvj '+folder+'/'+ID+'.tar.bz -C '+folder+'/'+ID)
			os.system('rm '+folder+'/'+ID+'.tar.bz')

        if os.path.isdir(folder+'/'+name) == False:
                os.mkdir(folder+'/'+name)
	
        projection = os.popen("listgeo "+folder+"/"+ID+"/"+ID+"_B4.TIF | grep 'PCS =' | cut -c7-11").read()[:-1]
        if os.path.isfile(folder+'/'+ID+'_NDVI.TIF')==False:
            os.system('/usr/bin/python ndvi_computation.py '+folder+'/'+ID+'/'+ID+'_B4.TIF '+folder+'/'+ID+'/'+ID+'_B5.TIF '+folder+'/'+ID+'_NDVI_temp.TIF')
            print "native projection : EPSG:",projection
            print "target projection : EPSG:3857"
            os.popen('gdalwarp -t_srs EPSG:3857 '+ folder+'/'+ID+'_NDVI_temp.TIF '+folder+"/"+ID+"_NDVI.TIF");
	    os.system('rm '+folder+'/'+ID+'_NDVI_temp.TIF')
        if os.path.isfile(folder+'/'+ID+'_RGB.TIF')==False:
            for band in [2,3,4]:
                os.system('gdal_contrast_stretch -ndv 0 -linear-stretch 70 30 '+folder+'/'+ID+'/'+ID+'_B'+str(band)+'.TIF '+folder+'/'+ID+'_B'+str(band)+'_8.TIF');
            os.system('gdal_merge_simple -in '+folder+'/'+ID+'_B4_8.TIF -in '+folder+'/'+ID+'_B3_8.TIF -in '+folder+'/'+ID+'_B2_8.TIF  -out '+folder+'/'+ID+'_RGB_temp.TIF')
            for band in [2,3,4]:
                os.system('rm '+folder+'/'+ID+'_B'+str(band)+'_8.TIF');
            os.popen('gdalwarp -t_srs EPSG:3857 '+ folder+'/'+ID+'_RGB_temp.TIF '+folder+"/"+ID+"_RGB.TIF");
	    os.system('rm '+folder+'/'+ID+'_RGB_temp.TIF')
	#continue;
        ndvi_grey = (cv2.imread(folder+'/'+ID+'_NDVI.TIF',-1).astype(np.float32))/(2**15-1)-1;
        ndvi_grey[ndvi_grey>1]=1
        print "ndvi min  :",np.min(ndvi_grey)
        print "ndvi max  :",np.max(ndvi_grey)
	rgb = cv2.imread(folder+'/'+ID+'_RGB.TIF',-1);
        if cmap_ndvi == []:
                cdict = {}
	        os.system('/usr/local/bin/landsat process '+folder+'/'+ID+' --ndvi')
                #os.popen("gdalwarp -t_srs EPSG:"+projection+" ~/landsat/processed/"+ID+"/"+ID+"_NDVI.TIF "+ folder+"/"+ID+"_NDVI.TIF");
                #os.popen("cp ~/landsat/processed/"+ID+"/"+ID+"_NDVI.TIF "+ folder+"/"+ID+"_NDVI.TIF")
		ndvi = cv2.imread(folder+'/'+ID+'_NDVI.TIF',-1);
                print ndvi_grey.shape
                print ndvi.shape
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

        ## FORM RGB ZOOM OF THE CITY

        #ul_x = float(os.popen("cat "+folder+"/"+ID+"/"+ID+"_MTL.txt | grep CORNER_UL_PROJECTION_X_PRODUCT | cut -d' ' -f7").read());
        #ul_x = float(os.popen("listgeo "+folder+"/"+ID+"/"+ID+"_B2.TIF | grep 'Upper Left' | cut -d'(' -f2 | cut -d')' -f1 | cut -d',' -f1").read());
        #ul_y = float(os.popen("cat "+folder+"/"+ID+"/"+ID+"_MTL.txt | grep CORNER_UL_PROJECTION_Y_PRODUCT | cut -d' ' -f7").read());
        #ul_y = float(os.popen("listgeo "+folder+"/"+ID+"/"+ID+"_B2.TIF | grep 'Upper Left' | cut -d'(' -f2 | cut -d')' -f1 | cut -d',' -f2").read());
        #br_x = float(os.popen("cat "+folder+"/"+ID+"/"+ID+"_MTL.txt | grep CORNER_LR_PROJECTION_X_PRODUCT | cut -d' ' -f7").read());
        #br_x = float(os.popen("listgeo "+folder+"/"+ID+"/"+ID+"_B2.TIF | grep 'Lower Right' | cut -d'(' -f2 | cut -d')' -f1 | cut -d',' -f1").read());
        #br_y = float(os.popen("cat "+folder+"/"+ID+"/"+ID+"_MTL.txt | grep CORNER_LR_PROJECTION_Y_PRODUCT | cut -d' ' -f7").read());
        #br_y = float(os.popen("listgeo "+folder+"/"+ID+"/"+ID+"_B2.TIF | grep 'Lower Right' | cut -d'(' -f2 | cut -d')' -f1 | cut -d',' -f2").read());
	geo = gdal.Open(folder+"/"+ID+"_RGB.TIF")
	geo_t = geo.GetGeoTransform()
	ul_x = geo_t[0]
	ul_y = geo_t[3]
        br_x = geo_t[0] + geo_t[1]*geo.RasterXSize
        br_y = geo_t[3] + geo_t[5]*geo.RasterYSize

        print "ul_y : ",ul_y
        print "br_y : ",br_y
        print "ul_x : ",ul_x
        print "br_x : ",br_x
        [c_x,c_y]=Proj("+init=EPSG:3857 +units=m +no_defs")(c_long,c_lat)
        print "c_y : ",c_y
        print "c_x : ",c_x
        d = np.sqrt(surface)*1000 # window size in meters 
        print "d : ",d
        print "h,w rgb : ",rgb.shape
        x1 = int(float(c_x-d/2-ul_x)/(br_x-ul_x)*rgb.shape[1])
        y1 = int(float(c_y-d/2-ul_y)/(br_y-ul_y)*rgb.shape[0])
        x2 = int(float(c_x+d/2-ul_x)/(br_x-ul_x)*rgb.shape[1])
        y2 = int(float(c_y+d/2-ul_y)/(br_y-ul_y)*rgb.shape[0])
	if ( (x1<0) | (x2<0) | (y1<0) | (y2<0) | (x2>rgb.shape[1]) | (y2>rgb.shape[0]) ):
		print name+" not in "+ID+"..."
		continue
        print "x1 : ",x1
        print "x2 : ",x2
        print "y1 : ",y1
        print "y2 : ",y2
        zoom_rgb = rgb[y2:y1,x1:x2,:];
        print "zoom rgb shape : ",zoom_rgb.shape

        ## FORM NDVI ZOOM OF THE CITY

        #ul_x = float(os.popen("cat "+folder+"/"+ID+"/"+ID+"_MTL.txt | grep CORNER_UL_PROJECTION_X_PRODUCT | cut -d' ' -f7").read());
        #ul_x = float(os.popen("listgeo "+folder+"/"+ID+"_NDVI.TIF | grep 'Upper Left' | cut -d'(' -f2 | cut -d')' -f1 | cut -d',' -f1").read());
        #ul_y = float(os.popen("cat "+folder+"/"+ID+"/"+ID+"_MTL.txt | grep CORNER_UL_PROJECTION_Y_PRODUCT | cut -d' ' -f7").read());
        #ul_y = float(os.popen("listgeo "+folder+"/"+ID+"_NDVI.TIF | grep 'Upper Left' | cut -d'(' -f2 | cut -d')' -f1 | cut -d',' -f2").read());
        #br_x = float(os.popen("cat "+folder+"/"+ID+"/"+ID+"_MTL.txt | grep CORNER_LR_PROJECTION_X_PRODUCT | cut -d' ' -f7").read());
        #br_x = float(os.popen("listgeo "+folder+"/"+ID+"_NDVI.TIF | grep 'Lower Right' | cut -d'(' -f2 | cut -d')' -f1 | cut -d',' -f1").read());
        #br_y = float(os.popen("cat "+folder+"/"+ID+"/"+ID+"_MTL.txt | grep CORNER_LR_PROJECTION_Y_PRODUCT | cut -d' ' -f7").read());
        #br_y = float(os.popen("listgeo "+folder+"/"+ID+"_NDVI.TIF | grep 'Lower Right' | cut -d'(' -f2 | cut -d')' -f1 | cut -d',' -f2").read());
	geo = gdal.Open(folder+"/"+ID+"_NDVI.TIF")
	geo_t = geo.GetGeoTransform()
	ul_x = geo_t[0]
	ul_y = geo_t[3]
	br_x = geo_t[0] + geo_t[1]*geo.RasterXSize
	br_y = geo_t[3] + geo_t[5]*geo.RasterYSize

        print "ul_y : ",ul_y
        print "br_y : ",br_y
        print "ul_x : ",ul_x
        print "br_x : ",br_x
        [c_x,c_y]=Proj("+init=EPSG:3857 +units=m +no_defs")(c_long,c_lat)
        print "c_y : ",c_y
        print "c_x : ",c_x
        d = np.sqrt(surface)*1000 # window size in meters 
        print "d : ",d
        print "h,w ndvi : ",ndvi_grey.shape
        x1 = int(float(c_x-d/2-ul_x)/(br_x-ul_x)*ndvi_grey.shape[1])
        y1 = int(float(c_y-d/2-ul_y)/(br_y-ul_y)*ndvi_grey.shape[0])
        x2 = int(float(c_x+d/2-ul_x)/(br_x-ul_x)*ndvi_grey.shape[1])
        y2 = int(float(c_y+d/2-ul_y)/(br_y-ul_y)*ndvi_grey.shape[0])
        print "x1 : ",x1
        print "x2 : ",x2
        print "y1 : ",y1
        print "y2 : ",y2
	if ((x1<0) | (x2<0) | (y1<0) | (y2<0) | (x2>ndvi_grey.shape[1]) | (y2>ndvi_grey.shape[0]) ):
		print name+" not in "+ID+"..."
		continue
	zoom = ndvi_grey[y2:y1,x1:x2];
        print "zoom ndvi shape :",zoom.shape

        fig = plt.figure(); 
        cplt=plt.imshow(zoom,cmap=cmap_ndvi,vmin=-1,vmax=1); 
        cbar = fig.colorbar(cplt, ticks=frontiers);
        cbar.ax.set_yticklabels([str(fr) for fr in frontiers])
        plt.savefig(folder+'/'+name+'/'+month+'_ndvi_colormap.png')
        zoom_rescaled = 255*(zoom-(-1))/(1-(-1));
        cv2.imwrite(folder+'/'+name+'/'+month+'_ndvi.png',(255*cmap_ndvi(zoom_rescaled.astype(np.uint8))).astype(np.uint8)[:,:,:3][:,:,::-1]);
        cv2.imwrite(folder+'/'+name+'/'+month+'_rgb.png',zoom_rgb);
        fig = plt.figure();
        bins = np.linspace(-1,1,nbins);
        histo = np.histogram(zoom,bins)[0]
        histo_per_month[unicode(month)] = histo
        plt.plot(bins[:-1],histo)
	print folder+'/'+name+'/'+month+'_ndvi_histo.png'
        plt.savefig(folder+'/'+name+'/'+month+'_ndvi_histo.png')

histo_per_month['density'] = densite
histo_per_month['surface'] = surface
np.save(folder+'/'+name+'/ndvi_histo.npy',histo_per_month)


legend_per_month = {
u'01':[u'Janvier','b-'],u'02':[u'Février','bx'],u'03':[u'Mars','b+'],
u'04':[u'Avril','g-'],u'05':[u'Mai','gx'],u'06':[u'Juin','g+'],
u'07':[u'Juillet','r-'],u'08':[u'Aout','rx'],u'09':[u'Septembre','r+'],
u'10':[u'Octobre','k-'],u'11':[u'Novembre','kx'],u'12':[u'Décembre','k+']
}

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
for month in ['12','01','02','03','04','05','06','07','08','09','10','11']:
    if month in histo_per_month.keys():
        ax1.plot(bins[:-1],histo_per_month[month],label=legend_per_month[month][0])
colormap = plt.cm.gist_rainbow
colors = [colormap(i) for i in np.linspace(0, 1,len(ax1.lines))]
for i,j in enumerate(ax1.lines):
    j.set_color(colors[i])
ax1.legend(loc=2,fontsize='x-small')
plt.savefig(folder+'/'+name+'/all_ndvi_histo.png')
