import pandas as pd
import numpy as np
import os
import sys
import cv2
import matplotlib.pyplot as plt
from itertools import izip
import utm
from pyproj import Proj

file = sys.argv[1]

folder = os.path.dirname(file)
print "processing "+file+"..."
print "\n"
image_names = list(pd.read_csv(file,sep=',',header=0,usecols=[52])['Display ID']) 
image_coords = np.array(pd.read_csv(file,sep=',',header=0,usecols=np.arange(42,52)))#*np.pi/180

direction0 = np.array([ direction[-1] for direction in list(pd.read_csv(file,sep=',',header=0,usecols=[32])['Center Latitude']) ]) 
direction1 = np.array([ direction[-1] for direction in list(pd.read_csv(file,sep=',',header=0,usecols=[33])['Center Longitude']) ])
direction2 = np.array([ direction[-1] for direction in list(pd.read_csv(file,sep=',',header=0,usecols=[34])['NW Corner Lat']) ])
direction3 = np.array([ direction[-1] for direction in list(pd.read_csv(file,sep=',',header=0,usecols=[35])['NW Corner Long']) ])
direction4 = np.array([ direction[-1] for direction in list(pd.read_csv(file,sep=',',header=0,usecols=[36])['NE Corner Lat']) ])
direction5 = np.array([ direction[-1] for direction in list(pd.read_csv(file,sep=',',header=0,usecols=[37])['NE Corner Long']) ])
direction6 = np.array([ direction[-1] for direction in list(pd.read_csv(file,sep=',',header=0,usecols=[38])['SE Corner Lat']) ])
direction7 = np.array([ direction[-1] for direction in list(pd.read_csv(file,sep=',',header=0,usecols=[39])['SE Corner Long']) ])
direction8 = np.array([ direction[-1] for direction in list(pd.read_csv(file,sep=',',header=0,usecols=[40])['SW Corner Lat']) ])
direction9 = np.array([ direction[-1] for direction in list(pd.read_csv(file,sep=',',header=0,usecols=[41])['SW Corner Long']) ])
d0 = np.where(direction0=='N',1,-1)
d1 = np.where(direction1=='W',-1,1)
d2 = np.where(direction2=='N',1,-1)
d3 = np.where(direction3=='W',-1,1)
d4 = np.where(direction4=='N',1,-1)
d5 = np.where(direction5=='W',-1,1)
d6 = np.where(direction6=='N',1,-1)
d7 = np.where(direction7=='W',-1,1)
d8 = np.where(direction8=='N',1,-1)
d9 = np.where(direction9=='W',-1,1)

cloud_cover = np.array(pd.read_csv(file,sep=',',header=0,usecols=[23]))
print "-- maximum cloud covering : "+str(np.max(cloud_cover))+"%"+" ("+image_names[np.argmin(cloud_cover)]+")"
print "-- minimum cloud covering : "+str(np.min(cloud_cover))+"%"+" ("+image_names[np.argmax(cloud_cover)]+")"
print "\n"

latitude1 = image_coords[:,2]
longitude1 = image_coords[:,9]
latitude2 = image_coords[:,6]
longitude2 = image_coords[:,5]
print "-- maximum latitude : "+str(np.max(latitude1))
print "-- minimum latitude : "+str(np.min(latitude1))
print "\n"
print "-- maximum longitude : "+str(np.max(longitude1))
print "-- minimum longitude : "+str(np.min(longitude1))
print "\n"

R = 6378137.0 # R in meters
zones = np.arange(1,61,1)
meridians = np.arange(-180,180,6)
ref={}
for l,z in izip(meridians,zones):
    ref[z] = (z-1)*R*6*np.pi/180+R*3*np.pi/180
    ref[z] = 0.75*ref[z]
    #print z,l,Proj("+proj=utm +zone="+str(z)+" +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")(l+3,0),ref[z]
zone_ref = str(zones[np.argmin(np.abs(np.mean(longitude1)-meridians))])
print "-- UTM zone for projection : "+zone_ref
print "\n"
#x1 = np.array([utm.from_latlon(lati, longi)[0]-500000+ref[utm.from_latlon(lati, longi)[2]] for lati,longi in izip(latitude1,longitude1)]).astype(int)/1000
#y1 = d2*np.array([utm.from_latlon(lati, longi)[1] for lati,longi in izip(latitude1,longitude1)]).astype(int)/1000
#x2 = np.array([utm.from_latlon(lati, longi)[0]-500000+ref[utm.from_latlon(lati, longi)[2]] for lati,longi in izip(latitude2,longitude2)]).astype(int)/1000
#y2 = d6*np.array([utm.from_latlon(lati, longi)[1] for lati,longi in izip(latitude2,longitude2)]).astype(int)/1000


#x1 = np.array([Proj("+proj=utm +zone="+os.popen('cat France/'+filename+'/'+filename+'_MTL.txt | grep ZONE | cut -c 16-17').read()[:-1]+" +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")(long1,lat1)[0] for lat1,long1,filename in izip(latitude1,longitude1,image_names)]).astype(np.int)/1000
#y1 = d2*np.array([Proj("+proj=utm +zone="+os.popen('cat France/'+filename+'/'+filename+'_MTL.txt | grep ZONE | cut -c 16-17').read()[:-1]+" +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")(long1,lat1)[1] for lat1,long1,filename in izip(latitude1,longitude1,image_names)]).astype(np.int)/1000
#x2 = np.array([Proj("+proj=utm +zone="+os.popen('cat France/'+filename+'/'+filename+'_MTL.txt | grep ZONE | cut -c 16-17').read()[:-1]+" +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")(long2,lat2)[0] for lat2,long2,filename in izip(latitude2,longitude2,image_names)]).astype(np.int)/1000
#y2 = d6*np.array([Proj("+proj=utm +zone="+os.popen('cat France/'+filename+'/'+filename+'_MTL.txt | grep ZONE | cut -c 16-17').read()[:-1]+" +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")(long2,lat2)[1] for lat2,long2,filename in izip(latitude2,longitude2,image_names)]).astype(np.int)/1000
projection_value = "+proj=utm +zone="+zone_ref+", +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
projection_value = "epsg:3857"
x1 =    np.array([Proj(init=projection_value)(long1,lat1)[0] for lat1,long1,filename in izip(latitude1,longitude1,image_names)]).astype(np.int)/1000
y1 = -1*np.array([Proj(init=projection_value)(long1,lat1)[1] for lat1,long1,filename in izip(latitude1,longitude1,image_names)]).astype(np.int)/1000
x2 =    np.array([Proj(init=projection_value)(long2,lat2)[0] for lat2,long2,filename in izip(latitude2,longitude2,image_names)]).astype(np.int)/1000
y2 = -1*np.array([Proj(init=projection_value)(long2,lat2)[1] for lat2,long2,filename in izip(latitude2,longitude2,image_names)]).astype(np.int)/1000

#x1 = 100*longitude1
#x1 = 100*image_coords[:,3]
#y1 = 100*d2*latitude1

offset_y = np.min(y1)
offset_x = np.min(x1)
print "min x1 :",np.min(x1)
print "min y1 :",np.min(y1)
print "max x1 :",np.max(x1)
print "max y1 :",np.max(y1)

#for i,filename in enumerate(image_names):
    #x1true = os.popen('cat France/'+filename+'/'+filename+'_MTL.txt | grep CORNER_UL_PROJECTION_X_PRODUCT | cut -c 38-48').read()
    #print x1[i],int(float(x1true[:-1]))/1000
    #x1[i] = int(float(x1true[:-1]))/1000


nargin = 25
h = int(np.max(y2)-np.min(y1)+2*nargin)+1
w = int(np.max(x2)-np.min(x1)+2*nargin)+1
print "h:",h
print "w:",w

draw = np.zeros((h,w,3))
factor = np.ones((h,w,3))
for i,filename in enumerate(image_names):
    temp = np.zeros((h,w,3))
    anchor_x = nargin+x1[i]-offset_x
    anchor_y = nargin+y1[i]-offset_y
    image = cv2.imread(folder+'/'+filename+".jpg",-1)
    if image == None:
        continue
    hi,wi,ci = image.shape
    hs = np.abs(y2[i]-y1[i])
    ws = np.abs(x2[i]-x1[i])
    image = cv2.resize(image,(ws,hs))
    temp[anchor_y:anchor_y+hs,anchor_x:anchor_x+ws,:] = 255-image[:,:,::-1]
    draw = draw + factor*temp
    factor[anchor_y:anchor_y+hs,anchor_x:anchor_x+ws,:] = np.dstack([np.where(cv2.cvtColor(image,cv2.COLOR_BGR2HSV)[:,:,2]<4,1,0)]*3)*factor[anchor_y:anchor_y+hs,anchor_x:anchor_x+ws,:]
plt.imshow(draw); 
plt.savefig(folder+'/covering-selection.png')
plt.show()



