import pandas as pd
import numpy as np
import os
import sys
import cv2
import matplotlib.pyplot as plt
from itertools import izip
import utm

file = sys.argv[1]
folder = os.path.dirname(file)
print "processing "+file+"..."

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
d0 = np.where(direction0=='N',-1,1)
d1 = np.where(direction1=='W',-1,1)
d2 = np.where(direction2=='N',-1,1)
d3 = np.where(direction3=='W',-1,1)
d4 = np.where(direction4=='N',-1,1)
d5 = np.where(direction5=='W',-1,1)
d6 = np.where(direction6=='N',-1,1)
d7 = np.where(direction7=='W',-1,1)
d8 = np.where(direction8=='N',-1,1)
d9 = np.where(direction9=='W',-1,1)

cloud_cover = np.array(pd.read_csv(file,sep=',',header=0,usecols=[23]))
print "maximum cloud covering : "+str(np.max(cloud_cover))+"%"
print "minimum cloud covering : "+str(np.min(cloud_cover))+"%"

latitude1 = image_coords[:,2]
longitude1 = image_coords[:,9]
latitude2 = image_coords[:,6]
longitude2 = image_coords[:,5]

R = 6370

x1 = (R*np.cos(d2*image_coords[:,2])*np.cos(image_coords[:,9])).astype(np.int)
y1 = (R*np.cos(d2*image_coords[:,2])*np.sin(image_coords[:,9])).astype(np.int)
x2 = (R*np.cos(d6*image_coords[:,6])*np.cos(image_coords[:,5])).astype(np.int)
y2 = (R*np.cos(d6*image_coords[:,6])*np.sin(image_coords[:,5])).astype(np.int)

x1 = (R*np.sin(image_coords[:,9])).astype(np.int)
y1 = (R*d2*np.sin(image_coords[:,2])).astype(np.int)
x2 = (R*np.sin(image_coords[:,5])).astype(np.int)
y2 = (R*d6*np.sin(image_coords[:,6])).astype(np.int)


lambert1 = 0.5*np.log10((1+np.sin(latitude1))/(1-np.sin(latitude1)))-0.08248325676/2*np.log10((1+(0.08248325676*np.sin(latitude1)))/(1-(0.08248325676*np.sin(latitude1))));
R = 11745793.39 * np.exp(-0.7289686274 * lambert1);
gamma = 0.7289686274 * (longitude1 - 0.040792344);
x1 = (600000.0 + R*np.sin(gamma)) / 1000;
y1 = (2000000.0 + 6199695.768 - R*np.cos(gamma)) / 1000; 
y1 = -1*y1

lambert2 = 0.5*np.log10((1+np.sin(latitude2))/(1-np.sin(latitude2)))-0.08248325676/2*np.log10((1+(0.08248325676*np.sin(latitude2)))/(1-(0.08248325676*np.sin(latitude2))));
R = 11745793.39 * np.exp(-0.7289686274 * lambert2);
gamma = 0.7289686274 * (longitude2 - 0.040792344);
x2 = (600000.0 + R*np.sin(gamma)) / 1000;
y2 = (2000000.0 + 6199695.768 - R*np.cos(gamma)) / 1000; 
y2 = -1*y2

x1 = np.array([utm.from_latlon(lati, longi)[0] for lati,longi in izip(latitude1,longitude1)]).astype(int)
y1 = np.array([utm.from_latlon(lati, longi)[1] for lati,longi in izip(latitude1,longitude1)]).astype(int)
x2 = np.array([utm.from_latlon(lati, longi)[0] for lati,longi in izip(latitude2,longitude2)]).astype(int)
y2 = np.array([utm.from_latlon(lati, longi)[1] for lati,longi in izip(latitude2,longitude2)]).astype(int)
x1 = x1/1000
y1 = -y1/1000
x2 = x2/1000
y2 = -y2/1000

offset_y = np.min(y1)
offset_x = np.min(x1)
print np.min(x1)
print np.min(y1)
print np.max(x2)
print np.max(y2)

nargin = 500
h = int(np.max(y2)-np.min(y1)+2*nargin)+1
w = int(np.max(x2)-np.min(x1)+2*nargin)+1
print "h:",h
print "w:",w
draw = np.zeros((h,w,3))
scaling = 5
resolution = 30 # meters

for i,filename in enumerate(image_names):
    temp = np.zeros((h,w,3))
    anchor_x = nargin+x1[i]-offset_x
    anchor_y = nargin+y1[i]-offset_y
    image = cv2.imread(folder+'/'+filename+".jpg",-1)
    hi,wi,ci = image.shape
    hs = hi*scaling*resolution/1000
    ws = wi*scaling*resolution/1000
    image = cv2.resize(cv2.imread(folder+'/'+filename+".jpg",-1),(ws,hs))
    temp[anchor_y:anchor_y+hs,anchor_x:anchor_x+ws,:] = image[:,:,::-1]
    draw = draw + temp
    draw[draw>255]=255
plt.imshow(draw); 
plt.savefig(folder+'/covering-selection.png')
plt.show()



