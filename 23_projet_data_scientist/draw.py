import pandas as pd
import numpy as np
import os
import sys
import cv2
import matplotlib.pyplot as plt

file = sys.argv[1]
folder = os.path.dirname(file)
print "processing "+file+"..."

image_names = list(pd.read_csv(file,sep=',',header=0,usecols=[52])['Display ID']) 
image_coords = np.array(pd.read_csv(file,sep=',',header=0,usecols=np.arange(42,52)))*np.pi/180

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

R = 6730
x1 = (R*np.cos(d2*image_coords[:,2])*np.cos(image_coords[:,9])).astype(np.int)
y1 = (R*np.cos(d2*image_coords[:,2])*np.sin(image_coords[:,9])).astype(np.int)
x2 = (R*np.cos(d6*image_coords[:,6])*np.cos(image_coords[:,5])).astype(np.int)
y2 = (R*np.cos(d6*image_coords[:,6])*np.sin(image_coords[:,5])).astype(np.int)

#x1 = (R*image_coords[:,9]).astype(np.int)
#y1 = (R*d2*image_coords[:,2]).astype(np.int)
#x2 = (R*image_coords[:,5]).astype(np.int)
#y2 = (R*d6*image_coords[:,6]).astype(np.int)

offset_y = np.min(y1)
offset_x = np.min(x1)

nargin = 50
h = int(np.max(y2)-np.min(y1)+2*nargin)+1
w = int(np.max(x2)-np.min(x1)+2*nargin)+1
draw = np.zeros((h,w,3))
mask = np.zeros((h,w))

for i,filename in enumerate(image_names):
    temp = np.zeros((h,w,3))
    anchor_x = nargin+x1[i]-offset_x
    anchor_y = nargin+y1[i]-offset_y
    image = cv2.resize(cv2.imread(folder+'/'+filename+".jpg",-1),(200,200))
    hi,wi,ci = image.shape
    temp[anchor_y:anchor_y+hi,anchor_x:anchor_x+wi,:] = image[:,:,::-1]
    draw = draw + temp
    draw[draw>255]=255
plt.imshow(draw); 
plt.savefig(folder+'/covering-selection.png')
plt.show()



