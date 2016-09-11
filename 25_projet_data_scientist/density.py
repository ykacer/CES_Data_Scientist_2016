import numpy as np
import pandas as pd
import sys
from itertools import izip
import matplotlib.pyplot as plt
import os
from pyproj import Proj

#file = sys.argv[1]
file = 'France/villes_france.csv'
folder = os.path.dirname(file)
print "processing "+file+"..."

data = np.array(pd.read_csv(file,sep=',',header=0,usecols=[17,19,20],na_values='NULL',keep_default_na=False).dropna(how='any'))
density = data[:,0]
longitude = data[:,1]
latitude = data[:,2]

condition = (longitude<-10) | (latitude<40) | (latitude>55) | (longitude>10)
latitude[condition] = 40
latitude[condition] = 40
longitude[condition] = 0
longitude[condition] = 0
density[condition] = 0
density[condition] = 0

print "-- longitude minimal :",np.min(longitude)
print "-- longitude maximal :",np.max(longitude)
print "-- latitude minimal :",np.min(latitude)
print "-- latitude maximal :",np.max(latitude)

R = 6378137.0 # R in meters
zones = np.arange(1,61,1)
meridians = np.arange(-180,180,6)
ref={}
for l,z in izip(meridians,zones):
    ref[z] = (z-1)*R*6*np.pi/180+R*3*np.pi/180
    ref[z] = 0.75*ref[z]
    #print z,l,Proj("+proj=utm +zone="+str(z)+" +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")(l+3,0),ref[z]
zone_ref = str(zones[np.argmin(np.abs(np.mean(longitude)-meridians))])
print "-- UTM zone for projection : "+zone_ref
print "\n"

x = np.array([Proj("+proj=utm +zone="+zone_ref+", +ellps=WGS84 +datum=WGS84 +units=m +no_defs")(longi,lati)[0] for lati,longi in izip(latitude,longitude)]).astype(np.int)/1000
y = -1*np.array([Proj("+proj=utm +zone="+zone_ref+", +ellps=WGS84 +datum=WGS84 +units=m +no_defs")(longi,lati)[1] for lati,longi in izip(latitude,longitude)]).astype(np.int)/1000
offset_y = np.min(y)
offset_x = np.min(x)

nargin = 200
h = int(np.max(y)-np.min(y)+2*nargin)+1
w = int(np.max(x)-np.min(x)+2*nargin)+1
print "h :",h
print "w :",w

dmin = int(np.min(density))
dmax = int(np.max(density))
print "density maximum : ",dmax
print "density minimum : ",dmin

area = np.zeros((h,w))
s = 1
for i,j,d in izip(x,y,density):
    area[nargin-offset_y+j-s:nargin-offset_y+j+s+1,nargin-offset_x+i-s:nargin-offset_x+i+s+1] = d
area[area>200]=200
fig = plt.figure()
cplt = plt.imshow(area,cmap="YlOrRd"); 
cbar = fig.colorbar(cplt, ticks=[0, 100, 200])
cbar.ax.set_yticklabels(['0', '100', '>200'])
plt.savefig(folder+"/density.png")
plt.show()
