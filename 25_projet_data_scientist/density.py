import numpy as np
import pandas as pd
import sys
from itertools import izip
import matplotlib.pyplot as plt
import os

#file = sys.argv[1]
file = 'France/villes_france.csv'
folder = os.path.dirname(file)
print "processing "+file+"..."

data = np.array(pd.read_csv(file,sep=',',header=0,usecols=[16,19,20],na_values='NULL',keep_default_na=False).dropna(how='any'))
density = data[:,0]
longitude = data[:,1]*np.pi/180.0
latitude = data[:,2]*np.pi/180.0

R = 6370
factor = 4
xmin = int(R*np.min(longitude)/factor)
xmax = int(R*np.max(longitude)/factor)
ymin = int(R*np.min(latitude)/factor)
ymax = int(R*np.max(latitude)/factor)
dmin = int(np.min(density))
dmax = int(np.max(density))
margin = 20

w = 2*margin + xmax-xmin
h = 2*margin + ymax-ymin


area = np.zeros((h,w)).astype(np.uint16)
for n,(i,j) in enumerate(izip(longitude,latitude)):
    x = margin+int(R*i)/factor-xmin 
    y = margin+int(R*j)/factor-ymin 
    d = int(65535.0*density[n]/dmax)
    area[y,x] = d

plt.imshow(area); plt.show(folder+'/density-france.png')



