import pandas as pd
import numpy as np
import os
import sys
import cv2
import matplotlib.pyplot as plt
from itertools import izip
import utm
from pyproj import Proj
from os.path import expanduser
home = expanduser("~")

file = sys.argv[1]

folder = os.path.dirname(file)
image_names = list(pd.read_csv(file,sep=',',header=0,usecols=[52])['Display ID'])
for ID in image_names:
	print "processing "+folder+'/'+ID+'...'
	if os.path.isdir(folder+'/'+ID) == False:
		os.system('/usr/local/bin/landsat download '+ID+' --bands 45 --dest '+folder)
		if os.path.isfile(folder+'/'+ID+'.tar.bz') == True:
			os.mkdir(folder+'/'+ID)
 			os.system('tar xfvj '+folder+'/'+ID+'.tar.bz -C '+folder+'/'+ID)
			os.system('rm '+folder+'/'+ID+'.tar.bz')
	else:
		os.system('/usr/local/bin/landsat process '+folder+'/'+ID+' --ndvigrey')

	

