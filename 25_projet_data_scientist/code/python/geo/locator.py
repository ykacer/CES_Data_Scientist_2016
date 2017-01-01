# coding: utf-8
import codecs
import time
from geopy.geocoders import GoogleV3
import geopy
import numpy as np
import pandas as pd
import sys
from itertools import izip

geolocatorv3 = GoogleV3()

input_file = sys.argv[1]
output_file = sys.argv[2]
separator = sys.argv[3]
if len(sys.argv)>4:
    country = sys.argv[4]
else:
    country = ""


if len(sys.argv)>5:
    region_header = sys.argv[5]
else:
    region_header = ""

df = pd.read_csv(input_file,sep=separator,encoding='utf8')
df['LAT']=0.0
df['LONG']=0.0

names = df['LIBMIN'].tolist()
if region_header != "":
    regions = df[region_header].tolist()
else:
    regions = list(" "*len(names))

inc = 0
for iter in np.arange(200000):
	try:
		for n,r in izip(names[inc:],regions[inc:]):
			location = geolocatorv3.geocode(n+u' '+r+u' '+country,timeout=30)
			if location==None:
				latitude = "NaN"
				longitude = "NaN"
			else:
				latitude = location.latitude
				longitude = location.longitude
			print n+"-"+r+" ("+str(latitude)+","+str(longitude)+")";
			df = df.set_value(inc, 'LAT', latitude);
			df = df.set_value(inc, 'LONG', longitude);
			inc = inc+1
	except geopy.exc.GeocoderServiceError:
		print str(inc) +u' cities currently geolocalized'
		print 'Now waiting authorization for more queries...'
		time.sleep(300)

df.to_csv(output_file,encoding='utf8')
