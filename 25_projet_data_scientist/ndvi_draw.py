# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle

month = unicode(sys.argv[1]);
data = {};
densities = [];

for row_path in sys.argv[2:]:
	for city in os.listdir(row_path):
        	if os.path.isdir(row_path+'/'+city) and 'LC8' not in city:
			histo_file = row_path+'/'+city+'/ndvi_histo.npy'
			if os.path.isfile(histo_file):
            			ndvi = np.load(row_path+'/'+city+'/ndvi_histo.npy').item()
				if 'surface' in ndvi.keys() and 'density' in ndvi.keys():
            				surface = ndvi['surface']
            				density = ndvi['density']
            				curve = ndvi[month]
            				print 'city:',city
	            			print 'surface:',surface
       		     			print 'density:',density
					print '\n'
       		     			densities.append(density)
            				data[density] = [city,curve,surface]

sorted_densities = np.sort(densities)
nbins = np.linspace(-1,1,curve.size)
for s in sorted_densities:
	curve = data[s][1]
	name = data[s][0]
	print s,name
	plt.plot(nbins,curve,label=unicode(name)+u' ('+str(s)+u' habs/km²)')

name_month = {
        u'01':u'Janvier',u'02':u'Février',u'03':u'Mars',
        u'04':u'Avril',u'05':u'Mai',u'06':u'Juin',
        u'07':u'Juillet',u'08':u'Aout',u'09':u'Septembre',
        u'10':u'Octobre',u'11':u'Novembre',u'12':u'Décembre'
        }
plt.legend(loc=2,fontsize='x-small')
pickle.dump(plt.subplot(111), file(name_month[month]+'.pickle', 'w'))
plt.savefig(name_month[month]+'.png')
