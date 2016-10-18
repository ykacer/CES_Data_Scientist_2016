# coding: utf-8



import numpy as np
import matplotlib.pyplot as plt
import os
import sys


month = unicode(sys.argv[1]);
data = {};
densities = [];

for row_path in sys.argv[2:]:
    for city in os.listdir(row_path):
        if os.path.isdir(row_path+'/'+city) and 'LC8' not in city:
            ndvi = np.load(row_path+'/'+city+'/ndvi_histo.npy').item()
            surface = ndvi['surface']
            density = ndvi['density']
            curve = ndvi[month]
            print 'city:',city
            print 'surface:',surface
            print 'density:',density
            densities.append(density)
            data[density] = [city,curve,surface]

sorted_densities = np.sort(densities)
nbins = np.linspace(-1,1,curve.size)
print sorted_densities
for s in sorted_densities:
    curve = data[s][1]
    name = data[s][0]
    print s,name
    plt.plot(nbins,curve,label=unicode(name)+u' ('+str(s)+u' habs/km\^2)')

plt.legend(loc=2,fontsize='medium')
plt.show()


