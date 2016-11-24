# -*- coding: utf-8 -*-
import codecs
import re
import unicodedata

f = codecs.open('Suisse/Portraits-regionaux-2015-communes.TXT','r','utf8')
A = f.readlines();

fsuisse = codecs.open('Suisse/population_surface_suisse.csv','w','utf8')
lines_suisse = []; lines_suisse.append(u'LIBMIN,PMUN13,SURFACE,DENSITE\n')
for i,l in enumerate(A):
    	if l==u'Population\r\n':
		lines_suisse.append(A[i-2][:-2]+u','+re.sub(u' ',u'',A[i+8][:-2])+u','+re.sub(u',',u'.',A[i+92][:-2])+u','+re.sub(u',',u'.',A[i+20][:-2])+u'\n');

for i,l in enumerate(lines_suisse[1:]):
	lines_suisse[i+1] = re.sub(u' ',u'',unicodedata.normalize("NFKD",l))

for ll in lines_suisse:
	fsuisse.write(ll);

fsuisse.close()
