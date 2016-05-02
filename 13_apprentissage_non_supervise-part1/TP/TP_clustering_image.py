# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 13:26:18 2014

@author: salmon
"""

from sklearn import cluster

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

matrix_image=mpimg.imread('Grey_square_optical_illusion.png')

f, axarr = plt.subplots(3,1)
axarr[0].imshow(matrix_image[:,:,0],cmap = plt.get_cmap('gray'))
axarr[0].set_title('Rouge')
axarr[1].imshow(matrix_image[:,:,1],cmap = plt.get_cmap('gray'))
axarr[1].set_title('Vert')
axarr[2].imshow(matrix_image[:,:,2],cmap = plt.get_cmap('gray'))
axarr[2].set_title('Bleu')

plt.savefig('image_channels.png')

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


matrix_gray = rgb2gray(matrix_image)
h,w = matrix_gray.shape
X = matrix_gray.ravel()
k_list = np.arange(2,5)
fig = plt.figure()
matrix_result = np.zeros((h,w))

fig, axes = plt.subplots(nrows=1, ncols=len(k_list))#, figsize=(20,4))
for i,ax in enumerate(axes):
    k = k_list[i]
    print "n_clusters : ",k
    kmeans = cluster.KMeans(n_clusters=k, n_init=10)
    kmeans.fit(X[:,np.newaxis])
    labels = kmeans.labels_.reshape((h,w))
    matrix_result = labels.astype(np.float)/(k-1)*255
    ax.set_title("K = "+str(k))
    ax.imshow(matrix_result,cmap='gray', aspect='equal', interpolation='nearest')
plt.axis('off')
plt.savefig('image_clustering.png')

