import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import izip
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from scipy import misc


def cut_images(image,roi_size_x,roi_size_y,overlap_x,overlap_y,flatten_or_not,mask):
    h,w,c = image.shape
    list_patchs_y = np.asarray([]);
    list_patchs_x = np.asarray([]);
    patchs = np.empty((0,roi_size_y*roi_size_x*c)).astype(np.uint8)
    Y,X = np.meshgrid(np.arange(0,h-roi_size_y,roi_size_y-overlap_y),np.arange(0,w-roi_size_x,roi_size_x-overlap_x)
)
    for n,(j,i) in enumerate(izip(Y.flatten(),X.flatten())):
        #background = np.unique(mask[j:j+roi_size_y,i:i+roi_size_x])
        background = 1.0*np.sum(mask[j:j+roi_size_y,i:i+roi_size_x]==0)/roi_size_x/roi_size_y
        #if 0 in background:
        if background>0.05:
            patchs = np.append(patchs,image[j:j+roi_size_y,i:i+roi_size_x,:].flatten().astype(np.uint8)[np.newaxis,:], axis=0)
            list_patchs_y = np.append(list_patchs_y,j)
            list_patchs_x = np.append(list_patchs_x,i)
    patchs = patchs.transpose()
    nb_patchs = patchs.shape[1]
    if flatten_or_not == False:
        patchs = np.reshape(patchs,(roi_size_y,roi_size_x,c,nb_patchs))
    return patchs,list_patchs_x,list_patchs_y

def plot_histo(img,filename,hsv=False):
    if hsv:
        imgt = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    else:
        imgt = img
    bins = np.arange(256)
    f = plt.figure()
    f.add_subplot(1,4,1)
    plt.imshow(img)
    f.add_subplot(1,4,2)
    plt.bar(bins[:-1],np.histogram(imgt[:,:,0],bins)[0])
    if hsv:
        plt.title('hue')
    else:
        plt.title('blue')
    plt.gca().axes.yaxis.set_ticklabels([])
    plt.tick_params(labelsize=8)
    f.add_subplot(1,4,3)
    plt.bar(bins[:-1],np.histogram(imgt[:,:,1],bins)[0])
    if hsv:
        plt.title('saturation')
    else:
        plt.title('green')
    plt.gca().axes.yaxis.set_ticklabels([])
    plt.tick_params(labelsize=8)
    f.add_subplot(1,4,4)
    plt.bar(bins[:-1],np.histogram(imgt[:,:,2],bins)[0])
    if hsv:
        plt.title('value')
    else:
        plt.title('red')
    plt.gca().axes.yaxis.set_ticklabels([])
    plt.tick_params(labelsize=8)
    st = f.suptitle('histogrammes')
    #plt.show()
    f.savefig(filename)
    f.clf()

