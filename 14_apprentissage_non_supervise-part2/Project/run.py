import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import izip
from sklearn.decomposition import PCA

def cut_images(image,roi_size_x,roi_size_y,overlap_x,overlap_y,flatten_or_not):
    h,w,c = image.shape
    list_patchs_x = np.arange(0,w-roi_size_x,roi_size_x-overlap_x)
    list_patchs_y = np.arange(0,h-roi_size_y,roi_size_y-overlap_y)
    nb_patchs = list_patchs_x.size*list_patchs_y.size
    if flatten_or_not:
        patchs = np.zeros((roi_size_x*roi_size_y*c,nb_patchs))
    else:
        patchs = np.zeros((roi_size_y,roi_size_x,c,nb_patchs))
    Y,X = np.meshgrid(list_patchs_y,list_patchs_x)
    for n,(j,i) in enumerate(izip(Y.flatten(),X.flatten())):
        if flatten_or_not:
            patchs[:,n] = image[j:j+roi_size_y,i:i+roi_size_x,:].flatten()
        else:
            patchs[:,:,:,n] = image[j:j+roi_size_y,i:i+roi_size_x,:]
    return patchs,list_patchs_x,list_patchs_y


 
if os.path.exists('data_papers') == False:
    os.mkdir('data_papers')
    if os.path.exists('dataset_segmentation.rar') == False:
        os.system('wget https://archive.ics.uci.edu/ml/machine-learning-databases/00306/dataset_segmentation.rar')
    os.system('unrar e dataset_segmentation.rar data_papers')

list_mask = glob.glob('data_papers/*_m.*')
roi_size_x = 50
roi_size_y = 50
overlap_x = 0
overlap_y = 0
flatten_or_not = True

for file_mask in list_mask:
    file_image = glob.glob(file_mask[:-6]+'.*')[0]
    image = cv2.imread(file_image)
    mask = cv2.imread(file_mask)
    patchs,list_patchs_x,list_patchs_y = cut_images(image,roi_size_x,roi_size_y,overlap_x,overlap_y,flatten_or_not)
    nmf = NMF(n_components=3)
    nmf.fit(patchs.transpose())
    patchs_transformed = nmf.transform(patchs.transpose())
    ind = np.argmax(patchs_transformed,axis=1)
    image_result = np.zeros((h,w,c))
    X,Y = np.meshgrid(list_patchs_x,list_patchs_y)
    for n,(j,i) in enumerate(izip(Y.flatten(),Y.flatten())):
        image_result[j:j+roi_size_y,i+i+roi_size_x] = ind[n]
    plt.figure()
    f.add_subplot(1,3,1)
    plt.imshow(image)
    f.add_subplot(1,3,2)
    plt.imshow(mask)
    f.add_subplot(1,3,3)
    plt.imshow(image_result)
    plt.show()

