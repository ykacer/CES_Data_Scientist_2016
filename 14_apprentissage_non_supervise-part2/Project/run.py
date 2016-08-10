import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from itertools import izip
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
  



def cut_images(image,roi_size_x,roi_size_y,overlap_x,overlap_y,flatten_or_not,mask):
    h,w,c = image.shape
    list_patchs_y = np.asarray([]);    
    list_patchs_x = np.asarray([]);    
    patchs = np.empty((0,roi_size_y*roi_size_x*c)).astype(np.uint8)
    Y,X = np.meshgrid(np.arange(0,h-roi_size_y,roi_size_y-overlap_y),np.arange(0,w-roi_size_x,roi_size_x-overlap_x)
)
    for n,(j,i) in enumerate(izip(Y.flatten(),X.flatten())):
	background = np.unique(mask[j:j+roi_size_y,i:i+roi_size_x])
        if 0 in background:
            patchs = np.append(patchs,image[j:j+roi_size_y,i:i+roi_size_x,:].flatten().astype(np.uint8)[np.newaxis,:], axis=0)
            list_patchs_y = np.append(list_patchs_y,j)
            list_patchs_x = np.append(list_patchs_x,i)
    patchs = patchs.transpose()
    nb_patchs = patchs.shape[1]
    if flatten_or_not == False:
	patchs = np.reshape(patchs,(roi_size_y,roi_size_x,c,nb_patchs))
    return patchs,list_patchs_x,list_patchs_y

def descr_rgb(patchs):
    patchs_preprocessed = np.zeros((patchs.shape[0]*patchs.shape[1]*patchs.shape[2],patchs.shape[3]))
    for n in np.arange(patchs.shape[3]):
	 patchs_preprocessed[:,n] = patchs[:,:,:,n].flatten()
    return patchs_preprocessed

def descr_grad(patchs):
    patchs_preprocessed = np.zeros((patchs.shape[0]*patchs.shape[1],patchs.shape[3])) 
    for n in np.arange(patchs.shape[3]):
        patchs_preprocessed[:,n] = cv2.cvtColor(cv2.Laplacian(np.squeeze(patchs[:,:,:,n].astype(np.uint8)),cv2.CV_8U),cv2.COLOR_RGB2GRAY).flatten()
    return patchs_preprocessed

def descr_hog(patchs):
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    winSize = (patchs.shape[0],patchs.shape[1])
    blockSize = (8,8)
    blockStride = (4,4)
    cellSize = (4,4)
    feature_size = (winSize[0]-(blockSize[0]-blockStride[0]))/blockStride[0]*(winSize[1]-(blockSize[1]-blockStride[1]))/blockStride[1]*blockSize[0]/cellSize[0]*blockSize[1]/cellSize[1]*nbins
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    patchs_preprocessed = np.zeros((feature_size,patchs.shape[3]))
    for n in np.arange(patchs.shape[3]):
        patchs_preprocessed[:,n] = hog.compute(np.squeeze(patchs[:,:,:,n].astype(np.uint8))).flatten()
    return patchs_preprocessed


if os.path.exists('data_papers') == False:
    os.mkdir('data_papers')
    if os.path.exists('dataset_segmentation.rar') == False:
        os.system('wget https://archive.ics.uci.edu/ml/machine-learning-databases/00306/dataset_segmentation.rar')
    os.system('unrar e dataset_segmentation.rar data_papers')

list_mask = glob.glob('data_papers/*_m.*')
flatten_or_not = False
color_mapping = {0:[255,0,0],1:[0,0,255],2:[255,255,255]}
resizing_factor = 4
roi_size_x = 128/resizing_factor
roi_size_y = 128/resizing_factor
overlap_x = 0/resizing_factor
overlap_y = 0/resizing_factor

prediction = np.asarray([])
ground_truth = np.asarray([])

precision = np.zeros(3)
recall = np.zeros(3)

for file_mask in list_mask:
    file_image = glob.glob(file_mask[:-6]+'.*')[0]
    image = cv2.imread(file_image)
    image = cv2.resize(image,(image.shape[1]/resizing_factor,image.shape[0]/resizing_factor))
    print "processing "+file_image+"..."
    gray = cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),(5,5),0)
    #ret,binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret,binary = cv2.threshold(gray,220,255,cv2.THRESH_BINARY)
    mask = cv2.imread(file_mask)
    mask = cv2.resize(mask,(mask.shape[1]/resizing_factor,mask.shape[0]/resizing_factor))
    mask = mask[:,:,::-1]
    h,w,c = image.shape
    patchs,list_patchs_x,list_patchs_y = cut_images(image,roi_size_x,roi_size_y,overlap_x,overlap_y,flatten_or_not,binary)
    patchs_preprocessed = np.vstack((descr_rgb(patchs),descr_hog(patchs)))
    #### NMF
    #nmf = NMF(n_components=3)
    #patchs_transformed = nmf.fit_transform(patchs_preprocessed.transpose())
    #ind = np.argmax(patchs_transformed,axis=1)
    #### KMEANS
    km = KMeans(n_clusters=2)#3)
    ind = km.fit_predict(patchs_preprocessed.transpose())
    
    sorted_labels = np.argsort([np.sum(ind==0),np.sum(ind==1)])#,np.sum(ind==2)])
    color_mapping[sorted_labels[0]] = [255,0,0]
    color_mapping[sorted_labels[1]] = [0,0,255]
    #color_mapping[sorted_labels[2]] = [255,255,255]
    image_result = 255*np.ones((h,w,c))
    for n,(j,i) in enumerate(izip(list_patchs_y,list_patchs_x)):
        image_result[j:j+roi_size_y,i:i+roi_size_x,:] = color_mapping[ind[n]]
    ground_truth_ = np.zeros((h,w))
    ground_truth_ = ground_truth_ + np.where(np.sum(mask-[0,0,255],axis=2)==0,1,0)
    ground_truth_ = ground_truth_ + np.where(np.logical_and(np.sum(mask-[0,0,255],axis=2)!=0,np.sum(mask-[255,0,0],axis=2)!=0),2,0)
    prediction_ = np.zeros((h,w))
    prediction_ = prediction_ + np.where(np.sum(image_result-[0,0,255],axis=2)==0,1,0)
    prediction_ = prediction_ + np.where(np.logical_and(np.sum(image_result-[0,0,255],axis=2)!=0,np.sum(image_result-[255,0,0],axis=2)!=0),2,0)
    ground_truth = np.append(ground_truth,ground_truth_) 
    print ground_truth.shape
    prediction = np.append(prediction,prediction_) 
    print prediction.shape
    f = plt.figure()
    f.add_subplot(1,4,1)
    plt.imshow(image)
    f.add_subplot(1,4,2)
    plt.imshow(binary,cmap=plt.get_cmap('gray'));
    f.add_subplot(1,4,3)
    plt.imshow(mask)
    f.add_subplot(1,4,4)
    plt.imshow(image_result.astype(np.uint8))
    #plt.show()
    f.savefig(file_mask[:-6]+'_res.png', dpi=f.dpi)
for cl in [0,1,2]:
    true_positive = np.sum(np.where(prediction==cl,1,0)==np.where(ground_truth==cl,1,0)) 
    precision[cl] = 1.0*true_positive/np.sum(ground_truth==cl)
    recall[cl] = 1.0*true_positive/np.sum(prediction==cl)

print "precision : ",precision
print "recall : ",recall
