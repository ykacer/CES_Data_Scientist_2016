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

def descr_rgb(patchs):
    patchs_preprocessed = np.zeros((patchs.shape[0]*patchs.shape[1]*patchs.shape[2],patchs.shape[3]))
    for n in np.arange(patchs.shape[3]):
	 patchs_preprocessed[:,n] = patchs[:,:,:,n].flatten()
    return patchs_preprocessed

def descr_hsv(patchs):
    patchs_preprocessed = np.zeros((patchs.shape[0]*patchs.shape[1]*patchs.shape[2],patchs.shape[3]))
    for n in np.arange(patchs.shape[3]):
	 patchs_preprocessed[:,n] = cv2.cvtColor(patchs[:,:,:,n],cv2.COLOR_BGR2HSV).flatten()
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
roi_size_x = 80/resizing_factor
roi_size_y = 80/resizing_factor
overlap_x = 0/resizing_factor
overlap_y = 0/resizing_factor

prediction = np.asarray([])
ground_truth = np.asarray([])

precision = np.zeros(3)
recall = np.zeros(3)

precision_ = np.zeros(3)
recall_ = np.zeros(3)

precision_best = np.zeros(3)
recall_best = np.zeros(3)

precision_worst = np.zeros(3)
recall_worst = np.zeros(3)

precision_mean_best = 0.0
precision_mean_worst = 1.0

recall_mean_best = 0.0
recall_mean_worst = 1.0

filename_precision_best = ""
filename_precision_worst = ""
filename_recall_best = ""
filename_recall_worst = ""

for file_mask in list_mask:
    file_image = glob.glob(file_mask[:-6]+'.*')[0]
    if file_image in ['data_papers/30.jpg','data_papers/33.jpg','data_papers/38.jpg','data_papers/56.jpg','data_papers/49.jpg']:
        continue
    
    # load image
    image = cv2.imread(file_image)

    ### resize image for faster computation
    image = cv2.resize(image,(image.shape[1]/resizing_factor,image.shape[0]/resizing_factor))
    print "processing "+file_image+"..."

    ### binarize image to find patchs corresponding to class 2 (background)
    binary_thresh = 200
    gray = cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),(3,3),0)
    ret,binary = cv2.threshold(gray,binary_thresh,255,cv2.THRESH_BINARY)
    #ret,binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask = cv2.imread(file_mask)
    mask = cv2.resize(mask,(mask.shape[1]/resizing_factor,mask.shape[0]/resizing_factor))
    mask = mask[:,:,::-1]
    h,w,c = image.shape

    ### cut image into patchs, keep only patchs which are not class 2.
    patchs,list_patchs_x,list_patchs_y = cut_images(image,roi_size_x,roi_size_y,overlap_x,overlap_y,flatten_or_not,binary)

    ### extract features of each patch
    #patchs_preprocessed = np.vstack((descr_rgb(patchs),descr_hog(patchs)))
    patchs_preprocessed = np.vstack((descr_hsv(patchs),descr_hog(patchs)))

    ### compute NMF classification to find class for left patchs : class 0 (illustration) or class 1 (texte) 
    #nmf = NMF(n_components=3)
    #patchs_transformed = nmf.fit_transform(patchs_preprocessed.transpose())
    #ind = np.argmax(patchs_transformed,axis=1)
    
    ### compute KMEANS classification to find class for left patchs : class 0 (illustration) or class 1 (texte) 
    km = KMeans(n_clusters=2)#3)
    ind = km.fit_predict(patchs_preprocessed.transpose())
    
    ### patchs are now divide into two categories, give class 0 (illustration) for category in minority, class 1 for the other one.
    #sorted_labels = np.argsort([np.sum(ind==0),np.sum(ind==1)])#,np.sum(ind==2)])
    #color_mapping[sorted_labels[0]] = [255,0,0] # red for class 0 (illustration) 
    #color_mapping[sorted_labels[1]] = [0,0,255] # blue for class 1 (texte) 
    #color_mapping[sorted_labels[2]] = [255,255,255]
    
    ### patchs are now divide into categories, give class 1 (illustration) for category with greater proportion of "white" (i.e. > binary_thresh).
    p0 = 1.0*np.sum(np.squeeze(0.1140*patchs[:,:,0,ind==0]+0.5870*patchs[:,:,1,ind==0]+0.2989*patchs[:,:,2,ind==0]).flatten()>binary_thresh)/patchs.size
    p1 = 1.0*np.sum(np.squeeze(0.1140*patchs[:,:,0,ind==1]+0.5870*patchs[:,:,1,ind==1]+0.2989*patchs[:,:,2,ind==1]).flatten()>binary_thresh)/patchs.size
    sorted_labels = np.argsort([p0,p1])#,np.sum(ind==2)])
    color_mapping[sorted_labels[0]] = [255,0,0] # red for class 0 (illustration) 
    color_mapping[sorted_labels[1]] = [0,0,255] # blue for class 1 (texte) 

    ### compute prediction (class 0,1, or 2) for each pixel
    image_result = 255*np.ones((h,w,c))
    for n,(j,i) in enumerate(izip(list_patchs_y,list_patchs_x)):
        image_result[j:j+roi_size_y,i:i+roi_size_x,:] = color_mapping[ind[n]]
    prediction_   = np.zeros((h,w))
    prediction_   = prediction_   + np.where(np.linalg.norm(image_result-[255,0,0],axis=2)==0,1,0)
    prediction_   = prediction_   + np.where(np.logical_and(np.linalg.norm(image_result-[0,0,255],axis=2)!=0,np.linalg.norm(image_result-[255,0,0],axis=2)!=0),2,0)
    prediction_   = prediction_.flatten()

    ### compute ground-truth (class 0,1, or 2) for each pixel 
    ground_truth_ = np.zeros((h,w))
    ground_truth_ = ground_truth_ + np.where(np.linalg.norm(mask-[255,0,0],axis=2)==0,1,0)
    ground_truth_ = ground_truth_ + np.where(np.logical_and(np.linalg.norm(mask-[0,0,255],axis=2)!=0,np.linalg.norm(mask-[255,0,0],axis=2)!=0),2,0)
    ground_truth_ = ground_truth_.flatten()

    ### compute precision/recall for each class
    for cl in [0,1,2]: 
        true_positive_ = np.sum(np.where(prediction_==cl,1,-5)==np.where(ground_truth_==cl,1,5)) 
        if np.sum(ground_truth_==cl) != 0:
            precision_[cl] = 1.0*true_positive_/np.sum(ground_truth_==cl)
        else:
            precision_[cl] = -1
        if np.sum(prediction_==cl) != 0:
            recall_[cl] = 1.0*true_positive_/np.sum(prediction_==cl)
        else:
            recall_[cl] = -1
    #print "** precision (per class) : ",precision_
    #print "** recall : (per class) ",recall_
    
    precision_mean_ = np.mean(precision_[precision_!=-1])
    recall_mean_ = np.mean(recall_[recall_!=-1])

    #print "** precision (mean) : ",precision_mean_
    #print "** recall (mean) : ",recall_mean_

    if precision_mean_ > precision_mean_best:
        precision_mean_best  = precision_mean_
        precision_best = precision_.copy()
        filename_precision_best = file_image
    if precision_mean_ < precision_mean_worst:
        precision_mean_worst  = precision_mean_
        precision_worst = precision_.copy()
        filename_precision_worst = file_image
    if recall_mean_ > recall_mean_best:
        recall_mean_best  = recall_mean_
        recall_best = recall_.copy()
        filename_recall_best = file_image
    if recall_mean_ < recall_mean_worst:
        recall_mean_worst  = recall_mean_
        recall_worst = recall_.copy()
        filename_recall_worst = file_image

    # gather predictions and ground-truth for latter total precision/recall computation
    ground_truth = np.append(ground_truth,ground_truth_) 
    prediction = np.append(prediction,prediction_) 

    # plot image, binary image (class 0), ground-truth, prediction
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
    f.clf()

# compute total precision/recall
for cl in [0,1,2]:
    true_positive = np.sum(np.where(prediction==cl,1,-5)==np.where(ground_truth==cl,1,5)) 
    precision[cl] = 1.0*true_positive/np.sum(ground_truth==cl)
    recall[cl] = 1.0*true_positive/np.sum(prediction==cl)

print "\n"
print "STATISTICS : "
print"\n"
print "** total precision (per class) : ",precision
print "** total recall (per class): ",recall
print "** total precision (mean) : ",np.mean(precision)
print "** total recall (mean) : ",np.mean(recall)
print"\n"
print filename_precision_best+":"
print "** best precision (per class) : ",precision_best
print "** best precision (mean) : ",precision_mean_best
print filename_recall_best+":"
print "** best recall (per class): ",recall_best
print "** best recall (mean) : ",recall_mean_best
print"\n"
print filename_precision_worst+":"
print "** worst precision (per class) : ",precision_worst
print "** worst precision (mean) : ",precision_mean_worst
print filename_recall_worst+":"
print "** worst recall (per class): ",recall_worst
print "** worst recall (mean) : ",recall_mean_worst
print"\n"
