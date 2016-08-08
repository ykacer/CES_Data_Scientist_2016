import os
import glob
import cv2
import matplotlib.pyplot as plt

if os.path.exists('data_papers') == False:
    os.mkdir('data_papers')
    if os.path.exists('dataset_segmentation.rar') == False:
        os.system('wget https://archive.ics.uci.edu/ml/machine-learning-databases/00306/dataset_segmentation.rar')
    os.system('unrar e dataset_segmentation.rar data_papers')

list_mask = glob.glob('data_papers/*_m.*')
for file_mask in list_mask:
    file_image = glob.glob(file_mask[:-6]+'.*')[0]
    image = cv2.imread(file_image)
    mask = cv2.imread(file_mask)
    f = plt.figure()
    f.add_subplot(1,2,1)
    plt.imshow(image)
    f.add_subplot(1,2,2)
    plt.imshow(mask)
    plt.show()

    
