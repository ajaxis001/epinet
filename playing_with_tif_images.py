# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:57:31 2018

@author: Anand K Nambisan

"""
import os
import glob2 as glob

import warnings
import pprint
import numpy as np
import matplotlib.pyplot as plt

from skimage import io


# Specifying paths
path_raw_img = '.\Images50\epi_imgs' # the raw images (same resolutiona as the svs)
path_masks = '.\Images50\Mask' # the segmentation masks
mask_extension = '_mask' # add extension to image name to get corresponding mask file name

# List out the images in folder/ path
img_files = glob.glob(os.path.join(path_raw_img,'*.tif'), recursive=False)
mask_files = glob.glob(os.path.join(path_masks,'*.tif'), recursive=False)
# pprint.pprint(mask_files)

# Storing image into a numpy array
img_arr = io.imread(img_files[1])
# img_arr = io.imread(r'C:\Users\akn36d\Desktop\OU13-002-001_obj0.jpeg')
# print(type(img_arr)) #: <class 'numpy.ndarray'>
# io.imshow(img_arr) # disp imge
img_height, img_width, _ = np.shape(img_arr)


# We will create smaller image patches over the image
patch_height = 50
patch_width = 50
patch_step = 25 # sets the number of pixels between start of one patch and the start of the succeeding patch

# Loop to generate image and corresponding mask patches
x_lim = int(np.ceil((img_width-1)/patch_step))  # number of patches we will get along the width of the image
y_lim = int(np.ceil((img_height-1)/patch_step)) # number of patches we will get along the height of the image

# run loop as long as patches dont overstep the image boundaries along height or the width of the image
for y_k in np.arange(0,y_lim):
    # if edge patch along image height overstepping img_height then pull it back so 
    # that the edge patch just covers image 
    y_idx = (y_k * patch_step)
    if((y_k * patch_step) + patch_step > img_height):    
        y_idx = img_height - patch_height; 
        
    for x_k in np.arange(0,x_lim):        
        x_idx = (x_k * patch_step) 
        #print('x: ' + str(x_idx) + 'and xk: ' + str(x_k) )
        #print('y: ' + str(y_idx) + 'and yk: ' + str(y_k) + '\n')
        
        # if edge patch along image width overstepping img_width then pull it back so 
        # that the edge patch just covers image 
        if( (x_k * patch_step) + patch_step > img_width):
            # print('x_idx_old : ' + str(x_idx))
            x_idx = img_width - patch_width; 
            # print('x_idx_new : ' + str(x_idx))
        
        im_patch = img_arr[ y_idx:y_idx+patch_height,x_idx:x_idx+patch_width,:]
        # io.imsave(os.path.join('patch_test','patch_'+ str(x_k) + '_' + str(y_k) + '.jpeg'), im_patch)
        
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(os.path.join('patch_test','patch_'+ str(x_k) + '_' + str(y_k) + '.jpeg'), im_patch)
        
   
