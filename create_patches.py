"""
Created on Fri Feb 23 17:46:46 2018
@author: akn36d

Script to create overlapping patches of given training images and corresponding patches of the labelled truth.
They will be stored in two seperate batches of .npy files (each in a seperate folder) one for the training images and one for the labelled truth.
These files can later be loaded for training of the model. 

We use batches of .npy for the images because a single .npy file with all  the training patches(or labelled truth patches) 
will be too big and will give out of memory errors.

"""

import os
import sys
# import glob2 as glob
import glob as glob

import pprint
import numpy as np
import matplotlib.pyplot as plt  

from skimage import io

from utilities_episeg import *

# Specifying paths (with the epithelium images and the segmentation mask images)
path_raw_img = os.path.join('Images50','img_val')# the raw images (same resolutiona as the svs)
path_masks = os.path.join('Images50','mask_val') # the segmentation mask


# We will create smaller image patches over the image
patch_rows = 256
patch_cols = 256
patch_step = 100 # sets the number of pixels between start of one patch and the start of the succeeding patch

number_of_batches = 0

val_per = 100

# Setting folders to store the batches of .npy files that will be generated
validation_batch_img_folder = os.path.join('Val_batches','images_'+ str(patch_rows) + '_' + str(patch_step) )
validation_batch_label_folder = os.path.join('Val_batches','labels_'+ str(patch_rows) + '_' + str(patch_step)) 

# make the folders if they dont exist 
makefolder_ifnotexists(validation_batch_img_folder)
makefolder_ifnotexists(validation_batch_label_folder)


print('The batches (validation) will be stored in the following folders : ')
print(validation_batch_img_folder)
print(validation_batch_label_folder)

# Extension for the images
img_extension = 'tif'

training_batch_img_folder = validation_batch_img_folder
training_batch_label_folder = validation_batch_label_folder

batch_patchCreateSave_v2(path_raw_img, path_masks,
                      training_batch_img_folder, training_batch_label_folder, 
                      validation_batch_img_folder, validation_batch_label_folder, 
                      number_of_batches,
                      patch_rows, patch_cols, patch_step, 
                      img_extension, 
                      val_per)
        
print('Stored data and labels (validation) in folders : \n' + validation_batch_img_folder + '\n' + validation_batch_label_folder)    
        
    
    

    
    
