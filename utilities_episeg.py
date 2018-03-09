# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 15:00:19 2018
@author: akn36d
"""
import os
import sys
import warnings
import re
# import glob2 as glob
import glob as glob

import numpy as np
import matplotlib.pyplot as plt  
import random 

from skimage import io

'''-------------------------------------------------------------------------------------------
'''
def makefolder_ifnotexists(foldername):
    if not os.path.exists(foldername):
        os.makedirs(foldername)


'''-------------------------------------------------------------------------------------------
'''
def create_patch_arr(img, patch_rows,patch_cols,patch_step):
    
    # If any dimension of patch bigger than image dimension throw error
    img_rows, img_cols, img_channels = np.shape(img) # size of image
    assert (img_rows > patch_rows), "Patch row size greater than image row size"
    assert (img_rows > patch_rows), "Patch col size greater than image col size"
    warnings.warn ("\nStep size too big along image rows i.e. patch_rows + patch_step > img_rows") 
    warnings.warn("\nStep size too big along image cols i.e. patch_cols + patch_step > img_cols")
    
    
    
    # Loop to generate image and corresponding mask patches
    # run loop as long as patches dont overstep the image boundaries along height or the width of the image
    r_k = 0 # counter along width (rows)
    row_edge = False
    im_patch_arr = np.empty((1,patch_rows, patch_cols, img_channels)) # array of patches to be used as training data
    
    while row_edge == False: # check if the patch along the rows has reached the edge of the image
        r_idx = (r_k * patch_step) # the row number of the top left pixel in the image patch  
        
        # if edge patch along image heught overstepping img_height then pull it back so 
        # that the edge patch just covers image 
        if( r_idx + patch_rows > img_rows ):
            r_idx = img_rows - patch_rows
            row_edge = True
        
        c_k = 0 # counter along height (cols)
        col_edge = False
        
        while col_edge == False:  # check if the patch along the columns has reached the edge of the image
            c_idx = (c_k * patch_step)  # the column number of the top left pixel in the image patch 
                    
            # if edge patch along image width overstepping img_width then pull it back so 
            # that the edge patch just covers image 
            if( c_idx + patch_cols > img_cols ):
                c_idx = img_cols - patch_cols
                col_edge = True
                       
            im_patch = img[ r_idx:r_idx+patch_rows, c_idx:c_idx+patch_cols, :]
            
            # Creating the array of patch images
            im_patch = im_patch[np.newaxis,:,:,:]
            im_patch_arr = np.concatenate((im_patch_arr,im_patch), axis=0) 
            
    #        # Save image patches for visualization (Ref:repl.it)
    #        with warnings.catch_warnings():
    #            warnings.simplefilter("ignore")
    #            io.imsave(os.path.join('patch_test','patch_'+ str(r_k) + '_' + str(c_k) + '.jpeg'), im_patch)    
                
            if(c_idx + patch_cols == img_cols ):
                col_edge = True    
                
            c_k = c_k + 1
            
                
        if(r_idx + patch_rows == img_rows ):
            row_edge = True    
            
        r_k = r_k + 1   
    im_patch_arr = im_patch_arr[1:]
    return im_patch_arr


'''-------------------------------------------------------------------------------------------
Description :
The function is used to create batches from the given training data images and corresponding label data images.
These will be then saved in the folders with names as follows:
    batch_name =      'tr_data(or label)_batch_' + str(batch_number)+ '.npy' 
    
NOTE:
    1. For this function make sure your label images are named as :
        label_image_name = name_of_corresponding_image + '_mask' + extension
        
        where extension could be formats compatible with scikit-images 'io' module e.g.:
            'jpeg'
            'tif'
            'png' etc.
            
    2. All Training images need to be of the same format
    3. Labels can only be single channel images

INPUTS:
path_raw_img - path to the images 
path_masks - path to masks
training_batch_img_folder - Location to store the data patch batches 
training_batch_label_folder - Location to store the mask/label patch batches 
patch_rows - number of rows  in patch
patch_cols - number of columns in patch
patch_step - steps to take inbetween patches
img_extension - can be 'tif', 'jpg', 'png' or other formats compatible with scikit-images 'io' module
 
img_extension - can be 'tif', 'jpg', 'png' or other formats compatible with scikit-images 'io' module
        
OUTPUTS:

        
''' 
def batch_patchCreateSave(path_raw_img,path_masks,training_batch_img_folder, training_batch_label_folder, number_of_batches ,patch_rows, patch_cols, patch_step, img_extension):
        
    # List out the images in folder/ path
    img_files = glob.glob(os.path.join(path_raw_img,'*.' + img_extension))
        
    num_imgs_in_batch = np.ceil(len(img_files)/ number_of_batches) # max number of images in a single batch
    
    mask_suffix = '_mask' # add extension to image name to get corresponding mask file name
    
    
    for idx in range(number_of_batches):
        img_batch_start = int(idx * num_imgs_in_batch) 
        img_batch_end = int(img_batch_start + num_imgs_in_batch)
        
        print('\nProcessing batch ' + str(idx) + ' of ' + str(number_of_batches-1))
        
        _,_,img_channels = np.shape(io.imread(img_files[0])) # Getting number of channels in training images
        
        # init first patches because we dont know what the number of patches for a single batch is going to be,
        # this is done specially for the last patch array as mod(number of images, number of batches) might not be equal to 0 
        epi_patch_arr = np.empty((1,patch_rows,patch_cols,img_channels))
        mask_patch_arr = np.empty((1,patch_rows,patch_cols, 1))
        
        for idx_i in range(img_batch_start, img_batch_end):
            print('\tCreating patches for image number : ' + str(idx_i))
            # Storing image into a numpy array
            img = io.imread(img_files[idx_i])
          
            _ , mask_file = os.path.split(img_files[idx_i])
            mask_name , _ = os.path.splitext(mask_file)
            mask_file = os.path.join( path_masks,mask_name + mask_suffix + '.' + img_extension)
            mask = io.imread(mask_file)
        
            mask = mask[...,np.newaxis] # adding an extra dimension, along which we will concatenate the mask patches
            
            if mask is None:
                print('\nErr 0 : Corresponding mask file missing.\n')
            # print(img_files[idx])
            # print(mask_file + '\n')
            # img_arr = io.imread('imagetest.jpg')
            # print(type(img_arr)) #: <class 'numpy.ndarray'>
            # io.imshow(img_arr) # disp imge
            
            data_patches = create_patch_arr(img, patch_rows,patch_cols,patch_step)
            mask_patches = create_patch_arr(mask, patch_rows,patch_cols,patch_step)
            
            epi_patch_arr = np.concatenate((epi_patch_arr,data_patches), axis = 0)
            mask_patch_arr = np.concatenate((mask_patch_arr,mask_patches), axis = 0)
            
        # Removing the the very first patch which was just used for initializing the dynamically increasing patch arrays
        epi_patch_arr = epi_patch_arr[1:,...] 
        mask_patch_arr = mask_patch_arr[1:,...]
        
        print('Train data shape : ' , str(epi_patch_arr.shape))
        print('Train labels shape : ' , str(mask_patch_arr.shape))
       
        np.save(os.path.join( training_batch_img_folder ,'tr_data_batch_' + str(idx) + '.npy'), epi_patch_arr)
        np.save(os.path.join(training_batch_label_folder ,'tr_label_batch_'+ str(idx) +  '.npy'), mask_patch_arr)



'''-------------------------------------------------------------------------------------------
Description :
The function is used to create batches from the given training data images and corresponding label data images.
These will be then saved in the folders with names as follows:
    batch_name =      'tr_data(or label)_batch_' + str(batch_number)+ '.npy' 

This is a better version of the above batch_patchCreateSave(). This randomizes the patches 
in each batch, which serves better for training. 

NOTE:
1. Make sure your np.seed() is set to a value at top of code before you use this function,
otherwise the data patches and mask patches will be shuffled in different ways.
    
2. For this function make sure your label images are named as :
    label_image_name = name_of_corresponding_image + '_mask' + extension
    
    where extension could be formats compatible with scikit-images 'io' module e.g.:
        'jpeg'
        'tif'
        'png' etc.
        
3. All Training images need to be of the same format

4. Labels can only be single channel images

INPUTS:
path_raw_img - path to the images 
path_masks - path to masks
training_batch_img_folder - Location to store the data patch batches 
training_batch_label_folder - Location to store the mask/label patch batches 
patch_rows - number of rows  in patch
patch_cols - number of columns in patch
patch_step - steps to take inbetween patches
img_extension - can be 'tif', 'jpg', 'png' or other formats compatible with scikit-images 'io' module
        
OUTPUTS:

        
''' 
def batch_patchCreateSave_v2(path_raw_img,path_masks,training_batch_img_folder, training_batch_label_folder, number_of_batches ,patch_rows, patch_cols, patch_step, img_extension):
        
    # List out the images in folder/ path
    img_files = glob.glob(os.path.join(path_raw_img,'*.' + img_extension))
    random.shuffle(img_files)
        
    mask_suffix = '_mask' # add extension to image name to get corresponding mask file name
    
              
    _,_,img_channels = np.shape(io.imread(img_files[0])) # Getting number of channels in training images
    
    # init first patches because we dont know what the number of patches for a single batch is going to be,
    # this is done specially for the last patch array as mod(number of images, number of batches) might not be equal to 0 
    epi_patch_arr = np.empty((1,patch_rows,patch_cols,img_channels))
    mask_patch_arr = np.empty((1,patch_rows,patch_cols, 1))
    
    for idx_i in range(len(img_files)):
        print('\tCreating patches for image number : ' + str(idx_i))
        # Storing image into a numpy array
        img = io.imread(img_files[idx_i])
      
        _ , mask_file = os.path.split(img_files[idx_i])
        mask_name , _ = os.path.splitext(mask_file)
        mask_file = os.path.join( path_masks,mask_name + mask_suffix + '.' + img_extension)
        mask = io.imread(mask_file)
    
        mask = mask[...,np.newaxis] # adding an extra dimension, along which we will concatenate the mask patches
        
        if mask is None:
            print('\nErr 0 : Corresponding mask file missing.\n')
        # print(img_files[idx])
        # print(mask_file + '\n')
        # img_arr = io.imread('imagetest.jpg')
        # print(type(img_arr)) #: <class 'numpy.ndarray'>
        # io.imshow(img_arr) # disp imge
        
        data_patches = create_patch_arr(img, patch_rows,patch_cols,patch_step)
        mask_patches = create_patch_arr(mask, patch_rows,patch_cols,patch_step)
        
        epi_patch_arr = np.concatenate((epi_patch_arr,data_patches), axis = 0)
        mask_patch_arr = np.concatenate((mask_patch_arr,mask_patches), axis = 0)
        
    # Removing the the very first patch which was just used for initializing the dynamically increasing patch arrays
    epi_patch_arr = epi_patch_arr[1:,...] 
    np.random.shuffle(epi_patch_arr)

    mask_patch_arr = mask_patch_arr[1:,...]
    np.random.shuffle(mask_patch_arr)

    print('Train data shape : ' , str(epi_patch_arr.shape))
    print('Train labels shape : ' , str(mask_patch_arr.shape))   


    # Number of patches in a single batch
    num_patches_in_batch = int(epi_patch_arr.shape[0]/number_of_batches)

    for idx in np.r_[0 : epi_patch_arr.shape[0] : number_of_batches]:
        print('\nProcessing batch ' + str(idx) + ' of ' + str(number_of_batches-1))

        epi_patch_batch = epi_patch_arr[idx:idx+num_patches_in_batch]
        mask_patch_batch = mask_patch_arr[idx:idx+num_patches_in_batch]

        print('\tSize of current data batch : ', str(epi_patch_batch.shape))
        print('\tSize of current mask batch : ', str(mask_patch_batch.shape))

        np.save(os.path.join( training_batch_img_folder ,'tr_data_batch_' + str(idx) + '.npy'), epi_patch_batch)
        np.save(os.path.join(training_batch_label_folder ,'tr_label_batch_'+ str(idx) +  '.npy'), mask_patch_batch)



'''--------------------------------------------------------------------------------------------
Function takes in the parameters given below and readies the data to be input to given into the 
model of desire

    params :
    path_to_data - a str object to the path of stored batches of data image file patches
        e.g. folder1/folder2/train_data.npy
    path_to_label - astr object to path of stored batches of label file patches
        e.g. folder1/folder2/train_labels.npy
    
    
    mode - a string which specifies if data is training data of test data
    Can be 'test' or 'train'

    returns :
    processed_data - a dictionaty with the processed and readied data
                     has the following labels,
                     X_data     : the trainig data 
                     Y_data     : the labels for the data (only available when mode='train')
                     mode       : the mode with which the function was called in 

'''

def ready_data(path_to_data, path_to_label, mode):
    
    print(mode + " data is in file : " + path_to_data)
    print(mode + " labels are in file : " + path_to_label)

    print('setting up data dictionary...')
    processed_data = {}
    processed_data['X_data'] = np.load(path_to_data)
    processed_data['Y_data'] = np.load(path_to_label)
    processed_data['mode'] = mode

    print(processed_data['X_data'].shape)
    print(processed_data['Y_data'].shape)

    print('processing data...')

    return processed_data