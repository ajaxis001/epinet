

from keras.preprocessing.image import ImageDataGenerator

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
path_raw_img = os.path.join('Images50','epi_imgs')# the raw images (same resolutiona as the svs)
path_masks = os.path.join('Images50','Mask') # the segmentation mask

# We will create smaller image patches over the image
patch_rows = 256
patch_cols = 256
patch_step = 100 # sets the number of pixels between start of one patch and the start of the succeeding patch


# Extension for the images
img_extension = 'tif'

img_files = glob.glob(os.path.join(path_raw_img, '*.*'))
mask_files = glob.glob(os.path.join(path_masks, '*.*'))

quit()

#======================================== Actual Testing of flow_from_directory() ========================================

seed = 1

data_gen_args = dict(rotation_range=90.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# -------------------------------- For Mask------------------------------------
i = 0
for batch in image_datagen.flow_from_directory(training_batch_img_folder, 
                                                target_size=(patch_rows,patch_cols),
                                                batch_size=10,
                                                class_mode=None,
                                                save_to_dir='preview_ffd',
                                                save_prefix='img_',
                                                save_format='jpeg',
                                                seed=seed):

    i+=1
    if i > 10:
        break

# -------------------------------- For Mask------------------------------------
i = 0
for batch in image_datagen.flow_from_directory(training_batch_img_folder, 
                                                target_size=(patch_rows,patch_cols),
                                                batch_size=10,
                                                class_mode=None,
                                                save_to_dir='preview_ffd',
                                                save_prefix='mask_',
                                                save_format='jpeg',
                                                seed=seed):

    i+=1
    if i > 10:
        break



# combine the two generators into one which yields both image and masks
train_gen = zip(image_datagen,mask_datagen)







