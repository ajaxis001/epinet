

from keras.preprocessing.image import ImageDataGenerator

import os
import sys
# import glob2 as glob
import glob as glob

from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt  

from skimage import io

from utilities_episeg import *

dirname = os.path.dirname(__file__)

# Specifying paths (with the epithelium images and the segmentation mask images)
path_raw_img = os.path.join('Images50','Images')# the raw images (same resolutiona as the svs)
path_masks = os.path.join('Images50','Mask') # the segmentation mask

print('path to images : ', path_raw_img)
print('path to masks : ', path_masks)

# We will create smaller image patches over the image
patch_rows = 256
patch_cols = 256
patch_step = 100 # sets the number of pixels between start of one patch and the start of the succeeding patch


# Extension for the images
img_extension = 'tif'

img_files = glob.glob(os.path.join(path_raw_img, '*.*'))
mask_files = glob.glob(os.path.join(path_masks, '*.*'))

pprint(img_files)
print(30*'=')
pprint(mask_files)


#======================================== Actual Testing of flow_from_directory() ========================================

seed = 1

data_gen_args = dict()
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Defining random_crop() to do random cropping to get patches randomly from data 
def random_crop(x, random_crop_size, sync_seed=None, **kwargs):
    np.random.seed(sync_seed)
    w, h = x.shape[1], x.shape[2]
    rangew = (w - random_crop_size[0]) // 2
    rangeh = (h - random_crop_size[1]) // 2
    offsetw = 0 if rangew == 0 else np.random.randint(rangew)
    offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
    return x[:, offsetw:offsetw+random_crop_size[0], offseth:offseth+random_crop_size[1]]

image_datagen.config['random_crop_size'] = (patch_rows, patch_cols)
mask_datagen.config['random_crop_size'] = (patch_rows, patch_cols)

image_datagen.set_pipeline([random_crop])
mask_datagen.set_pipeline([random_crop])


# -------------------------------- For Mask------------------------------------
i = 0
img_flow = image_datagen.flow_from_directory(path_raw_img, 
		                            target_size=(patch_rows,patch_cols),
		                            batch_size=10,
		                            class_mode=None,
		                            save_to_dir='preview_ffd',
		                            save_prefix='img_',
		                            save_format='jpeg',
		                            seed=seed)

x_batch = next(img_flow)

# -------------------------------- For Mask------------------------------------
i = 0
mask_flow = mask_datagen.flow_from_directory(path_masks, 
	                                target_size=(patch_rows,patch_cols),
	                                batch_size=10,
	                                class_mode=None,
	                                save_to_dir='preview_ffd',
	                                save_prefix='mask_',
	                                save_format='jpeg',
	                                seed=seed)
y_batch = next(mask_flow)






