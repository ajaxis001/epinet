

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

sync_seed = 10

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

# Defining random_crop() to do random cropping to get patches randomly from data 
def random_crop(x):
    random_crop_size = ()
    np.random.seed(seed)
    rows, cols = x.shape[1], x.shape[2]
    range_rows = (rows - random_crop_size[0]) // 2
    range_cols = (cols - random_crop_size[1]) // 2
    offset_rows = 0 if range_rows == 0 else np.random.randint(range_rows)
    offset_cols = 0 if range_cols == 0 else np.random.randint(range_cols)
    return x[:, offset_rows:offset_rows+random_crop_size[0], offset_cols:offset_cols+random_crop_size[1]]

data_gen_args = dict(preprocessing_function=random_crop)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)




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






