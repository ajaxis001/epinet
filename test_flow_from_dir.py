

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

# img_files = glob.glob(os.path.join(path_raw_img,'epi_imgs', '*.*'))
# mask_files = glob.glob(os.path.join(path_masks, 'mask_imgs','*.*'))

# pprint(img_files)
# print(30*'=')
# pprint(mask_files)


#======================================== Actual Testing of flow_from_directory() ========================================

seed = 1

# Defining random_crop() to do random cropping to get patches randomly from data 
def random_crop(x, **funcvars):
    crop_size = funcvars.pop('crop_size', None)
    if crop_size is None:
        raise ValueError(r'Required variable >crop_size< not defined')
    np.random.seed(seed)

    print('x.shape', x.shape)
    
    rows, cols = x.shape[0], x.shape[1]
    print('img rows : ', rows)
    print('img cols : ', cols)
    
    range_rows = (rows - crop_size[0]) 
    range_cols = (cols - crop_size[1]) 
    offset_rows = 0 if range_rows == 0 else np.random.randint(0,int(range_rows))
    offset_cols = 0 if range_cols == 0 else np.random.randint(0,int(range_cols))
    print('range_rows : ',range_rows)
    print('range_cols : ',range_cols)
    
    # io.imsave('test' + str(seed) + '.jpeg', x[offset_rows:offset_rows+crop_size[0], offset_cols:offset_cols+crop_size[1],:])
    return x[offset_rows:offset_rows+crop_size[0], offset_cols:offset_cols+crop_size[1],:]

crop_size = (patch_rows, patch_cols)
preprocessing_vars = {}
preprocessing_vars['crop_size'] = crop_size

preprocess_on_image_before_autoresize=True
data_gen_args = dict(preprocessing_function=random_crop, 
					 preprocessing_vars=preprocessing_vars, 
					 preprocess_on_image_before_autoresize=preprocess_on_image_before_autoresize)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)




# -------------------------------- For Mask------------------------------------
i = 0
img_flow = image_datagen.flow_from_directory(path_raw_img, 
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
	                                batch_size=10,
	                                class_mode=None,
	                                save_to_dir='preview_ffd',
	                                save_prefix='mask_',
	                                save_format='jpeg',
	                                seed=seed)
y_batch = next(mask_flow)






