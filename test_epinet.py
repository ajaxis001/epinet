import os
import sys
#import glob2 as glob
import glob as glob
import re
import pprint

import numpy as np
import matplotlib.pyplot as plt  

from skimage import io


from utilities_episeg import *
from Models.unet_model1 import *

# Setting the random number generator so that the results are reproducible
np.random.seed(123)

# get directory script resides in
dirname = os.path.dirname(__file__)
print(dirname)

# Suite Name and directory declaration (The place where all the info for a given run of a model is stored)
models_folder = 'Model_runs'

# Suites to test
test_suite1 = 'model1_run1'
test_suite2 = 'model1_run2'
suite_dirname = test_suite1

# Size of image patches we y=used for training 
patch_rows = 256
patch_cols = 256
patch_step = 100 # The number of pixels between start of one patch and the start of the succeeding patch


# Specifying paths (with the epithelium images and the segmentation mask images)
path_test_img = os.path.join('Images62','epi_imgs_test')# the raw images (same resolutiona as the svs)
path_test_masks = os.path.join('Images62','mask_test') # the segmentation mask

# The batches of training data
tst_data_imgs = sorted(glob.glob(os.path.join(dirname, path_test_img,'*.jpg')), 
                         key=os.path.getmtime)
tst_label_imgs = sorted(glob.glob(os.path.join(dirname, path_test_masks,'*.tif')),
                          key=os.path.getmtime)

#pprint.pprint(tst_data_imgs)
#pprint.pprint(tst_label_imgs)

# get number of channels in images (using a test image)
_,_,img_channels = np.shape(io.imread(tst_data_imgs[0]))

# Loading model and saved model weights
test_model =  unet_model1(img_rows=patch_rows, 
                          img_cols=patch_cols,
                          img_channels=img_channels)
epinet = test_model.get_model()

# Loading the weights of model for suite
saved_weights_to_path = 'weights'
load_weights_name =   'model_weights_idx_13.h5'
load_weights_full_path = os.path.join(dirname,
                                      models_folder,
                                      suite_dirname,
                                      saved_weights_to_path,
                                      load_weights_name)
epinet.load_weights(load_weights_full_path) 
print('Loaded weights from : ', load_weights_full_path)



for idx in range(1): # range(len(path_test_img)):
    tst_img = io.imread(tst_data_imgs[idx])
    tst_out = epinet.predict(tst_img)
    io.imsave('test_out.tif', tst_out)
    io.imsave('test_img.tif', tst_img)
    






 



