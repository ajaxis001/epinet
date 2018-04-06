# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 21:30:18 2018
@author: akn36d

"""

import os
import sys
#import glob2 as glob
import glob as glob
import re
import pprint

import numpy as np
import matplotlib.pyplot as plt  

from skimage import io
from sklearn.model_selection import train_test_split
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical, normalize

from utilities_episeg import *
from Models.unet_model1 import *


# Setting the random number generator so that the results are reproducible
seed = 123
np.random.seed(seed)

# Random state for train test split
random_state = 42

# get directory script resides in
dirname = os.path.dirname(__file__)
print(dirname)

# Specifying paths (with the epithelium images and the segmentation mask images)
path_raw_img = os.path.join('Images50','Images')# the raw images (same resolutiona as the svs)
path_masks = os.path.join('Images50','Mask') # the segmentation mask

# Suite Name and directory declaration (The place where all the info for a given run of a model will be stored)
models_folder = 'Model_runs'
makefolder_ifnotexists(os.path.join(dirname, 
                                    models_folder))

suite_dirname = 'model1_run4_batchv2'
# suite_dirname = input('\nEnter name for this run of model: ')
makefolder_ifnotexists(os.path.join(dirname, 
                                    models_folder, 
                                    suite_dirname))

# Size of image patches we are using as training data
patch_rows = 256
patch_cols = 256
patch_step = 100 # The number of pixels between start of one patch and the start of the succeeding patch

# The number of batches (this should be same as the last number on your training data/label folder)
number_of_batches = 20

# Setting folders to load the batches of .npy files that will be used
val_batch_img_folder = os.path.join('Val_batches','images_'+ str(patch_rows) + '_' + str(patch_step)) 
val_batch_label_folder = os.path.join('Val_batches','labels_'+ str(patch_rows) + '_' + str(patch_step)) 


print('The validation data folders are : ')
print(val_batch_img_folder)
print(val_batch_label_folder)

# The batches of training data w.r.t to time so that each data batch has the right label batch
val_data_batches = glob.glob(os.path.join(dirname, val_batch_img_folder, '*.npy'))
val_label_batches = glob.glob(os.path.join(dirname, val_batch_label_folder, '*.npy'))

pprint.pprint(val_data_batches)
pprint.pprint(val_label_batches)


# Parameters
loss='binary_crossentropy'
optimizer='adam'
batch_size = 5
epochs = 200
verbose_in_fit = 1
early_stopping_patience = 30

# details of the model which are appended to the log name 
append_to_model_name =    '_ep_'  + str(epochs) \
                        + '_bs_'  + str(batch_size) \
                        + '_esp_' + str(early_stopping_patience)


# log folder for run of model
log_path = 'logs'
makefolder_ifnotexists(os.path.join(dirname,models_folder,suite_dirname, log_path))


# model folder for saving weights in a run of model
save_weights_to_path = 'weights'
makefolder_ifnotexists(os.path.join(dirname,models_folder,suite_dirname ,save_weights_to_path))


mode = 'val'
save_weights_full_path = ''
log_full_path = ''

# Preprocesing / extracting data 
val_data_labels = ready_data(val_data_batches[0], val_label_batches[0], mode)

X_val = val_data_labels['X_data']
y_val = val_data_labels['Y_data'] > 0 # (converting  data  to -> 0 to 1)

print('X_val.shape : ' , X_val.shape)
print('y_val.shape : ' , y_val.shape)

num_val_patches, img_rows, img_cols, img_channels = X_val.shape

# Declaring the model
my_model = unet_model1(img_rows=img_rows,
                             img_cols=img_cols,
                             img_channels=img_channels)
epinet = my_model.get_model()

# Compile model
epinet.compile(loss=loss,
               optimizer='adam',
               metrics=['accuracy'])

# Print the model summary
epinet.summary()

# Save the model accuracy and loss
log_name = 'model.log'
log_full_path = os.path.join(dirname,
                         models_folder,
                         suite_dirname, 
                         log_path, 
                         log_name)
csv_logger_call = CSVLogger(log_full_path)


# Save model for testing purposes
save_weights_name =   'model_weights.h5'
save_weights_full_path = os.path.join(dirname,
                                  models_folder,
                                  suite_dirname,
                                  save_weights_to_path,
                                  save_weights_name)

model_chkpt_call = ModelCheckpoint(save_weights_full_path,
                                   monitor='val_acc',
                                   save_best_only=True,
                                   save_weights_only=True)

# Use early stopping w.r.t val_loss to stop model from overfitting or needlessly training (when bad model)
early_stopping = EarlyStopping(monitor='val_loss', 
                               patience=early_stopping_patience)

# Setting up image generator
crop_size = (patch_rows, patch_cols)
preprocessing_vars = {}
preprocessing_vars['crop_size'] = crop_size
preprocessing_vars['seed'] = seed

preprocess_on_image_before_autoresize=True
data_gen_args = dict(preprocessing_function=random_crop, 
					 preprocessing_vars=preprocessing_vars, 
					 preprocess_on_image_before_autoresize=preprocess_on_image_before_autoresize)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

image_generator = image_datagen.flow_from_directory(
    path_raw_img,
    class_mode=None,
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    path_masks,
    class_mode=None,
    seed=seed)

train_generator = zip(image_generator, mask_generator)

# Train model
epinet.fit_generator(train_generator,
              batch_size=batch_size,
              epochs=epochs,
              verbose=verbose_in_fit,
              callbacks=[csv_logger_call, model_chkpt_call, early_stopping],
              validation_data=(X_val, y_val))

print(save_weights_full_path)
print(log_full_path)
