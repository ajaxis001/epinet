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
np.random.seed(123)

# Random state for train test split
random_state = 42

# get directory script resides in
dirname = os.path.dirname(__file__)
print(dirname)
# Suite Name and directory declaration (The place where all the info for a given run of a model will be stored)
# MAKE DYNAMIC
models_folder = 'Model_runs'
makefolder_ifnotexists(os.path.join(dirname, 
                                    models_folder))

suite_dirname = 'model1_run1'
# suite_dirname = input('\nEnter name for this run of model: ')
makefolder_ifnotexists(os.path.join(dirname, 
                                    models_folder, 
                                    suite_dirname))

# Size of image patches we are using as training data
patch_rows = 256
patch_cols = 256
patch_step = 100 # The number of pixels between start of one patch and the start of the succeeding patch


# Setting folders to load the batches of .npy files that will be used
training_batch_img_folder = os.path.join('Train_batches','images_'+ str(patch_rows) + '_' + str(patch_step) + '_*') 
training_batch_label_folder = os.path.join('Train_batches','labels_'+ str(patch_rows) + '_' + str(patch_step) + '_*' ) 


# The batches of training data
def numericalSort(value):
    reg_exp = re.compile(r'\d+')
    parts = reg_exp.split(value)
    return parts

tr_data_batches = sorted(glob.glob(os.path.join(dirname, training_batch_img_folder, '*.npy')), 
                         key=numericalSort)
tr_label_batches = sorted(glob.glob(os.path.join(dirname, training_batch_label_folder, '*.npy')),
                          key=numericalSort)
pprint.pprint(tr_data_batches)
pprint.pprint(tr_label_batches)
quit()

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


# Number of batches we split our patches of images into
number_of_batches = len(tr_data_batches) 
print('number_of_batches : ', number_of_batches)   

mode = 'train'
save_weights_full_path = ''
log_full_path = ''
# test training using a single trdata and trlabel npy files 
for idx in range(number_of_batches): 
    
    if idx > 0:
        load_weights_name =   'model_weights_idx_'+ str(idx-1) +'.h5'
        load_weights_full_path = os.path.join(dirname,
                                              models_folder,
                                              suite_dirname,
                                              save_weights_to_path,
                                              load_weights_name)
        epinet.load_weights(load_weights_full_path) 
        print('Loaded weights from training patch batch : ' + str(idx-1)) 

    # Preprocesing / extracting data 
    train_data_labels = ready_data(tr_data_batches[idx], tr_label_batches[idx], mode)
    
    X_data = train_data_labels['X_data']
    num_of_tr_imgs, img_rows, img_cols, img_channels = X_data.shape

    y_data = train_data_labels['Y_data'] > 0 # (converting  data  to -> 0 to 1)
    
    print('X_data.shape : ' , X_data.shape)
    print('y_data.shape : ' , y_data.shape)

    quit()
    # Splitting training data into training and validation data (stratified cross validation)
    val_per = 0.20 # ratio of training data to be taken for validation
    X_train, X_val = train_test_split(X_data,
                                      random_state=random_state,
                                      test_size=val_per)
    y_train, y_val = train_test_split(y_data,
                                      random_state=random_state,
                                      test_size=val_per)
    
    
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
    log_name = 'model_idx_'+ str(idx) +'.log'
    log_full_path = os.path.join(dirname,
                             models_folder,
                             suite_dirname, 
                             log_path, 
                             log_name)
    csv_logger_call = CSVLogger(log_full_path)


    # Save model for testing purposes
    save_weights_name =   'model_weights_idx_'+ str(idx) +'.h5'
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


    # Train model
    epinet.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=verbose_in_fit,
                  callbacks=[csv_logger_call, model_chkpt_call, early_stopping],
                  validation_data=(X_val, y_val))

print(save_weights_full_path)
print(log_full_path)
