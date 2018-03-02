# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 21:30:18 2018
@author: akn36d


"""

import os
import sys
import glob2 as glob

import pprint
import numpy as np
import matplotlib.pyplot as plt  

from skimage import io
from sklearn.model_selection import train_test_split

from utilities_episeg import *


# Setting folders to store the batches of .npy files that will be generated
training_batch_img_folder = os.path.join('Train_batches','images')  
training_batch_label_folder = os.path.join('Train_batches','labels') 

# Number of batches of training data
tr_data_batches = glob.glob(os.path.join(training_batch_img_folder, '*.npy'))
tr_label_batches = glob.glob(os.path.join(training_batch_label_folder, '*.npy'))



# pprint.pprint(tr_data_batches)
# pprint.pprint(tr_label_batches)

number_of_batches = len(tr_data_batches) 
         
mode = 'train'
# test training using a single trdata and trlabel npy files 
for idx in range(1): # range(number_of_batches) 
    
    # Preprocesing / extracting data 
    train_data_labels = ready_data(tr_data_batches[idx], tr_label_batches[idx], mode)
    
    X_data = train_data_labels['X_data']
    y_data = train_data_labels['Y_data']    
    
    # Splitting training data into training and validation data (stratified cross validation)
    val_per = 0.20 # ratio of training data to be taken for validation
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data,
                                                      test_size=val_per,
                                                      stratify=y_data)
    
    # Declaring the model
    
    
   
