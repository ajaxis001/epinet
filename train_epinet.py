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

from utilities_episeg import *


# Setting folders to store the batches of .npy files that will be generated
training_batch_img_folder = os.path.join('Train_batches','images') 
training_batch_label_folder = os.path.join('Train_batches','labels') 

# Number of batches of training data
tr_data_batches = glob.glob(os.path.join(training_batch_img_folder, '*.npy'))
tr_label_batches = glob.glob(os.path.join(training_batch_label_folder, '*.npy'))

#pprint.pprint(tr_data_batches)
#pprint.pprint(tr_label_batches)


# testing on a single working file
idx = 1 
tr_data =  np.load(os.path.join( training_batch_img_folder ,'tr_data_batch_' + str(idx) + '.npy'))
tr_label = np.load(os.path.join(training_batch_label_folder ,'tr_label_batch_'+ str(idx) +  '.npy'))

