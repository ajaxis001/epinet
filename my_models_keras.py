# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 23:20:28 2018

@author: akn36d
"""

from kearas.models import Model
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D




###############################################################################
#                                  MODEL 1  (U-net)                           #
###############################################################################
class epinet_model1_cnn():
    def __init__(self, img_rows = 50, img_cols = 50, img_channels = 3):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_channels - img_channels
        
    # Ref: http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
    def get_model(self): 
        
        ############################ CONTRACTING PATH #########################
        # First conv section of contracting path 
        # ---------------------------------------------------------------------
        inpt = Input((self.img_rows,self.img_cols,self.img_channels))
        conv1 = Conv2D(filters=64,
                       kernel_size=3,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(inpt)
        print("conv1 shape: ", conv1.shape)
        conv1 = Conv2D(filters=64,
                       kernel_size=3,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(conv1)
        print("conv1 shape: ", conv1.shape)
        
        
        # ---------------------------------------------------------------------        
        # Do Max-pooling
        pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
        print("pool1 shape: ",pool1.shape)
        
        
        # ---------------------------------------------------------------------        
        # Second conv section of contracting path
        conv2 = Conv2D(filters=128,
                       kernel_size=3,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(pool1)
        print("conv2 shape: ", conv2.shape)
        conv2 = Conv2D(filters=128,
                       kernel_size=3,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(conv2)
        print("conv2 shape: ", conv2.shape)
        
        
        # ---------------------------------------------------------------------        
        # Do Max-pooling
        pool2 = MaxPooling2D(pool_size=(2,2))(conv2)
        print("pool2 shape: ",pool2.shape)
        
        
        # ---------------------------------------------------------------------        
        # Third conv section of contracting path
        conv3 = Conv2D(filters=256,
                       kernel_size=3,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(pool2)
        print("conv3 shape: ", conv3.shape)
        conv3 = Conv2D(filters=256,
                       kernel_size=3,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(conv3)
        print("conv3 shape: ", conv3.shape)
        
        
        # ---------------------------------------------------------------------        
        # Do Max-pooling
        pool3 = MaxPooling2D(pool_size=(2,2))(conv3)
        print("pool3 shape: ",pool3.shape)
        
        
        # ---------------------------------------------------------------------        
        # Fourth conv section of contracting path
        conv4 = Conv2D(filters=512,
                       kernel_size=3,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(pool3)
        print("conv4 shape: ", conv4.shape)
        conv4 = Conv2D(filters=512,
                       kernel_size=3,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(conv4)
        print("conv4 shape: ", conv4.shape)
        drop4 = Dropout(0.5)(conv4)
        
        
        # ---------------------------------------------------------------------        
        # Do Max-pooling
        pool4 = MaxPooling2D(pool_size=(2,2))(drop4)
        print("pool4 shape: ",pool4.shape)
        
        
        # ---------------------------------------------------------------------   
        # Fifth conv section of contracting path
        conv5 = Conv2D(filters=1024,
                       kernel_size=3,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(pool4)
        print("conv5 shape: ", conv5.shape)
        conv5 = Conv2D(filters=1024,
                       kernel_size=3,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(conv5)
        print("conv5 shape: ", conv5.shape)
        drop5 = Dropout(0.5)(conv5)
        # ---------------------------------------------------------------------   
        
        
        ############################# EXPANDING PATH ##########################
        # ---------------------------------------------------------------------
        # Do Upsampling 
        up6 = Conv2D(filters=512,
                     kernel_size=2,
                     activation='relu',
                     padding='same',
                     kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(drop5))
        
        
        # ---------------------------------------------------------------------
        # Fisrt conv and merge in expanding path
        merge6 = merge([drop4,up6],
                       mode='concat',
                       concat_axis=3)
        conv6 = Conv2D(filters=512,
                       kernel_size=2,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(merge6)
        covn6 = Conv2D(filters=512,
                       kernel_size=2,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(conv6)


        # ---------------------------------------------------------------------
        # Do Upsampling 
        up7 = Conv2D(filters=256,
                     kernel_size=2,
                     activation='relu',
                     padding='same',
                     kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv6))
        
        
        # ---------------------------------------------------------------------
        # Second conv and merge in expanding path
        merge7 = merge([conv3,up7],
                       mode='concat',
                       concat_axis=3)
        conv7 = Conv2D(filters=256,
                       kernel_size=2,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(merge7)
        covn7 = Conv2D(filters=256,
                       kernel_size=2,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(conv7)

        # ---------------------------------------------------------------------
        # Do Upsampling 
        up8 = Conv2D(filters=128,
                     kernel_size=2,
                     activation='relu',
                     padding='same',
                     kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv7))
        
        
        # ---------------------------------------------------------------------
        # Third conv and merge in expanding path
        merge8 = merge([conv2,up8],
                       mode='concat',
                       concat_axis=3)
        conv8 = Conv2D(filters=128,
                       kernel_size=2,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(merge8)
        covn8 = Conv2D(filters=128,
                       kernel_size=2,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(conv8)                       

        # ---------------------------------------------------------------------
        # Do Upsampling 
        up9 = Conv2D(filters=64,
                     kernel_size=2,
                     activation='relu',
                     padding='same',
                     kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv7))
        
        
        # ---------------------------------------------------------------------
        # Fourth conv and merge in expanding path
        merge9 = merge([conv2,up8],
                       mode='concat',
                       concat_axis=3)
        conv9 = Conv2D(filters=64,
                       kernel_size=2,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(merge9)
        covn9 = Conv2D(filters=64,
                       kernel_size=2,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(conv9)                       
        conv9 = Conv2D(filters=2,
                       kernel_size=3,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(conv9)
                       

        # ---------------------------------------------------------------------
        # Final Output                             
        conv9 = Conv2D(filters=1,
                       kernel_size=1,
                       activation='relu',
                       padding='same',
                       kernel_initializer='he_normal')(conv9)
        # ---------------------------------------------------------------------

        model = Model(input=inpt, output=conv9)

        return model