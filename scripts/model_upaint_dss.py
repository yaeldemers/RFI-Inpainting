"""
model_upaint_v2.py

Description: Class implemetation of the UPAINT CNN model using DSS layers instead of convolution blocks.

Authors: Yael-Jeremy Demers

Last Modified: 23-12-2022

Changes:
- 23-08-2023 Demers: Converting convolution blocks to DSS + FC layer
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Average
from tensorflow import keras
from tensorflow.keras.optimizers import *
from dss_layer import DSS
from fc_layer import CustomFullyConnectedLayer

#TODO: Replace ConvDown2D by maxpooling + dropout

class Unet:
    '''
       Standard U-net, description to come
    '''

    def __init__(self, input_shape, loss_class, weights_dir):
        # inherit from Sequential
        super(Unet, self).__init__()
        
        # we need the shape for the compile step
        self.input_shape = input_shape
        # we need the loss for the compile step, this is just the object though, we'll initialize the
        # actual loss function (which is wrapped) later on
        masked_loss  = loss_class.wrapper()

        self.weights_dir = weights_dir

        # Apparently Sequential does this step automatically for you
        # This is a little less constrained than Sequential, so we need to make our input layer manually
        # this feels a little like a placeholder to me. Sequential must make all the placeholders for you

        input_layer = Input(self.input_shape)

        # label each layer so that we can pass it to the next one
        # let us start with 64 layers, this is going to keep doubling, so maybe as a test subject, we should keep this small. oh well.
        # as per the u-net figure, don't boil down the information yet.

        # Normally Convolution blocks have 2 Conv2D layers but since we have two in DSS's, we only need one DSS here
        dss_1 = DSS(filters=64, name="dss_1-1")([input_layer, input_layer])
        fc_1 = CustomFullyConnectedLayer(units=64, activation='relu')(dss_1)

        # this is not a conv transpose, just another layer but this time has different output, we can initialize this without issue
        convDownSample1 = Conv2D(64, 3, strides = (2,2), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(fc_1)

        #--------- drop a step in the U-pattern --------#

        # half the image size, double this channels
        dss_2 = DSS(filters=128, name="dss_2-1")([convDownSample1, convDownSample1])
        fc_2 = CustomFullyConnectedLayer(units=128, activation='relu')(dss_2)

        convDownSample2 = Conv2D(64, 3, strides = (2,2), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(fc_2)
        # note that the kernel size = 2 is irrelavant for padding = same and no stride. The padding will adjust (and pad less)

        #--------- drop a step in the U-pattern --------#
        dss_3 = DSS(filters=256, name="dss_3-1")([convDownSample2, convDownSample2])
        fc_3 = CustomFullyConnectedLayer(units=256, activation='relu')(dss_3)

        convDownSample3 = Conv2D(64, 3, strides = (2,2), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(fc_3)

        #--------- drop a step in the U-pattern --------#
        dss_4 = DSS(filters=512, name="dss_4-1")([convDownSample3, convDownSample3])
        fc_4 = CustomFullyConnectedLayer(units=512, activation='relu')(dss_4)

        Dropout1 = Dropout(0.5)(fc_4)
        convDownSample4 = Conv2D(64, 3, strides = (2,2), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(Dropout1)

        #--------- drop a step in the U-pattern --------#
        #***********THIS IS THE BOTTOM OF THE U, INSTEAD OF CONV2D with STRIDE, WERE 2D TRANSPOSE WITH STRIDE
        dss_bottleneck = DSS(filters=1024, name="dss_5-1")([convDownSample4, convDownSample4])
        fc_bottleneck = CustomFullyConnectedLayer(units=1024, activation='relu')(dss_bottleneck)

        
        Dropout2  = Dropout(0.5)(fc_bottleneck)
        ConvUpSample1 = Conv2DTranspose(64, 2, strides = (2,2), kernel_initializer = 'he_normal' )(Dropout2)
        
        #--------- Go up a step in the U-pattern --------#
        # concatenate along the channel direction
        # add a layer before the concatenation to make sure they are the same size
        ConvL9 = Conv2D(512, 2, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal' )(ConvUpSample1)
        Merge1 = concatenate( [Dropout1,ConvL9], axis = 3)
    
        #After concatenation, go back to using the same channel direction
        dss_4_up = DSS(filters=512, name="dss_4-1-up")([Merge1, Merge1])
        fc_4_up = CustomFullyConnectedLayer(units=512, activation='relu')(dss_4_up)        
        
        #--------- Go up a step in the U-pattern --------#
        ConvUpSample2 = Conv2DTranspose(64, 2, strides = (2,2))(fc_4_up)
        ConvL12 = Conv2D(256, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(ConvUpSample2)
        Merge2 = concatenate([fc_3, ConvL12], axis = 3)
        dss_3_up = DSS(filters=256, name="dss_3-1-up")([Merge2, Merge2])
        fc_3_up = CustomFullyConnectedLayer(units=256, activation='relu')(dss_3_up)

        #--------- Go up a step in the U-pattern --------#
        ConvUpSample3 = Conv2DTranspose(64, 2, strides = (2,2), padding = 'same')(fc_3_up)
        ConvL15 = Conv2D(128, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(ConvUpSample3)
        Merge3 = concatenate([fc_2, ConvL15], axis = 3)
        dss_2_up = DSS(filters=128, name="dss_2-1-up")([Merge3, Merge3])
        fc_2_up = CustomFullyConnectedLayer(units=128, activation='relu')(dss_2_up)

        #--------- Go up a step in the U-pattern --------#
        ConvUpSample4 = Conv2DTranspose(64, 3, strides = (2,2), padding = 'same')(fc_2_up)
        ConvL18 = Conv2D(64, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(ConvUpSample4)
        Merge4 = concatenate([fc_1, ConvL18], axis = 3)
        dss_1_up = DSS(filters=64, name="dss_1-1-up")([Merge4, Merge4])
        fc_1_up = CustomFullyConnectedLayer(units=64, activation='relu')(dss_1_up)


        # mold it to the final shape that we want
        ConvL21 = Conv2D(3, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(fc_1_up)

        # fully connected layer
        # this doesn't change the shape of the inbetween dimensions

        dense1 = Dense(256, activation='relu')(ConvL21)
        dense2 = Dense(2)(dense1)

        # this makes an object of the Model class, takes the place of the Sequential line in the CNN
        # reads the outputs and compiles all the layers
        self.model = Model(inputs = input_layer, outputs = dense2)
        
        if self.weights_dir != None:
            try:
                print('Loading weights' )
                self.model.load_weights(self.weights_dir)
            except:
                pass
                print('No weights to load' , flush = True)
            #self.model.load_weights(self.weights)​
        self.model.compile(optimizer = Adam(learning_rate = 1e-4), loss = masked_loss , metrics = [ masked_loss ])

        #self.model.summary()​​​​

'''
    note to future self, the concatenation takes place along the channel direction, so we thicken the channels and then down sample that using the usual 2D conv. doesn't change image shape. This downsampling in the channel direction is trivial: you can sample N channels using N/2 channels, you need only change the last dimension
'''