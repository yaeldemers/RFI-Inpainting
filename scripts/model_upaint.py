"""
model_upaint.py

Description: Class implemetation of the UPAINT CNN model using Keras subclassing. The model follows a U-Net architecture.

Authors: Michael Pagano, Yael-Jeremy Demers

Last Modified: 26-07-2023

Changes:
- 26-07-2023 Demers: setting the dropout rate back to 50%
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow import keras
from tensorflow.keras.optimizers import *

class DeepCNN(Sequential):
    # this is not a very extensive list right now but let us only require that the user specify the bare
    # minimum. We'll make this model more flexible in the future
    def __init__(self, shape, loss_class, **kwargs):
        super(DeepCNN, self).__init__()
        # make sure RGB images have 3 channels at the end as input
        self.shape = shape
        self.masked_loss = loss_class.wrapper()
        self.optimizer  = kwargs.pop('optimizer', 'adam')
        self.CNN_metrics = kwargs.pop('metrics', 'accuracy')
        
        # initialize the model from tensorflow. Vanilla sequential
        self.model = Sequential()
        # QUESTION: Should BN be done here or in the next class, what does that change?
        # add first layer, convolution layer: 64 layers dense, 3x3 Kernel
        self.model.add(Conv2D(64 , (20,20), padding = 'same', input_shape = self.shape  , name = 'layer1' ))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

        # add second layer
        # why don't I need to initialize these arrays with a bunch of values? (apparently it defaults)
        self.model.add(Conv2D(64, (15,15), padding = 'same', name = 'layer2'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

        # 3rd layer
        # try initializing the arrays within the add method
        self.model.add(Conv2D(64, (15,15), padding = 'same', name = 'layer3'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

        # 4th layer
        self.model.add(Conv2D(64, (10,10), padding = 'same', name = 'layer4'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

        # 5th layer
        self.model.add(Conv2D(64, (6,6), padding = 'same', name = 'layer5'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

        # 6th layer
        self.model.add(Conv2D(64, (4,4), padding = 'same', name = 'layer6'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

        # 7th layer
        self.model.add(Dense(256))
        self.model.add(Activation('relu'))

        # 8th layer
        self.model.add(Dense(3))
        
        self.model.compile(loss = 'binary_crossentropy', optimizer = self.optimizer, metrics = ['accuracy'])
        
        self.model.build()

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
        # TODO fix above
        #masked_loss = loss_class

        self.weights_dir = weights_dir

        # Apparently Sequential does this step automatically for you
        # This is a little less constrained than Sequential, so we need to make our input layer manually
        # this feels a little like a placeholder to me. Sequential must make all the placeholders for you

        input_layer = Input(self.input_shape)

        # label each layer so that we can pass it to the next one
        # let us start with 64 layers, this is going to keep doubling, so maybe as a test subject, we should keep this small. oh well.
        # as per the u-net figure, don't boil down the information yet.
        # ask Jing whether the kernel initializer makes much of a difference
        ConvL1 = Conv2D(64, 3, padding= 'same', activation = 'relu', kernel_initializer = 'he_normal' )(input_layer)
        ConvL2 = Conv2D(64, 3, padding = 'same',  activation = 'relu', kernel_initializer = 'he_normal')(ConvL1)
        #print('ConvL2' , tf.shape(ConvL2))
        keras.backend.shape(ConvL2)
        # this is not a conv transpose, just another layer but this time has different output, we can initialize this without issue
        convDownSample1 = Conv2D(64, 3, strides = (2,2), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(ConvL2)
        #--------- drop a step in the U-pattern --------#
        
        # half the image size, double this channels
        ConvL3 = Conv2D(128, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(convDownSample1)
        ConvL4 = Conv2D(128, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(ConvL3)
        convDownSample2 = Conv2D(64, 3, strides = (2,2), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(ConvL4)
        # note that the kernel size = 2 is irrelavant for padding = same and no stride. The padding will adjust (and pad less)
        #--------- drop a step in the U-pattern --------#
        ConvL5 = Conv2D(256, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(convDownSample2)
        ConvL6 = Conv2D(256, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(ConvL5)
        convDownSample3 = Conv2D(64, 3, strides = (2,2), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(ConvL6)
        
        
        #--------- drop a step in the U-pattern --------#
        ConvL7 = Conv2D(512, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(convDownSample3)
        ConvL8 = Conv2D(512, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(ConvL7)
        Dropout1 = Dropout(0.5)(ConvL8)
        convDownSample4 = Conv2D(64, 3, strides = (2,2), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(Dropout1)

        #--------- drop a step in the U-pattern --------#
        #***********THIS IS THE BOTTOM OF THE U, INSTEAD OF CONV2D with STRIDE, WERE 2D TRANSPOSE WITH STRIDE
        ConvL7 = Conv2D(1024, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(convDownSample4)
        ConvL8 = Conv2D(1024, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(ConvL7)
        Dropout2  = Dropout(0.5)(ConvL8)
        ConvUpSample1 = Conv2DTranspose(64, 2, strides = (2,2), kernel_initializer = 'he_normal' )(Dropout2)
        
        #--------- Go up a step in the U-pattern --------#
        # concatenate along the channel direction
        # add a layer before the concatenation to make sure they are the same size
        ConvL9 = Conv2D(512, 2, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal' )(ConvUpSample1)
        Merge1 = concatenate( [Dropout1,ConvL9], axis = 3)
    
        #After concatenation, go back to using the same channel direction
        ConvL10 = Conv2D(512, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal' )(Merge1)
        ConvL11 = Conv2D(512, 3, padding = 'same', activation = 'relu' , kernel_initializer = 'he_normal')(ConvL10)
        ConvUpSample2 = Conv2DTranspose(64, 2, strides = (2,2))(ConvL11)
        
        #--------- Go up a step in the U-pattern --------#
        ConvL12 = Conv2D(256, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(ConvUpSample2)
        Merge2 = concatenate([ConvL6, ConvL12], axis = 3)
        ConvL13 = Conv2D(256, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(Merge2)
        ConvL14 = Conv2D(256, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(ConvL13)
        ConvUpSample3 = Conv2DTranspose(64, 2, strides = (2,2), padding = 'same')(ConvL14)

        #--------- Go up a step in the U-pattern --------#
        ConvL15 = Conv2D(128, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(ConvUpSample3)
        Merge3 = concatenate([ConvL4, ConvL15], axis = 3)
        ConvL16 = Conv2D(128, 3, padding = 'same', activation = 'relu' , kernel_initializer = 'he_normal')(Merge3)
        ConvL17 = Conv2D(128, 3, padding = 'same', activation = 'relu' , kernel_initializer = 'he_normal')(ConvL16)
        ConvUpSample4 = Conv2DTranspose(64, 3, strides = (2,2), padding = 'same')(ConvL17)

        #--------- Go up a step in the U-pattern --------#
        ConvL18 = Conv2D(64, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(ConvUpSample4)
        Merge4 = concatenate([ConvL2, ConvL18], axis = 3)
        ConvL19 = Conv2D(64, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(Merge4)
        ConvL20 = Conv2D(64, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(ConvL19)

        # mold it to the final shape that we want
        ConvL21 = Conv2D(3, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(ConvL20)

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