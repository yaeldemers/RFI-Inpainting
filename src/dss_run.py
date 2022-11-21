import numpy as np
import datetime
import random
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow import keras
from keras import backend
from tensorflow.keras.optimizers import *

class custom_loss:
    def wrapper(self):

        def masked_loss(Y_true, Y_pred):
            '''
                This loss function takes the prediction Y_pred, applies the mask to the first two channels
                which correspond to the real and imag part of the visibilities, then computes the 'chi^2'
                of those two channels.
            '''

            '''
            Uncomment if 0 -> flag and 1 -> not flagged
            
            ones = tf.ones_like(Y_true[:,:,:,2])
            #invert the mask, we want only the masked areas to enter the chi^2
            mask_array = ones - Y_true[:,:,:,2]
            '''
            #mask_array = Y_true[:,:,:,2]
            #mask_array = tf.ones_like(Y_true[:,:,:,2]) - Y_true[:,:,:,2]
            mask_array = Y_true[:,:,:,2]

            Y_pred_real = tf.math.multiply(tf.cast(Y_pred[:,:,:,0] , tf.float64) , tf.cast(mask_array[:,:,:], tf.float64))
            Y_true_real= tf.math.multiply(tf.cast(Y_true[:,:,:,0], tf.float64) , tf.cast(mask_array[:,:,:], tf.float64))
            
            Y_pred_imag = tf.math.multiply(tf.cast(Y_pred[:,:,:,1], tf.float64), tf.cast(mask_array[:,:,:], tf.float64))
            Y_true_imag = tf.math.multiply(tf.cast(Y_true[:,:,:,1], tf.float64), tf.cast(mask_array[:,:,:], tf.float64))
            
            
            ground_truth_reconstructed = tf.complex(Y_true_real, Y_true_imag)
            
            predictions_reconstructed = tf.complex(Y_pred_real, Y_pred_imag)
            
            chi = ground_truth_reconstructed - predictions_reconstructed
            
            chi2 = tf.math.conj(chi) * chi
            
            return tf.math.real(tf.math.reduce_sum(chi2))

        return masked_loss
    
    
class DSS(keras.layers.Layer):
    def __init__(self, filters, name=None):
        super(DSS, self).__init__()
        self.conv = Conv2D(filters, kernel_size=3, padding='same', activation="relu", kernel_initializer='he_normal')
        self.add = Add()
        if name: self._name = name

    def call(self, inputs):
        input_a, input_b = inputs
        
        conv_1a = self.conv(input_a)
        
        conv_1b = self.conv(input_b)
        
        merged = self.add([input_a, input_b])
        
        conv_2 = self.conv(merged)
        
        output_a = self.add([conv_1a, conv_2])
        
        output_b = self.add([conv_1b, conv_2])
        
        return output_a, output_b
    
    
#TODO: put this in a class with proper constructer for better instantiation following OOP
#TODO: consider weights

#--------- Getting input, noise for now --------#
input_a = Input(np.random.normal(0, 1, size = (512, 512, 3)).shape, name="input_a")
input_b = Input(np.random.normal(0, 1, size = (512, 512, 3)).shape, name="input_b")

#--------- Initial step of the U-pattern -------#
dss_1 = DSS(filters=64, name="dss_1-1")([input_a, input_b])
dss_1 = DSS(filters=64, name="dss_1-2")([dss_1[0], dss_1[1]])
pool_1a = MaxPooling2D((2,2), name="maxpool_1a")(dss_1[0])
pool_1a = Dropout(0.5, name="dropout_1a")(pool_1a)
pool_1b = MaxPooling2D((2,2), name="maxpool_1b")(dss_1[1])
pool_1b = Dropout(0.5, name="dropout_1b")(pool_1b)

#--------- drop a step in the U-pattern --------#
dss_2 = DSS(filters=128, name="dss_2-1")([pool_1a, pool_1b])
dss_2 = DSS(filters=128, name="dss_2-2")([dss_2[0], dss_2[1]])
pool_2a = MaxPooling2D((2,2), name="maxpool_2a")(dss_2[0])
pool_2a = Dropout(0.5, name="dropout_2a")(pool_2a)
pool_2b = MaxPooling2D((2,2), name="maxpool_2b")(dss_2[1])
pool_2b = Dropout(0.5, name="dropout_2b")(pool_2b)


#--------- drop a step in the U-pattern --------#
dss_3 = DSS(filters=256, name="dss_3-1")([pool_2a, pool_2b])
dss_3 = DSS(filters=256, name="dss_3-2")([dss_3[0], dss_3[1]])
pool_3a = MaxPooling2D((2,2), name="maxpool_3a")(dss_3[0])
pool_3a = Dropout(0.5, name="dropout_3a")(pool_3a)
pool_3b = MaxPooling2D((2,2), name="maxpool_3b")(dss_3[1])
pool_3b = Dropout(0.5, name="dropout_3b")(pool_3b)


#--------- drop a step in the U-pattern --------#
dss_4 = DSS(filters=512, name="dss_4-1")([pool_3a, pool_3b])
dss_4 = DSS(filters=512, name="dss_4-2")([dss_4[0], dss_4[1]])
pool_4a = MaxPooling2D((2,2), name="maxpool_4a")(dss_4[0])
pool_4a = Dropout(0.5, name="dropout_4a")(pool_4a)
pool_4b = MaxPooling2D((2,2), name="maxpool_4b")(dss_4[1])
pool_4b = Dropout(0.5, name="dropout_4b")(pool_4b)


#--------- drop a step in the U-pattern --------#
#---------    bottom of the U-pattern   --------#
dss_bottleneck = DSS(filters=1024, name="dss_bot-1")([pool_4a, pool_4b])
dss_bottleneck = DSS(filters=1024, name="dss_bot-2")([dss_bottleneck[0], dss_bottleneck[1]])


#--------- climb a step in the U-pattern -------#
deconv_4a = Conv2DTranspose(512, 2, strides=(2,2), kernel_initializer='he_normal', name="deconv_4a")(dss_bottleneck[0])
deconv_4b = Conv2DTranspose(512, 2, strides=(2,2), kernel_initializer='he_normal', name="deconv_4b")(dss_bottleneck[1])
uconv_4a = concatenate([deconv_4a, dss_4[0]], name="concat_4a")
uconv_4a = Dropout(0.5, name="dropout_up_4a")(uconv_4a)
uconv_4b = concatenate([deconv_4b, dss_4[1]], name="concat_4b")
uconv_4b = Dropout(0.5, name="dropout_up_4b")(uconv_4b)
dss_up_4 = DSS(filters=512, name="updss_4-1")([uconv_4a, uconv_4b])
dss_up_4 = DSS(filters=512, name="updss_4-2")([dss_up_4[0], dss_up_4[1]])


#--------- climb a step in the U-pattern -------#
deconv_3a = Conv2DTranspose(256, 2, strides=(2,2), kernel_initializer='he_normal', name="deconv_3a")(dss_up_4[0])
deconv_3b = Conv2DTranspose(256, 2, strides=(2,2), kernel_initializer='he_normal', name="deconv_3b")(dss_up_4[1])
uconv_3a = concatenate([deconv_3a, dss_3[0]], name="concat_3a")
uconv_3a = Dropout(0.5, name="dropout_up_3a")(uconv_3a)
uconv_3b = concatenate([deconv_3b, dss_3[1]], name="concat_3b")
uconv_3b = Dropout(0.5, name="dropout_up_3b")(uconv_3b)
dss_up_3 = DSS(filters=256, name="updss_3-1")([uconv_3a, uconv_3b])
dss_up_3 = DSS(filters=256, name="updss_3-2")([dss_up_3[0], dss_up_3[1]])


#--------- climb a step in the U-pattern -------#
deconv_2a = Conv2DTranspose(128, 2, strides=(2,2), kernel_initializer='he_normal', name="deconv_2a")(dss_up_3[0])
deconv_2b = Conv2DTranspose(128, 2, strides=(2,2), kernel_initializer='he_normal', name="deconv_2b")(dss_up_3[1])
uconv_2a = concatenate([deconv_2a, dss_2[0]], name="concat_2a")
uconv_2a = Dropout(0.5, name="dropout_up_2a")(uconv_2a)
uconv_2b = concatenate([deconv_2b, dss_2[1]], name="concat_2b")
uconv_2b = Dropout(0.5, name="dropout_up_2b")(uconv_2b)
dss_up_2 = DSS(filters=128, name="updss_2-1")([uconv_2a, uconv_2b])
dss_up_2 = DSS(filters=128, name="updss_2-2")([dss_up_2[0], dss_up_2[1]])


#--------- climb a step in the U-pattern -------#
#---------     top of the U-pattern     --------#
deconv_1a = Conv2DTranspose(64, 2, strides=(2,2), kernel_initializer='he_normal', name="deconv_1a")(dss_up_2[0])
deconv_1b = Conv2DTranspose(64, 2, strides=(2,2), kernel_initializer='he_normal', name="deconv_1b")(dss_up_2[1])
uconv_1a = concatenate([deconv_1a, dss_1[0]], name="concat_1a")
uconv_1a = Dropout(0.5, name="dropout_up_1a")(uconv_1a)
uconv_1b = concatenate([deconv_1b, dss_1[1]], name="concat_1b")
uconv_1b = Dropout(0.5, name="dropout_up_1b")(uconv_1b)
dss_up_1 = DSS(filters=64, name="updss_1-1")([uconv_1a, uconv_1b])
dss_up_1 = DSS(filters=64, name="updss_1-2")([dss_up_1[0], dss_up_1[1]])


#-------- reshaping the data to fit inputs ------#
output_a = Conv2D(3, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal', name="output_a")(dss_up_1[0])
output_b = Conv2D(3, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal', name="output_b")(dss_up_1[1])


CNN = Model([input_a, input_b], [output_a, output_b])
CNN._name = "U-NET-V0.4"

#CNN.summary()
#keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
#keras.utils.plot_model(CNN, "multi_input_and_output_model.png")


# Generating sample data to test the network
x_test = np.random.normal(0, 1, size = (5, 512, 512, 3)) # data validation

x_train = np.random.normal(0, 1, size = (5, 512, 512, 3)) # data

y_train = np.random.normal(0, 1, size = (5 ,512, 512, 3)) # labels

# Creating path where networl progress is saved
checkpoint_path = '../latest.hdf5'


# Creating an instance of the loss class
loss = custom_loss()  #the loss needs to know the shape of the batch size too

modelcheckpoint = ModelCheckpoint(save_best_only = True, save_weights_only = True, verbose = 1, filepath = checkpoint_path, monitor = 'val_loss')


# compiling network
masked_loss = custom_loss().wrapper()
CNN.compile(optimizer = Adam(learning_rate = 1e-4), loss = masked_loss , metrics = [ masked_loss ])

# fitting the model
CNN_obj.model.fit([x_train, x_train], [y_train, y_train], batch_size = 5, epochs = 80, callbacks = [callback_list], validation_split = 0.1)

print('Done, moving to predictions', flush = True)

# Making predictions
predictions = CNN.model.predict([x_test[:, :, :], x_test[:, :, :]])