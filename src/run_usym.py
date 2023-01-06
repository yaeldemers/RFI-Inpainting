import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Dropout, Conv2D, MaxPooling2D, Add
from tensorflow import keras
from tensorflow.keras.optimizers import *
from dss_layer import DSS
from misc import prediction_plot, learning_plot, custom_loss
    
#--------- Getting input, noise for now --------#
input_a = Input(np.random.normal(0, 1, size = (128, 128, 3)).shape, name="input_a")
input_b = Input(np.random.normal(0, 1, size = (128, 128, 3)).shape, name="input_b")

#--------- Initial step of the U-pattern -------#
dss_1 = DSS(filters=64, name="dss_1-1")([input_a, input_b])
#dss_1 = DSS(filters=64, name="dss_1-2")([dss_1[0], dss_1[1]])
pool_1a = MaxPooling2D((2,2), name="maxpool_1a")(dss_1[0])
pool_1a = Dropout(0.5, name="dropout_1a")(pool_1a)
pool_1b = MaxPooling2D((2,2), name="maxpool_1b")(dss_1[1])
pool_1b = Dropout(0.5, name="dropout_1b")(pool_1b)

#--------- drop a step in the U-pattern --------#
dss_2 = DSS(filters=128, name="dss_2-1")([pool_1a, pool_1b])
#dss_2 = DSS(filters=128, name="dss_2-2")([dss_2[0], dss_2[1]])
pool_2a = MaxPooling2D((2,2), name="maxpool_2a")(dss_2[0])
pool_2a = Dropout(0.5, name="dropout_2a")(pool_2a)
pool_2b = MaxPooling2D((2,2), name="maxpool_2b")(dss_2[1])
pool_2b = Dropout(0.5, name="dropout_2b")(pool_2b)


#--------- drop a step in the U-pattern --------#
dss_3 = DSS(filters=256, name="dss_3-1")([pool_2a, pool_2b])
#dss_3 = DSS(filters=256, name="dss_3-2")([dss_3[0], dss_3[1]])
pool_3a = MaxPooling2D((2,2), name="maxpool_3a")(dss_3[0])
pool_3a = Dropout(0.5, name="dropout_3a")(pool_3a)
pool_3b = MaxPooling2D((2,2), name="maxpool_3b")(dss_3[1])
pool_3b = Dropout(0.5, name="dropout_3b")(pool_3b)


#--------- drop a step in the U-pattern --------#
dss_4 = DSS(filters=512, name="dss_4-1")([pool_3a, pool_3b])
#dss_4 = DSS(filters=512, name="dss_4-2")([dss_4[0], dss_4[1]])
pool_4a = MaxPooling2D((2,2), name="maxpool_4a")(dss_4[0])
pool_4a = Dropout(0.5, name="dropout_4a")(pool_4a)
pool_4b = MaxPooling2D((2,2), name="maxpool_4b")(dss_4[1])
pool_4b = Dropout(0.5, name="dropout_4b")(pool_4b)


#--------- drop a step in the U-pattern --------#
#---------    bottom of the U-pattern   --------#
dss_bottleneck = DSS(filters=1024, name="dss_bot-1")([pool_4a, pool_4b])
#dss_bottleneck = DSS(filters=1024, name="dss_bot-2")([dss_bottleneck[0], dss_bottleneck[1]])


#--------- climb a step in the U-pattern -------#
deconv_4a = Conv2DTranspose(512, 2, strides=(2,2), kernel_initializer='he_normal', name="deconv_4a")(dss_bottleneck[0])
deconv_4b = Conv2DTranspose(512, 2, strides=(2,2), kernel_initializer='he_normal', name="deconv_4b")(dss_bottleneck[1])
uconv_4a = concatenate([deconv_4a, dss_4[0]], name="concat_4a")
uconv_4a = Dropout(0.5, name="dropout_up_4a")(uconv_4a)
uconv_4b = concatenate([deconv_4b, dss_4[1]], name="concat_4b")
uconv_4b = Dropout(0.5, name="dropout_up_4b")(uconv_4b)
dss_up_4 = DSS(filters=512, name="updss_4-1")([uconv_4a, uconv_4b])
#dss_up_4 = DSS(filters=512, name="updss_4-2")([dss_up_4[0], dss_up_4[1]])


#--------- climb a step in the U-pattern -------#
deconv_3a = Conv2DTranspose(256, 2, strides=(2,2), kernel_initializer='he_normal', name="deconv_3a")(dss_up_4[0])
deconv_3b = Conv2DTranspose(256, 2, strides=(2,2), kernel_initializer='he_normal', name="deconv_3b")(dss_up_4[1])
uconv_3a = concatenate([deconv_3a, dss_3[0]], name="concat_3a")
uconv_3a = Dropout(0.5, name="dropout_up_3a")(uconv_3a)
uconv_3b = concatenate([deconv_3b, dss_3[1]], name="concat_3b")
uconv_3b = Dropout(0.5, name="dropout_up_3b")(uconv_3b)
dss_up_3 = DSS(filters=256, name="updss_3-1")([uconv_3a, uconv_3b])
#dss_up_3 = DSS(filters=256, name="updss_3-2")([dss_up_3[0], dss_up_3[1]])


#--------- climb a step in the U-pattern -------#
deconv_2a = Conv2DTranspose(128, 2, strides=(2,2), kernel_initializer='he_normal', name="deconv_2a")(dss_up_3[0])
deconv_2b = Conv2DTranspose(128, 2, strides=(2,2), kernel_initializer='he_normal', name="deconv_2b")(dss_up_3[1])
uconv_2a = concatenate([deconv_2a, dss_2[0]], name="concat_2a")
uconv_2a = Dropout(0.5, name="dropout_up_2a")(uconv_2a)
uconv_2b = concatenate([deconv_2b, dss_2[1]], name="concat_2b")
uconv_2b = Dropout(0.5, name="dropout_up_2b")(uconv_2b)
dss_up_2 = DSS(filters=128, name="updss_2-1")([uconv_2a, uconv_2b])
#dss_up_2 = DSS(filters=128, name="updss_2-2")([dss_up_2[0], dss_up_2[1]])


#--------- climb a step in the U-pattern -------#
#---------     top of the U-pattern     --------#
deconv_1a = Conv2DTranspose(64, 2, strides=(2,2), kernel_initializer='he_normal', name="deconv_1a")(dss_up_2[0])
deconv_1b = Conv2DTranspose(64, 2, strides=(2,2), kernel_initializer='he_normal', name="deconv_1b")(dss_up_2[1])
uconv_1a = concatenate([deconv_1a, dss_1[0]], name="concat_1a")
uconv_1a = Dropout(0.5, name="dropout_up_1a")(uconv_1a)
uconv_1b = concatenate([deconv_1b, dss_1[1]], name="concat_1b")
uconv_1b = Dropout(0.5, name="dropout_up_1b")(uconv_1b)
dss_up_1 = DSS(filters=64, name="updss_1-1")([uconv_1a, uconv_1b])
#dss_up_1 = DSS(filters=64, name="updss_1-2")([dss_up_1[0], dss_up_1[1]])


#-------- reshaping the data to fit inputs ------#
output_a = Conv2D(3, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal', name="output_a")(dss_up_1[0])
output_b = Conv2D(3, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal', name="output_b")(dss_up_1[1])


CNN = Model([input_a, input_b], [output_a, output_b])

CNN._name = "U-SYM-V0.6"

#------------- visualising the model ------------#
#CNN.summary()
#keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
#keras.utils.plot_model(CNN, "multi_input_and_output_model.png")

image_train = np.load('../data/sample_data.npy')# data

# Normalising the data
mean = np.mean(image_train, axis=(1,2), keepdims=True)
std = np.std(image_train, axis=(1,2), keepdims=True)
image_train = (image_train - mean) / std

label_train = image_train # labels

# Ground truth
prediction_plot(image_train[0,:,:,0], "Simulated data (ground truth)", '../run/figures/ground_truth.png')

# Applying masks
image_train[:,:,80:90,0:1] = 0

# Creating path where networl progress is saved
checkpoint_path = '../checkpoints/latest_usym.hdf5'

# Creating an instance of the loss class
loss = custom_loss()

modelcheckpoint  = ModelCheckpoint(save_best_only=True, save_weights_only = True,  verbose = 1, filepath = checkpoint_path, monitor = 'val_loss')
csvlogger = CSVLogger( filename = '../run/log_usym.csv', separator = ',' , append = False )
callback_list  = [modelcheckpoint , csvlogger]

# Compiling network
masked_loss = custom_loss().wrapper()
CNN.compile(optimizer = Adam(learning_rate = 1e-4), loss = masked_loss , metrics = [ masked_loss ])

# Fitting the model
CNN.fit([image_train, image_train], [label_train, label_train], batch_size = 5, epochs = 12, callbacks = [callback_list], validation_split = 0.2)

print('Done, moving to predictions', flush = True)

# Making predictions
predictions = CNN.predict([image_train[:, :, :], image_train[:, :, :]])

prediction_plot(predictions[0][0,:,:,0], 'U-SYM A (real)', '../run/figures/usym_a_real.png', vmax=25)
prediction_plot(predictions[0][0,:,:,1], 'U-SYM A (imaginary)', '../run/figures/usym_a_imaginary.png', vmax=25)
prediction_plot(predictions[1][0,:,:,0], 'U-SYM B (real)', '../run/figures/usym_b_real.png', vmax=25)
prediction_plot(predictions[1][0,:,:,1], 'U-SYM B (imaginary)', '../run/figures/usym_b_imaginary.png', vmax=25)

learning_plot('../run/log_usym.csv', "U-SYM (learning curves)", '../run/figures/usym_learning.png', end=7)