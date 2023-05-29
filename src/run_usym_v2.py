from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from misc import prediction_plot, learning_plot, custom_loss 
from model_usym import USYM
import numpy as np

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

# Creating path where network progress is saved
checkpoint_path = '../checkpoints/latest_usym.hdf5'

modelcheckpoint  = ModelCheckpoint(save_best_only=True, save_weights_only = True,  verbose = 1, filepath = checkpoint_path, monitor = 'val_loss')
csvlogger = CSVLogger( filename = '../run/log_usym.csv', separator = ',' , append = False )
callback_list  = [modelcheckpoint , csvlogger]

# Creating an instance of the loss class
loss = custom_loss()
input_a = Input(np.random.normal(0, 1, size = (128, 128, 3)).shape, name="input_a")
input_b = Input(np.random.normal(0, 1, size = (128, 128, 3)).shape, name="input_b")

# Creating a model with the CNN
USYM_obj = USYM([input_a, input_b], loss, checkpoint_path)

# Fitting the model
USYM_obj.model.fit([image_train, image_train], [label_train, label_train], batch_size = 5, epochs = 12, callbacks = [callback_list], validation_split = 0.2)

print('Done, moving to predictions', flush = True)

# Making predictions
predictions = USYM_obj.model.predict([image_train[:, :, :], image_train[:, :, :]])

prediction_plot(predictions[0][0,:,:,0], 'U-SYM A (real)', '../run/figures/usym_a_real.png', vmax=25)
prediction_plot(predictions[0][0,:,:,1], 'U-SYM A (imaginary)', '../run/figures/usym_a_imaginary.png', vmax=25)
prediction_plot(predictions[1][0,:,:,0], 'U-SYM B (real)', '../run/figures/usym_b_real.png', vmax=25)
prediction_plot(predictions[1][0,:,:,1], 'U-SYM B (imaginary)', '../run/figures/usym_b_imaginary.png', vmax=25)

learning_plot('../run/log_usym.csv', "U-SYM (learning curves)", '../run/figures/usym_learning.png', end=7)