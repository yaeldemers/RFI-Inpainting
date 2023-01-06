import numpy as np
import model_upaint as UPAINT
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from misc import prediction_plot, learning_plot, custom_loss

X_masked = np.load('../data/sample_data.npy')# data

# Normalising the data
mean = np.mean(X_masked, axis=(1,2), keepdims=True)
std = np.std(X_masked, axis=(1,2), keepdims=True)
X_masked = (X_masked - mean) / std

Y_not_masked = X_masked # labels

# Applying masks
X_masked[:,:,80:90,0:1] = 0

# Creating path where network progress is saved
checkpoint_path = '../checkpoints/latest_upaint.hdf5'

modelcheckpoint   = ModelCheckpoint(save_best_only=True, save_weights_only = True,  verbose = 1, filepath = checkpoint_path, monitor = 'val_loss'  )
csvlogger = CSVLogger( filename = '../run/log_upaint.csv', separator = ',' , append = False )
callback_list  = [modelcheckpoint , csvlogger]

# Creating an instance of the loss class
loss = custom_loss()
    
# Creating a model with the CNN
UPAINT_obj = UPAINT.Unet(X_masked[1,:,:].shape, loss, checkpoint_path)

# Running the network
UPAINT_obj.model.fit(X_masked, Y_not_masked, batch_size = 5, epochs = 12, callbacks = [callback_list], validation_split = 0.2)

print('Done, moving to predictions', flush = True)

# Making predictions
predictions = UPAINT_obj.model.predict(X_masked[:, :, :, :])

prediction_plot(predictions[0,:,:,0], 'U-Paint Prediction (real)', '../run/figures/upaint_real.png')
prediction_plot(predictions[0,:,:,1], 'U-Paint Prediction (imaginary)', '../run/figures/upaint_imaginary.png')

learning_plot('../run/log_upaint.csv', "U-Paint (learning curves)", '../run/figures/upaint_learning.png')