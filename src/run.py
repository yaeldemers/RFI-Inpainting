import numpy as np
import CNN_model as CNN

from datetime import date
today = date.today()

from tensorflow.keras.callbacks import ModelCheckpoint
from loss import custom_loss

# Generating sample data to test the network
X_masked_validation = np.random.normal(0, 1, size = (5, 512, 512, 3)) # data validation

X_masked = np.random.normal(0, 1, size = (5, 512, 512, 3)) # data

Y_not_masked = np.random.normal(0, 1, size = (5, 512, 512, 3)) # labels

# Creating path where network progress is saved
checkpoint_path = '../checkpoints/latest.hdf5'

# Creating an instance of the loss class
loss = custom_loss()  #the loss needs to know the shape of the batch size too
    
# Creating a model with the CNN
CNN_obj = CNN.Unet(X_masked[1,:,:].shape, loss, checkpoint_path)

# Debugging, testing and evaluation of network (i.e. is the network good?)
CNN.model.summary() 

modelcheckpoint = ModelCheckpoint(save_best_only = True, save_weights_only = True, verbose = 1, filepath = checkpoint_path, monitor = 'val_loss')

# Running the network
#CNN_obj.model.fit(X_masked, Y_not_masked, batch_size = 5, epochs = 80, callbacks = [callback_list], validation_split = 0.1)
CNN_obj.model.fit(X_masked, Y_not_masked, batch_size = 5, epochs = 8, validation_split = 0.1)

print('Done, moving to predictions', flush = True)

# Making predictions
predictions = CNN_obj.model.predict(X_masked_validation[:, :, :, :])