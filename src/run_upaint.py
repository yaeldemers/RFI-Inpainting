import numpy as np
import model_upaint as UPAINT
#import model_upaint_v2 as UPAINT2
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import misc as misc

data = np.load('../data/hera_256_data.npy')  

data_masked = data
means_real, stds_real = np.zeros(len(data)), np.zeros(len(data)) 
means_im, stds_im = np.zeros(len(data)), np.zeros(len(data)) 

# Nomalising data
for i in range(len(data)):
    norm_real, mean_real, std_real = misc.normalize(data[i,:,:,0])
    norm_im, mean_im, std_im = misc.normalize(data[i,:,:,1])
    data_masked[i,:,:,0], data_masked[i,:,:,1]  = norm_real, norm_im
    means_real[i], stds_real[i] = mean_real, std_real 
    means_im[i], stds_im[i] = mean_im, std_im
    
data_not_masked = data_masked.copy() # labels
data_not_masked[:,:,80:90,2] = 1

# Applying masks
data_masked[:,:,80:90,0:2], data_masked[:,:,80:90,2]  = 0 , 1

x_train, y_train = data_masked[:17], data_not_masked[:17]
x_val, y_val = data_masked[17:21], data_not_masked[17:21]
x_pred, y_pred = data_masked[21:], data_not_masked[21:]

# Creating path where network progress is saved
checkpoint_path = '../checkpoints/latest_upaint.hdf5'

modelcheckpoint   = ModelCheckpoint(save_best_only=True, save_weights_only = True,  verbose = 1, filepath = checkpoint_path, monitor = 'val_loss'  )
csvlogger = CSVLogger( filename = '../run/log_upaint.csv', separator = ',' , append = False )
callback_list  = [modelcheckpoint , csvlogger]

# Creating an instance of the loss class
loss = misc.custom_loss()
#loss = custom_SE()
    
# Creating a model with the CNN
UPAINT_obj = UPAINT.Unet(data_masked[1,:,:].shape, loss, checkpoint_path)

# Running the network
UPAINT_obj.model.fit(x_train, y_train, batch_size = 4, epochs = 12, callbacks = [callback_list], validation_data=(x_val, y_val))
#UPAINT_obj.model.fit(X_masked, Y_not_masked, batch_size = 1, epochs = 24, callbacks = [callback_list], validation_split = 0.5)
print('Done, moving to predictions', flush = True)

# Making predictions
predictions = UPAINT_obj.model.predict(x_pred[:, :, :, :])

# Unnormalizing predictions
for i in range(len(data[:4])):
    predictions[i,:,:,0] = misc.unnormalize(predictions[i,:,:,0], means_real[i], stds_real[i])
    predictions[i,:,:,1] = misc.unnormalize(predictions[i,:,:,1], means_im[i], stds_im[i])

np.savez("../run/data_out.npz", predictions=predictions, ground_truths=data_not_masked, masks=data_not_masked[:,:,:,2]) 

misc.learning_plot('../run/log_upaint.csv', "U-Paint (learning curves)", '../run/figures/upaint_learning.png')