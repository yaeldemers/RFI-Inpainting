import numpy as np
import model_upaint as UPAINT
#import model_upaint_v2 as UPAINT2
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import misc as misc

data = np.load('../data/256x256/sample_data1.npy') 

masked_data= misc.create_masked_data(data, mask_width=10, num_masks=1)

x_train, y_train, masks_train, x_val, y_val, masks_val, x_test, y_test, masks_test, indices = misc.split_dataset(
    masked_data['masked_data'], 
    masked_data['unmasked_data'], 
    masked_data['masks']
    ) 

with open('../run/indices.csv','a') as f:
    np.savetxt(f, indices.reshape(1, -1), fmt='%i', delimiter=",")

# Creating path where network progress is saved
checkpoint_path = '../checkpoints/latest_upaint.hdf5'

modelcheckpoint   = ModelCheckpoint(save_best_only=True, save_weights_only = True,  verbose = 1, filepath = checkpoint_path, monitor = 'val_loss'  )
csvlogger = CSVLogger( filename = '../run/log_upaint.csv', separator = ',' , append = False )
callback_list  = [modelcheckpoint , csvlogger]

# Creating an instance of the loss class
loss = misc.custom_loss()
    
# Creating a model with the CNN
UPAINT_obj = UPAINT.Unet(data[1,:,:].shape, loss, checkpoint_path)

# Running the network
UPAINT_obj.model.fit(x_train, y_train, batch_size = 4, epochs = 12, callbacks = [callback_list], validation_data=(x_val, y_val))
print('Done, moving to predictions', flush = True)

# Making predictions
predictions = UPAINT_obj.model.predict(x_test[:, :, :, :])

np.savez("../run/data3.npz", predictions=predictions, ground_truths=y_test, masks=masks_test) 

plt.imshow(x_test[0,:,:,0], origin = 'lower')
plt.title('Flagged')
clb = plt.colorbar()
plt.show()

plt.imshow(y_test[0,:,:,0], origin = 'lower')
plt.title('Ground Truth')
clb = plt.colorbar()
plt.show()

plt.imshow(predictions[0,:,:,0], origin = 'lower')
plt.title('Predictions')
clb = plt.colorbar()
plt.show()

misc.learning_plot('../run/log_upaint.csv', "U-Paint (learning curves)", '../run/figures/upaint_learning.png', start=0)
