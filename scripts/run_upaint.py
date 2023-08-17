"""
run_upaint.py

Description: Main script of the project. Responsible data processing, training and predictions

Authors: Michael Pagano, Yael-Jeremy Demers

Last Modified: 03-08-2023

Changes:
- 03-08-2023 Demers: Implementing parameters setup from bash file
- 28-07-2023 Demers: Setting up multi-GPU strategy and file cleanup
- 26-07-2023 Demers: Setting up script for 512x512 dataset
"""

import numpy as np
#import model_upaint as UPAINT
import model_upaint_v2 as UPAINT2
#import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from utils import learning_plot, prediction_plot, create_masked_data, split_dataset, custom_loss

import tensorflow as tf
from tensorflow.keras.models import load_model

wd = '/home/ydemers/projects/rrg-acliu/ydemers/RFI-Inpainting'

data = np.load(wd+'/data/data.npy') 
#flags = np.load(wd+'/data/flags.npy')

masked_data = create_masked_data(data, mask_width=5, num_masks=4)
#masked_data = create_masked_data(data, masks=flags)

x_train, y_train, masks_train, x_val, y_val, masks_val, x_test, y_test, masks_test, indices = split_dataset(
    masked_data['masked_data'], 
    masked_data['unmasked_data'], 
    masked_data['masks']
    ) 

# Creating path where network progress is saved
checkpoint_path = wd+'/checkpoints/latest_upaint.hdf5'

modelcheckpoint = ModelCheckpoint(
    filepath = checkpoint_path, 
    save_best_only = True, 
    save_weights_only = True,  
    verbose = 1, 
    monitor = 'val_loss' 
    #period=5 # Save every 5 epochs to save diskspace
    )

csvlogger = CSVLogger( filename = wd+'/logs/log_upaint.csv', separator = ',' , append = False )
callback_list  = [modelcheckpoint , csvlogger]

# Creating an instance of the loss class
loss = custom_loss()

# Request all available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth for each GPU (optional)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # Create a MirroredStrategy
        strategy = tf.distribute.MirroredStrategy()

        # Now define and compile model within the strategy's scope
        with strategy.scope():
            UPAINT_obj = UPAINT2.Unet(data[1,:,:].shape, loss, checkpoint_path)

            # Uncomment below if loading previously trained model. Note that the current bash file deletes previous checkpoints.
            # UPAINT_obj.model.load_weights(checkpoint_path)

            UPAINT_obj.model.fit(x_train, y_train, batch_size = 4, epochs = 256, callbacks = [callback_list], validation_data=(x_val, y_val))

    except RuntimeError as e:
        print(e)

print('Done, moving to predictions', flush = True)

predictions = UPAINT_obj.model.predict(x_test[:, :, :, :])

np.savez(wd+'/outputs/upaint_data_out.npz', predictions=predictions, ground_truths=y_test, masks=masks_test) 

#plt.imshow(predictions[0,:,:,0], origin = 'lower')
#plt.title('Remote Predictions Sample')
#clb = plt.colorbar()
#plt.savefig(wd+'/figures/prediction_remote')
#plt.close()

prediction_plot(predictions[0,:,:,0], "Remote Prediction Sample", wd+'/figures/prediction_remote')

learning_plot(wd+'/logs/log_upaint.csv', 'U-Paint (learning curves)', wd+'/figures/upaint_learning_final.png', start=1)