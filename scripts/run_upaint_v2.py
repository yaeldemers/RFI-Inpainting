print("I'm in! (BEFORE input statements)")

import numpy as np
#from models.upaint import model_upaint as UPAINT
import model_upaint as UPAINT
#import model_upaint_v2 as UPAINT2
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
#from utils import misc as misc
import misc

#import os
#import sys

import tensorflow as tf
from tensorflow.keras.models import load_model

print("Still in! (AFTER input statements)")

# Add the RFI-Inpainting directory to the Python path
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

wd = '/home/ydemers/projects/def-acliu/ydemers/RFI-Inpainting'

np.random.seed(0);

data = np.load(wd+'/data/local_hera_256_data.npy') 

permutations = np.loadtxt(wd+'/scripts/permutations.csv', delimiter=',', dtype ='int')[:1]

masked_data= misc.create_masked_data(data, mask_width=10, num_masks=4)

for i in range(len(permutations)):

    print()
    print('STARTING TRAINING '+str(i))
    print()

    x_train, y_train, masks_train, x_val, y_val, masks_val, x_test, y_test, masks_test, indices = misc.split_dataset(
        masked_data['masked_data'], 
        masked_data['unmasked_data'], 
        masked_data['masks'],
        permutations[i]
        ) 

    #with open(wd+'/run/indices.csv','a') as f:
    #    np.savetxt(f, indices.reshape(1, -1), fmt='%i', delimiter=",")

    # Creating path where network progress is saved
    checkpoint_path = wd+'/models/upaint/checkpoints/log_upaint.hdf5'

    #TODO: Check whether or not I am replacing or appending the checkpoints
    modelcheckpoint = ModelCheckpoint(
        filepath = checkpoint_path, 
        save_best_only=True, 
        save_weights_only = True,  
        verbose = 1, 
        monitor = 'val_loss', 
        period=5 # Save every 5 epochs to save diskspace
        )
    csvlogger = CSVLogger( filename = wd+'/logs/log_upaint.csv', separator = ',' , append = False )
    callback_list  = [modelcheckpoint , csvlogger]

    # Creating an instance of the loss class
    loss = misc.custom_loss()

    # Creating a model with the CNN
    UPAINT_obj = UPAINT.Unet(data[1,:,:].shape, loss, checkpoint_path)

    #UPAINT_obj.model.load_weights(checkpoint_path)

    # Running the network
    UPAINT_obj.model.fit(x_train, y_train, batch_size = 4, epochs = 96, callbacks = [callback_list], validation_data=(x_val, y_val))
    print('Done, moving to predictions', flush = True)

    # Making predictions
    """
    ##############################
    model_path = wd+'/checkpoints/test_upaint.hdf5'

    #testing to see if model can predict the same things as local
    model = UPAINT.Unet(data[1,:,:].shape, loss, model_path)
    model.model.load_weights(model_path)    
    predictions = model.model.predict(x_test[:, :, :, :])
    ##############################
    """
    predictions = UPAINT_obj.model.predict(x_test[:, :, :, :])

    np.savez(wd+'/outputs/data_out_remote_final_true.npz', predictions=predictions, ground_truths=y_test, masks=masks_test) 

    plt.imshow(predictions[0,:,:,0], origin = 'lower')
    plt.title('Remote Predictions')
    clb = plt.colorbar()
    plt.savefig(wd+'/figures/prediction_remote')
    plt.close()

    misc.learning_plot(wd+'/logs/log_upaint.csv', 'U-Paint (learning curves)', wd+'/figures/upaint_learning_final.png', start=0)
