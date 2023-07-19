import numpy as np
import model_upaint as UPAINT
#import model_upaint_v2 as UPAINT2
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import misc as misc

import tensorflow as tf
from tensorflow.keras.models import load_model

np.random.seed(0);

data = np.load('../data/local_hera_256_data.npy') 

wd = '/home/ydemers/projects/def-acliu/ydemers/RFI-Inpainting'

permutations = np.loadtxt('permutations.csv', delimiter=',', dtype ='int')[:1]

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
    checkpoint_path = wd+'/checkpoints/test_upaint.hdf5'

    #TODO: Check whether or not I am replacing or appending the checkpoints
    modelcheckpoint = ModelCheckpoint(save_best_only=True, save_weights_only = True,  verbose = 1, filepath = checkpoint_path, monitor = 'val_loss')
    csvlogger = CSVLogger( filename = wd+'/run/log_upaint.csv', separator = ',' , append = False )
    callback_list  = [modelcheckpoint , csvlogger]

    # Creating an instance of the loss class
    loss = misc.custom_loss()

    # Creating a model with the CNN
    UPAINT_obj = UPAINT.Unet(data[1,:,:].shape, loss, checkpoint_path)

    UPAINT_obj.model.load_weights(checkpoint_path)

    # Running the network
    UPAINT_obj.model.fit(x_train, y_train, batch_size = 4, epochs = 256, callbacks = [callback_list], validation_data=(x_val, y_val))
    print('Done, moving to predictions', flush = True)

    # Making predictions
    """
    ##############################
    model_path = wd+'/checkpoints/test_upaint.hdf5'

    #testing to see if model can predict the same things as local
    model = UPAINT.Unet(data[1,:,:].shape, loss, model_path)
    model.model.load_weights(model_path)    
    predictions = model.model.predict(x_test[:, :, :, :])
    """
    ##############################

    predictions = UPAINT_obj.model.predict(x_test[:, :, :, :])

    np.savez(wd+"/run/data_out_remote_final_true.npz", predictions=predictions, ground_truths=y_test, masks=masks_test) 

    plt.imshow(predictions[0,:,:,0], origin = 'lower')
    plt.title('Remote Predictions')
    clb = plt.colorbar()
    plt.savefig('../run/figures/prediction_remote')
    plt.close()

    misc.learning_plot(wd+'/run/log_upaint.csv', "U-Paint (learning curves)", wd+'/run/figures/upaint_learning_final.png', start=0)
