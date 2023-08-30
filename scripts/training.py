"""
run_upaint.py

Description: Main script of the project. Responsible data processing, training and predictions

Authors: Michael Pagano, Yael-Jeremy Demers

Last Modified: 16-08-2023

Changes:
- 16-08-2023 Demers: Modyfing multi-GPU strategy when running on CPU
- 03-08-2023 Demers: Implementing parameters setup from bash file
- 28-07-2023 Demers: Setting up multi-GPU strategy and file cleanup
- 26-07-2023 Demers: Setting up script for 512x512 dataset
"""

import argparse
import numpy as np
import model_upaint as UPAINT
import scripts.model_upaint_dss as UPAINT_DSS
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from utils import learning_plot, prediction_plot, create_masked_data, split_dataset, custom_loss, test_loss

import tensorflow as tf
from tensorflow.keras.models import load_model

def train_upaint(model_path = 'checkpoints/latest_upaint.hdf5', model_type = 'UPAINT', epochs = 96, batch_size = 4):

    wd = '/home/ydemers/projects/rrg-acliu/ydemers/RFI-Inpainting'

    data = np.load(wd + '/data/data.npy') 
    flags = np.load(wd + '/data/flags.npy')

    masked_data = create_masked_data(data, mask_width = 5, num_masks = 4)
    #masked_data = create_masked_data(data, masks = flags)

    x_train, y_train, masks_train, x_val, y_val, masks_val, x_test, y_test, masks_test = split_dataset(
        masked_data['masked_data'], 
        masked_data['unmasked_data'], 
        masked_data['masks']
        ) 

    modelcheckpoint = ModelCheckpoint(
        filepath = model_path, 
        save_best_only = True, 
        save_weights_only = True,  
        verbose = 1, 
        monitor = 'val_loss' 
        #period = 5 # Save every 5 epochs to save diskspace
        )

    earlystopping = EarlyStopping(patience = 40, verbose = 0, restore_best_weights = True)
    csvlogger = CSVLogger( filename = wd + '/logs/log_' + model_type + '.csv', separator = ',' , append = False )
    callback_list  = [earlystopping, modelcheckpoint, csvlogger]

    # Creating an instance of the loss class
    #loss = custom_loss()
    loss = test_loss

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
                if model_type == 'UPAINT':
                    model_constructor = UPAINT.Unet
                elif model_type == 'UPAINT2_DSS':
                    model_constructor = UPAINT_DSS.Unet
                else:
                    raise ValueError("Invalid model_type. Use 'UPAINT' or 'UPAINT_DSS'.")

                UPAINT_obj = model_constructor(data[1,:,:].shape, loss, model_path)

                # Uncomment below if loading previously trained model. Note that the current bash file deletes previous checkpoints.
                # UPAINT_obj.model.load_weights(checkpoint_path)

                UPAINT_obj.model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, callbacks = [callback_list], validation_data = (x_val, y_val))

        except RuntimeError as e:
            print(e)

    else:
        if model_type == 'UPAINT':
            model_constructor = UPAINT.Unet
        elif model_type == 'UPAINT_DSS':
            model_constructor = UPAINT_DSS.Unet
        else:
            raise ValueError("Invalid model_type. Use 'UPAINT' or 'UPAINT_DSS'.")

        UPAINT_obj = model_constructor(data[1,:,:].shape, loss, model_path)

                # Uncomment below if loading previously trained model. Note that the current bash file deletes previous checkpoints.
                # UPAINT_obj.model.load_weights(checkpoint_path)

        UPAINT_obj.model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, callbacks = [callback_list], validation_data = (x_val, y_val))

    print('Done, moving to predictions', flush = True)

    predictions = UPAINT_obj.model.predict(x_test[:, :, :, :])

    print('time to save...')

    np.savez(wd + '/outputs/upaint_data_out_test.npz', predictions = predictions, ground_truths = y_test, masks = masks_test)

    print('time to plot...')

    prediction_plot(predictions[0,:,:,0], 'Remote Prediction Sample', wd + '/figures/prediction_remote_test')

    learning_plot(wd + '/logs/log_' + model_type + '.csv', 'U-Paint (learning curves)', wd + '/figures/upaint_learning_final_test.png', start = 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U-Paint training script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--model_type", type=str, required=True, choices=["UPAINT", "UPAINT2"], help="Choose the model type: 'UPAINT' or 'UPAINT2'.")
    parser.add_argument("--epochs", type=int, default=96, help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
    args = parser.parse_args()

    train_upaint(args.model_path, args.model_type, args.epochs, args.batch_size)