"""
utils.py

Description: Collection of classes and helper functions used throughout this project

Authors: Yael-Jeremy Demers

Last Modified: 14-08-2023

Changes:
- 30-08-2023 Demers: Optimized and documented current helper functions
- 29-08-2023 Demers: Moved plotting functions into visualization notebook
- 14-08-2023 Demers: Added the option of using existing HERA flags with create_masked_data()
- 26-07-2023 Demers: Modifying slice_data() to account for dataset with more than one images
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv
    
def normalize(data):
    """
    Normalize the input data by subtracting the mean and dividing by the standard deviation.

    Args:
    data (np.ndarray): Input data to be normalized.

    Returns:
    np.ndarray: Normalized data.
    float: Mean of the original data.
    float: Standard deviation of the original data.
    """
    mean = np.mean(data)
    std = np.std(data)
    normalized_data = (data - mean) / std
    return normalized_data, mean, std
    
def unnormalize(data, mean, std):
    """
    Unnormalize the input data by reversing the normalization process.

    Args:
    data (np.ndarray): Normalized data to be unnormalized.
    mean (float): Mean of the original data.
    std (float): Standard deviation of the original data.

    Returns:
    np.ndarray: Unnormalized data.
    """
    return (data * std) + mean

def splice_data(size, location_from, location_to):
    """
    Splice and rearrange data from one location to another.

    Args:
    size (int): Size of the spliced segments.
    location_from (str): Path to the source data file.
    location_to (str): Path to save the spliced data.

    Returns:
    None
    """
    data_in = np.load(location_from)
    
    shape = data_in.shape
    
    count_y = int(np.floor(shape[1]/size))
    count_x = int(np.floor(shape[2]/size))
    
    data_in = data_in[:,:size*count_y,:size*count_x]
    data_out = []
    
    for i in range(shape[0]):
        for j in range(count_x):
            for k in range(count_y):
            
                curr=np.zeros(shape=(size, size, 3))
    
                real = np.real([data_in[i, k*size:(k+1)*size, j*size:(j+1)*size]])
                imaginary = np.imag([data_in[i, k*size:(k+1)*size, j*size:(j+1)*size]])
            
                curr[:,:,0] = real
                curr[:,:,1] = imaginary

                data_out.append(curr)
            
                # Commented out plotting to avoid excessive visualizations
                # plt.figure(figsize=(10, 10))
                # plt.imshow(curr, origin='lower')
                # plt.colorbar()
                        
    np.save(location_to, data_out)

def test_loss(Y_true, Y_pred):
    """
    Custom loss function that calculates the chi^2 of masked regions in predicted and ground truth visibility data.

    Args:
    Y_true (tf.Tensor): Ground truth visibility data with shape [batch_size, height, width, channels].
    Y_pred (tf.Tensor): Predicted visibility data with shape [batch_size, height, width, channels].

    Returns:
    tf.Tensor: Loss value.
    """
    mask_array = Y_true[:, :, :, 2]

    Y_pred_real = Y_pred[:, :, :, 0] * mask_array
    Y_true_real = Y_true[:, :, :, 0] * mask_array
    
    Y_pred_imag = Y_pred[:, :, :, 1] * mask_array
    Y_true_imag = Y_true[:, :, :, 1] * mask_array
    
    ground_truth_reconstructed = tf.complex(Y_true_real, Y_true_imag)
    predictions_reconstructed = tf.complex(Y_pred_real, Y_pred_imag)
    
    chi = ground_truth_reconstructed - predictions_reconstructed
    chi2 = tf.math.conj(chi) * chi
    
    return tf.math.reduce_sum(tf.math.real(chi2))

class custom_loss:
    def wrapper(self):

        def masked_loss(Y_true, Y_pred):
            """
            Custom loss function that calculates the chi^2 of masked regions in predicted and ground truth visibility data.

            Args:
            Y_true (tf.Tensor): Ground truth visibility data with shape [batch_size, height, width, channels].
            Y_pred (tf.Tensor): Predicted visibility data with shape [batch_size, height, width, channels].

            Returns:
            tf.Tensor: Loss value.
            """
            mask_array = Y_true[:, :, :, 2]

            Y_pred_real = Y_pred[:, :, :, 0] * mask_array
            Y_true_real = Y_true[:, :, :, 0] * mask_array
    
            Y_pred_imag = Y_pred[:, :, :, 1] * mask_array
            Y_true_imag = Y_true[:, :, :, 1] * mask_array
    
            ground_truth_reconstructed = tf.complex(Y_true_real, Y_true_imag)
            predictions_reconstructed = tf.complex(Y_pred_real, Y_pred_imag)
    
            chi = ground_truth_reconstructed - predictions_reconstructed
            chi2 = tf.math.conj(chi) * chi
    
            return tf.math.reduce_sum(tf.math.real(chi2))

def create_masked_data(data, mask_width=10, num_masks=1, masks=[]):
    """
    Create masked and unmasked data for neural network training.

    Args:
    data (np.ndarray): Input data with shape [n, height, width, channels].
    mask_width (int, optional): Width of the mask to be applied. Default is 10.
    num_masks (int, optional): Number of masks to be applied. Default is 1.
    masks (list or np.ndarray, optional): List of pre-defined masks. Default is an empty list.

    Returns:
    dict: A dictionary containing masked_data, unmasked_data, and masks.
    """
    
    data_masked = data.copy()
    
    # Compute means and standard deviations for each image separately
    means = np.mean(data[:, :, :, 0:2], axis=(1,2), keepdims=True)
    std_devs = np.std(data[:, :, :, 0:2], axis=(1,2), keepdims=True)

    # Normalize each image separately, excluding third channel
    data_masked[:, :, :, 0:2] = (data_masked[:, :, :, 0:2] - means[:, :, :, 0:2]) / std_devs[:, :, :, 0:2]

    # This if-else ladder generates flags if not provided. Should be removed down the line when more flag options are provided.
    if len(masks)==0:
    
        height, width = data.shape[1:3]
        
        # Randomly select mask locations
        #mask_indices = np.random.randint(low=0, high=width-mask_width, size=(len(data), num_masks))
    
        # Alternatively, uncomment to setup static mask locations
        increment = 50
        mask_indices = np.arange(0, num_masks*increment, increment) + increment
        
        # Create masks
        masks = []
        inverse_masks = []
        
        for i in range(len(data)):
            mask = np.ones((height, width)).astype(int)
            inverse_mask = np.zeros((height, width)).astype(int)
            for j in range(num_masks):
                # Uncomment if using random flags
                #mask[:, mask_indices[i,j]:mask_indices[i,j]+mask_width] = 0
                #inverse_mask[:, mask_indices[i,j]:mask_indices[i,j]+mask_width] = 1
                
                # Uncomment if using static flags
                mask[:, mask_indices[j]:mask_indices[j] + mask_width] = 0
                inverse_mask[:, mask_indices[j]:mask_indices[j] + mask_width] = 1
            
            masks.append(mask)
            inverse_masks.append(inverse_mask)
        masks, inverse_masks = np.array(masks), np.array(inverse_masks)
    
    else:
        inverse_masks = masks.copy().astype(int)
        masks = np.logical_not(masks).astype(int)
    
    # Apply masks
    data_not_masked = data.copy() # labels

    data_masked[:,:,:,0] = data_masked[:,:,:,0] * masks
    data_masked[:,:,:,1] = data_masked[:,:,:,1] * masks
    data_masked[:,:,:,2] = inverse_masks
    data_not_masked[:,:,:,2] = inverse_masks
    
    return {'masked_data': data_masked,
                'unmasked_data': data_not_masked,
                'masks': inverse_masks.squeeze()
                }

def split_dataset(masked_data, unmasked_data, masks, indices=[]):
    """
    Split the dataset into training, validation, and testing subsets.

    Args:
    masked_data (np.ndarray): Masked data for training.
    unmasked_data (np.ndarray): Unmasked data for labels.
    masks (np.ndarray): Binary masks indicating masked regions.
    indices (list or np.ndarray, optional): Indices for dataset splitting. Default is an empty list.

    Returns:
    tuple: A tuple containing training and validation data along with masks.
           x_train (np.ndarray): Masked training data.
           y_train (np.ndarray): Unmasked training labels.
           masks_train (np.ndarray): Masks for training data.
           x_val (np.ndarray): Masked validation data.
           y_val (np.ndarray): Unmasked validation labels.
           masks_val (np.ndarray): Masks for validation data.
           x_test (np.ndarray): Masked testing data.
           y_test (np.ndarray): Unmasked testing labels.
           masks_test (np.ndarray): Masks for testing data.
    """
    n = masked_data.shape[0]  # Number of images in the dataset

    # Generate random indices for sampling without replacement
    if len(indices) == 0:
        indices = np.random.permutation(n)

    # Calculate the sizes of the three subsets
    n_train = int(0.8 * 0.8 * n)  # 64% of total data
    n_val = int(0.2 * 0.8 * n)    # 16% of total data
    n_test = n - n_train - n_val   # Remaining 20% of total data
    subset_sizes = [n_train, n_val, n_test]

    # Split the indices into three subarrays based on subset sizes
    sub_indices = np.split(indices, np.cumsum(subset_sizes)[:-1])
    
    # Use the indices to create the three subdatasets
    x_train, y_train, masks_train = masked_data[sub_indices[0]], unmasked_data[sub_indices[0]], masks[sub_indices[0]]
    x_val, y_val, masks_val = masked_data[sub_indices[1]], unmasked_data[sub_indices[1]], masks[sub_indices[1]]
    x_test, y_test, masks_test = masked_data[sub_indices[2]], unmasked_data[sub_indices[2]], masks[sub_indices[2]]
    
    return (
        x_train, y_train, masks_train,
        x_val, y_val, masks_val,
        x_test, y_test, masks_test
    )