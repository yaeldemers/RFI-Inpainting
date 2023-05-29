import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv

def prediction_plot(data, title, save_at, vmax=None, vmin=None):
    plt.imshow(data, origin = 'lower', vmax=vmax, vmin=vmin)
    plt.title(title)
    plt.xlabel('Frequency [MHz]')
    plt.ylabel('$LST_0+[min]$')
    clb = plt.colorbar()
    clb.ax.set_title('$[Jy]/\sigma$')
    plt.savefig(save_at)
    plt.show()
    plt.close()
    
def learning_plot(logs, title, save_at, start=0, end=0):
    data = read_csv(logs)

    # converting column data to list
    loss = data['loss'].tolist()
    val_loss = data['val_loss'].tolist()
    
    if end==0:
        end=len(loss)
    
    plt.plot(val_loss[start:end], label="Validation loss")
    plt.plot(loss[start:end], label="Training loss")
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Learning curve', fontsize=15)
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.show()
    plt.savefig(save_at)
    plt.close()

def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    return [(data - mean) / std, mean, std]
    
def unnormalize(data, mean, std):
    return (data*std+mean) 

def splice_data(size, location, stats_data):
    count_y = int(np.floor(1920/size))
    count_x = int(np.floor(1024/size))
    
    data_in = np.load('../data/raw/data1.npy')[:size*count_y,:size*count_x]
    data_out = []
    
    for i in range(count_x):
        for j in range(count_y):
            
            curr=np.zeros(shape=(size, size, 3))
    
            real = np.real([data_in[j*size:(j+1)*size, i*size:(i+1)*size]])
            imaginary = np.imag([data_in[j*size:(j+1)*size, i*size:(i+1)*size]])

            curr[:,:,0] = real
            curr[:,:,1] = imaginary

            data_out.append(curr)
            
            plt.figure(figsize = (10,10))
            plt.imshow(curr, origin = 'lower')
            plt.colorbar()
            
            
    print(data_out)
            
    np.save(location, data_out)

class custom_loss:
    def wrapper(self):

        def masked_loss(Y_true, Y_pred):
            
            '''
                This loss function takes the prediction Y_pred, applies the mask to the first two channels
                which correspond to the real and imag part of the visibilities, then computes the 'chi^2'
                of those two channels.
            '''

            '''
            Uncomment if 0 -> flag and 1 -> not flagged
            
            ones = tf.ones_like(Y_true[:,:,:,2])
            #invert the mask, we want only the masked areas to enter the chi^2
            mask_array = ones - Y_true[:,:,:,2]
            '''
    
            mask_array = Y_true[:,:,:,2]

            Y_pred_real = tf.math.multiply(tf.cast(Y_pred[:,:,:,0], tf.float64), tf.cast(mask_array[:,:,:], tf.float64))
            Y_true_real = tf.math.multiply(tf.cast(Y_true[:,:,:,0], tf.float64), tf.cast(mask_array[:,:,:], tf.float64))
            
            Y_pred_imag = tf.math.multiply(tf.cast(Y_pred[:,:,:,1], tf.float64), tf.cast(mask_array[:,:,:], tf.float64))
            Y_true_imag = tf.math.multiply(tf.cast(Y_true[:,:,:,1], tf.float64), tf.cast(mask_array[:,:,:], tf.float64))
                        
            ground_truth_reconstructed = tf.complex(Y_true_real, Y_true_imag)
            
            predictions_reconstructed = tf.complex(Y_pred_real, Y_pred_imag)
                           
            chi = ground_truth_reconstructed - predictions_reconstructed
            
            chi2 = tf.math.conj(chi) * chi
            
            return tf.math.real(tf.math.reduce_sum(chi2))
            
        return masked_loss

def create_masked_data(data, mask_width=10, num_masks=1):
    """
    Apply masks to the input data to simulate missing or corrupted data.

    Args:
    data (np.ndarray): Input data with shape [n, 256, 256, 3] where the last dimension represents 3 different 
    channels: real values, imaginary values, and mask location.
    mask_width (int): Width of each mask.
    num_masks (int): Number of masks to apply to each image.

    Returns:
    np.ndarray: Masked data with shape [n, 256, 256, 3].
    np.ndarray: Unmasked data with shape [n, 256, 256, 3].à
    np.ndarray: Masks used for each image with shape [n, 256, 256].
    """
    
    data_masked = data.copy()
    
    # Compute means and standard deviations for each image separately
    means = np.mean(data[:, :, :, 0:2], axis=(1,2), keepdims=True)
    std_devs = np.std(data[:, :, :, 0:2], axis=(1,2), keepdims=True)

    # Normalize each image separately, excluding third channel
    data_masked[:, :, :, 0:2] = (data_masked[:, :, :, 0:2] - means[:, :, :, 0:2]) / std_devs[:, :, :, 0:2]

    # Randomly select mask locations
    height, width = data.shape[1:3]
    mask_indices = np.random.randint(low=0, high=width-mask_width, size=(len(data), num_masks))
    
    # Create masks
    masks = []
    inverse_masks = []
    for i in range(len(data)):
        mask = np.ones((height, width)).astype(int)
        inverse_mask = np.zeros((height, width)).astype(int)
        for j in range(num_masks):
            mask[:, mask_indices[i,j]:mask_indices[i,j]+mask_width] = 0
            inverse_mask[:, mask_indices[i,j]:mask_indices[i,j]+mask_width] = 1
        masks.append(mask)
        inverse_masks.append(inverse_mask)
        
    masks, inverse_masks = np.array(masks), np.array(inverse_masks)
    
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

def split_dataset(masked_data, unmasked_data, masks):
    n = masked_data.shape[0]  # Number of images in the dataset

    # Generate random indices for sampling without replacement
    indices = np.random.permutation(n)

    # Calculate the sizes of the three subsets
    n_train = int(0.8 * 0.8 * n)
    n_val = int(0.2 * 0.8 * n)
    n_test = n - n_train - n_val
    subset_sizes = [n_train, n_val, n_test]

    # Split the indices into three subarrays based on subset sizes
    sub_indices = np.split(indices, np.cumsum(subset_sizes)[:-1])
    
    # Use the indices to create the three subdatasets
    x_train, y_train, masks_train = masked_data[sub_indices[0]], unmasked_data[sub_indices[0]], masks[sub_indices[0]]
    x_val, y_val, masks_val = masked_data[sub_indices[1]], unmasked_data[sub_indices[1]], masks[sub_indices[1]]
    x_test, y_test, masks_test = masked_data[sub_indices[2]], unmasked_data[sub_indices[2]], masks[sub_indices[2]]
    
    return x_train, y_train, masks_train, x_val, y_val, masks_val, x_test, y_test, masks_test, indices