import tensorflow as tf
import matplotlib.pyplot as plt
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
        end=len(logs)
    
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



class custom_loss:
    def wrapper(self):

        def masked_loss(Y_true, Y_pred):
            
            """
            TODO: try using Dice Coeff instead
            """
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
            #mask_array = Y_true[:,:,:,2]
            #mask_array = tf.ones_like(Y_true[:,:,:,2]) - Y_true[:,:,:,2]
           
            
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