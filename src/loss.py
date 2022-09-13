import tensorflow as tf

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
            #mask_array = Y_true[:,:,:,2]
            #mask_array = tf.ones_like(Y_true[:,:,:,2]) - Y_true[:,:,:,2]
            mask_array = Y_true[:,:,:,2]

            Y_pred_real = tf.math.multiply(tf.cast(Y_pred[:,:,:,0] , tf.float64) , tf.cast(mask_array[:,:,:], tf.float64))
            Y_true_real= tf.math.multiply(tf.cast(Y_true[:,:,:,0], tf.float64) , tf.cast(mask_array[:,:,:], tf.float64))
            
            Y_pred_imag = tf.math.multiply(tf.cast(Y_pred[:,:,:,1], tf.float64), tf.cast(mask_array[:,:,:], tf.float64))
            Y_true_imag = tf.math.multiply(tf.cast(Y_true[:,:,:,1], tf.float64), tf.cast(mask_array[:,:,:], tf.float64))
            
            
            ground_truth_reconstructed = tf.complex(Y_true_real, Y_true_imag)
            
            predictions_reconstructed = tf.complex(Y_pred_real, Y_pred_imag)
            
            chi = ground_truth_reconstructed - predictions_reconstructed
            
            chi2 = tf.math.conj(chi) * chi
            
            return tf.math.real(tf.math.reduce_sum(chi2))

        return masked_loss
