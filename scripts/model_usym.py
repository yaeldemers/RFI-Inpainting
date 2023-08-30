"""
model_usym.py

Description: First tentative of building USYM network following the UNET architecture and DSS layers

Authors: Yael-Jeremy Demers

Last Modified: 22-08-2023

Changes:
- 22-08-2023 Demers: Fixing reversed concatenation layer input order
- 15-08-2023 Demers: Setting up dropout rate on each step to 20% 
- 04-11-2022 Demers: Removing the second DSS block in a row in each layer of the U-Net
"""

from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Dropout, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dense
from dss_layer import DSS

class USYM: 
    '''
        Multi-input U-net architecture where Conv2D layers are replaced by custom DSS ones
    '''
    def __init__(self, inputs, loss_class, weights_dir):
        # inherit from Functional API
        super(USYM, self).__init__()
        self.inputs = inputs
        self.weights_dir = weights_dir
        
        masked_loss  = loss_class.wrapper()

        input_a, input_b = self.inputs

        #--------- Initial step of the U-pattern -------#
        dss_1 = DSS(filters=64, name="dss_1-1")([input_a, input_b])
        #dss_1 = DSS(filters=64, name="dss_1-2")([dss_1[0], dss_1[1]])
        pool_1a = MaxPooling2D((2,2), name="maxpool_1a")(dss_1[0])
        pool_1a = Dropout(0.2, name="dropout_1a")(pool_1a)
        pool_1b = MaxPooling2D((2,2), name="maxpool_1b")(dss_1[1])
        pool_1b = Dropout(0.2, name="dropout_1b")(pool_1b)

        #--------- drop a step in the U-pattern --------#
        dss_2 = DSS(filters=128, name="dss_2-1")([pool_1a, pool_1b])
        pool_2a = MaxPooling2D((2,2), name="maxpool_2a")(dss_2[0])
        pool_2a = Dropout(0.2, name="dropout_2a")(pool_2a)
        pool_2b = MaxPooling2D((2,2), name="maxpool_2b")(dss_2[1])
        pool_2b = Dropout(0.2, name="dropout_2b")(pool_2b)

        #--------- drop a step in the U-pattern --------#
        dss_3 = DSS(filters=256, name="dss_3-1")([pool_2a, pool_2b])
        pool_3a = MaxPooling2D((2,2), name="maxpool_3a")(dss_3[0])
        pool_3a = Dropout(0.2, name="dropout_3a")(pool_3a)
        pool_3b = MaxPooling2D((2,2), name="maxpool_3b")(dss_3[1])
        pool_3b = Dropout(0.2, name="dropout_3b")(pool_3b)

        #--------- drop a step in the U-pattern --------#
        dss_4 = DSS(filters=512, name="dss_4-1")([pool_3a, pool_3b])
        pool_4a = MaxPooling2D((2,2), name="maxpool_4a")(dss_4[0])
        pool_4a = Dropout(0.2, name="dropout_4a")(pool_4a)
        pool_4b = MaxPooling2D((2,2), name="maxpool_4b")(dss_4[1])
        pool_4b = Dropout(0.2, name="dropout_4b")(pool_4b)

        #--------- drop a step in the U-pattern --------#
        #---------    bottom of the U-pattern   --------#
        dss_bottleneck = DSS(filters=1024, name="dss_bot-1")([pool_4a, pool_4b])

        #--------- climb a step in the U-pattern -------#
        deconv_4a = Conv2DTranspose(512, 2, strides=(2,2), kernel_initializer='he_normal', name="deconv_4a")(dss_bottleneck[0])
        deconv_4b = Conv2DTranspose(512, 2, strides=(2,2), kernel_initializer='he_normal', name="deconv_4b")(dss_bottleneck[1])
        uconv_4a = concatenate([deconv_4a, dss_4[0]], name="concat_4a")
        uconv_4a = Dropout(0.2, name="dropout_up_4a")(uconv_4a)
        uconv_4b = concatenate([deconv_4b, dss_4[1]], name="concat_4b")
        uconv_4b = Dropout(0.2, name="dropout_up_4b")(uconv_4b)
        dss_up_4 = DSS(filters=512, name="updss_4-1")([uconv_4a, uconv_4b])

        #--------- climb a step in the U-pattern -------#
        deconv_3a = Conv2DTranspose(256, 2, strides=(2,2), kernel_initializer='he_normal', name="deconv_3a")(dss_up_4[0])
        deconv_3b = Conv2DTranspose(256, 2, strides=(2,2), kernel_initializer='he_normal', name="deconv_3b")(dss_up_4[1])
        uconv_3a = concatenate([dss_3[0], deconv_3a], name="concat_3a")
        uconv_3a = Dropout(0.2, name="dropout_up_3a")(uconv_3a)
        uconv_3b = concatenate([dss_3[1], deconv_3b], name="concat_3b")
        uconv_3b = Dropout(0.2, name="dropout_up_3b")(uconv_3b)
        dss_up_3 = DSS(filters=256, name="updss_3-1")([uconv_3a, uconv_3b])

        #--------- climb a step in the U-pattern -------#
        deconv_2a = Conv2DTranspose(128, 2, strides=(2,2), kernel_initializer='he_normal', name="deconv_2a")(dss_up_3[0])
        deconv_2b = Conv2DTranspose(128, 2, strides=(2,2), kernel_initializer='he_normal', name="deconv_2b")(dss_up_3[1])
        uconv_2a = concatenate([dss_2[0], deconv_2a], name="concat_2a")
        uconv_2a = Dropout(0.2, name="dropout_up_2a")(uconv_2a)
        uconv_2b = concatenate([dss_2[1], deconv_2b], name="concat_2b")
        uconv_2b = Dropout(0.2, name="dropout_up_2b")(uconv_2b)
        dss_up_2 = DSS(filters=128, name="updss_2-1")([uconv_2a, uconv_2b])

        #--------- climb a step in the U-pattern -------#
        #---------     top of the U-pattern     --------#
        deconv_1a = Conv2DTranspose(64, 2, strides=(2,2), kernel_initializer='he_normal', name="deconv_1a")(dss_up_2[0])
        deconv_1b = Conv2DTranspose(64, 2, strides=(2,2), kernel_initializer='he_normal', name="deconv_1b")(dss_up_2[1])
        uconv_1a = concatenate([dss_1[0], deconv_1a], name="concat_1a")
        uconv_1a = Dropout(0.2, name="dropout_up_1a")(uconv_1a)
        uconv_1b = concatenate([dss_1[1], deconv_1b], name="concat_1b")
        uconv_1b = Dropout(0.2, name="dropout_up_1b")(uconv_1b)
        dss_up_1 = DSS(filters=64, name="updss_1-1")([uconv_1a, uconv_1b])

        #-------- reshaping the data to fit inputs ------#
        output_a = Conv2D(3, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal', name="output_a")(dss_up_1[0])
        output_b = Conv2D(3, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal', name="output_b")(dss_up_1[1])

        #-------- dense layers to allow neg values ------#
        output_a = Dense(64, activation='relu')(output_a)
        output_a = Dense(2)(output_a)
        output_b = Dense(64, activation='relu')(output_b)
        output_b = Dense(2)(output_b)
        
        self.model = Model(inputs, [output_a, output_b])
        self.model._name = "U-SYM-V0.7"
        
        #----------- handling missing weights -----------#
        if self.weights_dir != None:
            try:
                print('Loading weights' )
                self.model.load_weights(self.weights_dir)
            except:
                pass
                print('No weights to load' , flush = True)
        
        #--------------- compiling the model -------------#
        self.model.compile(optimizer = Adam(learning_rate = 1e-4), loss = masked_loss , metrics = [ masked_loss ])

        #------------- visualising the model ------------#
        #CNN.summary()
        #keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
        #keras.utils.plot_model(CNN, "multi_input_and_output_model.png")