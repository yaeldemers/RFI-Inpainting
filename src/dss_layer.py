from tensorflow.keras.layers import Conv2D, Add
from tensorflow import keras

class DSS(keras.layers.Layer):
    def __init__(self, filters, name=None):
        super(DSS, self).__init__()
        self.conv = Conv2D(filters, kernel_size=3, padding='same', activation="relu", kernel_initializer='he_normal')
        self.add = Add()
        if name: self._name = name

    def call(self, inputs):
        input_a, input_b = inputs
        
        # applies a Conv2D on the inputs
        conv_1a = self.conv(input_a)
        
        conv_1b = self.conv(input_b)
        
        # sums the original inputs
        merged = self.add([input_a, input_b])
        
        # applies a Conv2D to the sum above
        conv_2 = self.conv(merged)
        
        # sums the previous convulutions
        output_a = self.add([conv_1a, conv_2])
        
        output_b = self.add([conv_1b, conv_2])
        
        return output_a, output_b