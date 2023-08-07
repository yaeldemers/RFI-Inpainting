import tensorflow as tf

class CustomFullyConnectedLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation=None):
        super(CustomFullyConnectedLayer, self).__init__()
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        # Create two sets of weights and biases for each input tensor
        input_1_dim, input_2_dim = input_shape[0][-1], input_shape[1][-1]
        self.kernel_1 = self.add_weight("kernel_1", shape=[input_1_dim, self.units])
        self.bias_1 = self.add_weight("bias_1", shape=[self.units])
        self.kernel_2 = self.add_weight("kernel_2", shape=[input_2_dim, self.units])
        self.bias_2 = self.add_weight("bias_2", shape=[self.units])

    def call(self, inputs):
        # Unpack the inputs
        input_1, input_2 = inputs

        # Perform fully connected operations for each input
        fc_output_1 = tf.matmul(input_1, self.kernel_1) + self.bias_1
        fc_output_2 = tf.matmul(input_2, self.kernel_2) + self.bias_2

        # Combine the outputs from both inputs
        combined_output = fc_output_1 + fc_output_2

        if self.activation is not None:
            combined_output = self.activation(combined_output)

        return combined_output