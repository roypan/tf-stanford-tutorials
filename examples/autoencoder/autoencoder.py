import tensorflow as tf

from layers import *

def encoder(input):
    # Create a conv network with 3 conv layers and 1 FC layer
    # Conv 1: filter: [3, 3, 1], stride: [2, 2], relu
    conv1 = conv(input, 'conv1', [3, 3, 1], [2, 2], padding='SAME', non_linear_fn=tf.nn.relu)
    # Conv 2: filter: [3, 3, 8], stride: [2, 2], relu
    conv2 = conv(conv1, 'conv2', [3, 3, 8], [2, 2], padding='SAME', non_linear_fn=tf.nn.relu)
    # Conv 3: filter: [3, 3, 8], stride: [2, 2], relu
    conv3 = conv(conv2, 'conv3', [3, 3, 8], [2, 2], padding='SAME', non_linear_fn=tf.nn.relu)
    # FC: output_dim: 100, no non-linearity
    en_fc = fc(conv3, 'en_fc', out_dim=100, non_linear_fn=tf.identity)
    return en_fc

def decoder(input):
    # Create a deconv network with 1 FC layer and 3 deconv layers
    # FC: output dim: 128, relu
    de_fc = fc(input, 'de_fc', out_dim=128, non_linear_fn=tf.nn.relu)
    # Reshape to [batch_size, 4, 4, 8]
    de_fc = tf.reshape(de_fc, shape=[-1, 4, 4, 8]) 
    # Deconv 1: filter: [3, 3, 8], stride: [2, 2], relu
    deconv1 = deconv(de_fc, 'deconv1', [3, 3, 8], [2, 2], padding='SAME', non_linear_fn=tf.nn.relu)
    # Deconv 2: filter: [8, 8, 1], stride: [2, 2], padding: valid, relu
    deconv2 = deconv(deconv1, 'deconv2', [8, 8, 1], [2, 2], padding='VALID', non_linear_fn=tf.nn.relu)
    # Deconv 3: filter: [7, 7, 1], stride: [1, 1], padding: valid, sigmoid
    deconv3 = deconv(deconv2, 'deconv3', [7, 7, 1], [1, 1], padding='VALID', non_linear_fn=tf.nn.sigmoid)
    return deconv3
    
def autoencoder(input_shape):
    # Define place holder with input shape
    with tf.name_scope('data'):
        input_image = tf.placeholder(tf.float32, input_shape, name="input_placeholder")
    # Define variable scope for autoencoder
    with tf.variable_scope('autoencoder') as scope:
        # Pass input to encoder to obtain encoding
        en_out = encoder(input_image)
        # Pass encoding into decoder to obtain reconstructed image
        de_out = decoder(en_out)
        # Return input image (placeholder) and reconstructed image
    return [input_image, de_out]
