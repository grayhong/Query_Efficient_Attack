import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm, variance_scaling_initializer

#the implements of leakyRelu
def lrelu(x , alpha = 0.2 , name="LeakyReLU"):
    return tf.maximum(x , alpha*x)

#squash function
def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector

# Wrappers
def conv2d_t(x, W, b, strides = 1):
    x = tf.nn.conv2d(x, W, strides = [1, strides, strides, 1], padding = "SAME")
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k = 2):
    return tf.nn.max_pool(x, ksize = [1, k, k, 1], strides = [1, k, k, 1], padding = "SAME")


def conv2d(input_, output_dim, k_h=3, k_w=3, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)
        
        return conv

def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer = tf.random_normal_initializer(stddev = stddev))

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape = output_shape, strides = [1, d_h, d_w, 1])
        biases = tf.get_variable('biases', [output_shape[-1]], initializer = tf.constant_initializer(0.0))
        deconv = tf.nn.bias_add(deconv, biases)

        return deconv

def fully_connect(input_, output_size, scope=None, with_w=False):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 variance_scaling_initializer())
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(0.0))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    
    tile_shape=tf.stack([1, x_shapes[1], x_shapes[2], 1])
    
    return tf.concat([x , tf.tile(y, tile_shape)], 3)
    #return tf.concat([x , y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2] , y_shapes[3]])], 3)

def batch_normal(input , scope="scope" , reuse=False):
    return batch_norm(input , epsilon=1e-5, decay=0.9 , scale=True, scope=scope , reuse = reuse , updates_collections=None)