import tensorflow as tf
from utils import Golabal_Variable as GV
"""This script defines basic operations.
"""
################################################################################
# Basic operations building the network
################################################################################
def Pool3d(inputs, kernel_size, strides):
    """Performs 3D max pooling."""

    return tf.layers.max_pooling3d(
            inputs=inputs,
            pool_size=kernel_size,
            strides=strides,
            padding='same')
def deconv3d(inputs, filters,kernel_size,strides,out_shape,
                        stddev=0.05, name="deconv3d"):
  with tf.variable_scope(name):
    w = tf.get_variable('kernel', [kernel_size,kernel_size,kernel_size, filters, inputs.get_shape()[-1]],
                          initializer=tf.random_normal_initializer(stddev=stddev))
    deconv= tf.nn.conv3d_transpose(inputs, w,  output_shape=out_shape,
                                          strides=[1,strides,strides,strides, 1], padding="SAME")
    biases = tf.get_variable('biases',  [out_shape[-1]],
                                          initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
    return deconv

def Deconv3D(inputs, filters,kernel_size,strides, out_shape,
                 stddev=0.05, name="deconv3d",use_bias=True):
  with tf.variable_scope(name):
    w = tf.get_variable('kernel',
                        [kernel_size,kernel_size,kernel_size, filters, inputs.get_shape()[-1]],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
    deconv= tf.nn.conv3d_transpose(inputs, w, output_shape=out_shape,
                                          strides=[1,strides,strides,strides,1], padding="SAME")
    biases = tf.get_variable('biases', [out_shape[-1]],
                                             initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
    return deconv

def Conv33d(inputs, filters, kernel_size,
           strides, stddev=0.05, name="conv3d",use_bias=True,reuse=False):
  with tf.variable_scope(name):
    w = tf.get_variable('kernel',[kernel_size,kernel_size,kernel_size, inputs.get_shape()[-1], filters],
                          initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv3d(inputs,w, strides=[1, strides,strides,strides, 1],
                        padding='SAME')
    biases = tf.get_variable('biases', [filters],
                                    initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
    return conv

def Conv3D(inputs, filters,kernel_size,
           strides, stddev=0.05, name="conv3d",use_bias=True,reuse=False):
  with tf.variable_scope(name):
    w = tf.get_variable('kernel',[kernel_size,kernel_size,kernel_size, inputs.get_shape()[-1], filters],
                         initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv3d(inputs, w, strides=[1, strides,strides,strides, 1], padding='SAME')
    biases = tf.get_variable('biases', [filters],
                                    initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
    return conv

def Dilated_Conv3D(inputs, filters, kernel_size,dilation_rate, strides, stddev=0.05, name="conv3d", use_bias=False):
    """Performs 3D dilated convolution without bias and activation function."""
    with tf.variable_scope(name):
        w = tf.get_variable('kernel', [kernel_size, kernel_size, kernel_size, inputs.get_shape()[-1], filters],
                            use_bias=use_bias,dilation_rate=dilation_rate,
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv3d(inputs, w, strides=[1, strides, strides, strides, 1],
                             padding='SAME')
        biases = tf.get_variable('biases', [filters],
                                        initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        return conv


def lrelu(x, leak=0.5, name="lrelu"):
  return tf.maximum(x, leak*x)

def BN_ReLU(inputs, training):
    """Performs a batch normalization followed by a ReLU6."""
    # We set fused=True for a significant performance boost. See
    # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
    inputs = tf.layers.batch_normalization(
                inputs=inputs,
                axis=-1,
                momentum=0.97,
                epsilon=1e-5,
                center=True,
                scale=True,
                training=training,
                fused=True)
    return tf.nn.relu6(inputs)

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum,
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)
