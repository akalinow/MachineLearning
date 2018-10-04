import tensorflow as tf
import numpy as np

##############################################################################
##############################################################################
##############################################################################
def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    #initial = tf.truncated_normal(shape, stddev=0.1)
    numberOfInputs = shape[0]
    initial = tf.random_uniform(shape)*np.sqrt(2.0/numberOfInputs)
    return tf.Variable(initial)
##############################################################################
##############################################################################
##############################################################################
def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)
##############################################################################
##############################################################################
##############################################################################
def variable_summaries(var):
    return #temporary switch off
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)
##############################################################################
##############################################################################
##############################################################################
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.

    It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
      # This Variable will hold the state of the weights for the layer
      with tf.name_scope('weights'):
        weights = weight_variable([input_dim, output_dim])
        variable_summaries(weights)
      with tf.name_scope('biases'):
        biases = bias_variable([output_dim])
        variable_summaries(biases)
      with tf.name_scope('Wx_plus_b'):
        preactivate = tf.matmul(input_tensor, weights) + biases
        tf.summary.histogram('pre_activations', preactivate)
      activations = act(preactivate, name='activation')
      tf.summary.histogram('activations', activations)
      return activations
##############################################################################
##############################################################################
##############################################################################
