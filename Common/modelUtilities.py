import tensorflow as tf
import numpy as np
import argparse, textwrap

from dataManipulations import *
from model import *

##############################################################################
##############################################################################
class Framework:

##############################################################################
##############################################################################
    def initializeParser(self):

        self.myParser = argparse.ArgumentParser()

        self.myParser.add_argument('--max_epoch', type=int, default=50,
                                   help='Number of epochs')

        self.myParser.add_argument('--learning_rate', type=float, default=0.001,
                                   help='Initial learning rate')
        
        self.myParser.add_argument('--lambda_lagrange', type=float, default=0.1,
                                   help='Largange multipler for L2 loss')

        self.myParser.add_argument('--dropout', type=float, default=0.2,
                                   help='Drop probability for training dropout.')

        self.myParser.add_argument('--train_data_file', type=str,
                                   default=os.path.join(os.getenv('PWD', './'),
                                                        'data/htt_features_train.pkl'),
                                   help='Directory for storing training data')

        self.myParser.add_argument('--model_dir', type=str,
                                   default=os.path.join(os.getenv('PWD', './'),
                                                        'model'),
                                   help='Directory for storing model state')
        
        self.myParser.add_argument('--log_dir', type=str,
                                   default=os.path.join(os.getenv('PWD', './'),
                                                        'logs'),
                                   help='Summaries log directory')
        
        self.myParser.add_argument('--debug', type=int, default=0,
                                   help=textwrap.dedent('''\
                                   Runs debug methods: 
                                   0 - disabled 
                                   1 - list graph operations 
                                   2 - run debug method defined by the user'''))        
##############################################################################
##############################################################################
def parse_known_args(self):
    
    self.FLAGS, self.unparsed = self.myParser.parse_known_args()

##############################################################################
##############################################################################
def __init__(self):

         initializeParser()


##############################################################################
##############################################################################
def listOperations():

    print("All operations:")
    ops = tf.get_default_graph().get_operations()
    for op in ops:
        print(op.name)    

    print("Losses:")
    ops = tf.get_collection(tf.GraphKeys.LOSSES)
    for op in ops:
        print(op.name)
##############################################################################
##############################################################################
def saveTheModel(sess, flags):

    x = tf.get_default_graph().get_operation_by_name("data/x-input").outputs[0]
    y = tf.get_default_graph().get_operation_by_name("model/output/Identity").outputs[0]
    yTrue = tf.get_default_graph().get_operation_by_name("data/y-input").outputs[0]
    
    tf.saved_model.simple_save(sess, flags.model_dir,
                               inputs={"x": x, "yTrue": yTrue},
                               outputs={"y": y})
    print("Model saved in file: %s" % flags.model_dir) 
##############################################################################
##############################################################################
def runTraining(sess, iEpoch, myWriter, flags, trainingMode=True):
    
    train_step = tf.get_default_graph().get_operation_by_name("model/train/Adam")
    dropout_prob = tf.get_default_graph().get_operation_by_name("model/dropout_prob").outputs[0]
    trainingModeFlag = tf.get_default_graph().get_operation_by_name("model/trainingMode").outputs[0]
    mergedSummary = tf.get_default_graph().get_operation_by_name("model/monitor/Merge/MergeSummary").outputs[0]

    ops = tf.get_collection("MY_UPDATE_OPS")
    if trainingMode:
        ops.insert(0, train_step)
    ops.append(mergedSummary)

    result = []
    while True:
        try:
            result = sess.run(ops, feed_dict={dropout_prob: flags.dropout, trainingModeFlag: trainingMode})
        except tf.errors.OutOfRangeError:
            break
        
    if iEpoch%10==0:
        print("Epoch:",iEpoch)
        if trainingMode:
            print("Training:")
        else:
            print("Validation:")

        trainSummary = result[-1]
        myWriter.add_summary(trainSummary, iEpoch)  
    
        ops = tf.get_collection("MY_RUNNING_VALS")
        result = sess.run(ops)

        for index, aOp in enumerate(ops):
            print(aOp.name, result[index])
##############################################################################
##############################################################################            




def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    numberOfInputs = shape[0]
    initial = tf.random_uniform(shape)*np.sqrt(2.0/numberOfInputs)
    return tf.Variable(initial)
##############################################################################
##############################################################################
def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.01, shape=shape)    
    return tf.Variable(initial)
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
def nn_layer(input_tensor, input_dim, output_dim, layer_name, trainingMode, act=tf.nn.relu):
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
        #preactivate = tf.layers.batch_normalization(preactivate, training=trainingMode)  
        tf.summary.histogram('pre_activations', preactivate)

      activations = act(preactivate, name='activation')
      tf.summary.histogram('activations', activations)
      return activations
##############################################################################
##############################################################################
