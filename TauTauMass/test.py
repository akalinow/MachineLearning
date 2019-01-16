from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse, textwrap
import os
import sys

sys.path.append('../Common/')

import tensorflow as tf
import numpy as np
from dataManipulations import *
from model import *

FLAGS = None

#deviceName = '/cpu:0'
#deviceName = '/:GPU:0'
deviceName = None
 
##############################################################################
##############################################################################
def runDebug(sess):

    testDropout = 0.0
    trainingMode = False

    #myDataManipulations.initializeDataIteratorForCVFold(sess, aFold=0, trainingMode=trainingMode)
    
    train_step = tf.get_default_graph().get_operation_by_name("model/train/Adam")
    dropout_prob = tf.get_default_graph().get_operation_by_name("model/dropout_prob").outputs[0]
    trainingModeFlag = tf.get_default_graph().get_operation_by_name("model/trainingMode").outputs[0]
    
    dataIter = tf.get_default_graph().get_operation_by_name("IteratorGetNext").outputs
    y = tf.get_default_graph().get_operation_by_name("model/output/Identity").outputs[0]
    response = tf.get_default_graph().get_operation_by_name("model/performance/response").outputs[0]

    init_local = tf.local_variables_initializer()
    sess.run([init_local])

    ops = [dataIter, response]
    if trainingMode:
        ops.insert(0, train_step)

    result = []    
    while True:
        try:
            result = sess.run(ops, feed_dict={dropout_prob: testDropout, trainingModeFlag: False})
            print(result)
            print("labels:",result[0][0])
            print("features:",result[0][1])
            print("model result:",result[1])
            break
        except tf.errors.OutOfRangeError:
            break


##############################################################################
##############################################################################
def train():

    sess = tf.Session()

    print("Available devices:")
    devices = sess.list_devices()
    for d in devices:
        print(d.name)

    nFolds = 2 #data split into equal training and validation parts
    nEpochs = FLAGS.max_epoch
    batchSize = 1024
    fileName = FLAGS.train_data_file
    nLabelBins = 1
    myDataManipulations = dataManipulations(fileName, nFolds, batchSize, nLabelBins,  smearMET=False)
    aDataIterator  = myDataManipulations.dataIterator.get_next()
    numberOfFeatures = myDataManipulations.numberOfFeatures
    nNeurons = [numberOfFeatures, 16, 16]
    nOutputNeurons = nLabelBins

    with tf.name_scope('model'):
        myModel = Model(aDataIterator, nNeurons, nOutputNeurons, FLAGS.learning_rate, FLAGS.lambda_lagrange)

    init_global = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    sess.run([init_global, init_local])
        
    myTrainWriter = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    myValidationWriter = tf.summary.FileWriter(FLAGS.log_dir + '/validation', sess.graph)
   
    for iEpoch in range(0,FLAGS.max_epoch):

        sess.run([init_local])
        myDataManipulations.initializeDataIteratorForCVFold(sess, aFold=0, trainingMode=True)
        runTraining(sess, iEpoch, myTrainWriter, FLAGS, trainingMode=True)

        if iEpoch%10==0:
            myDataManipulations.initializeDataIteratorForCVFold(sess, aFold=0, trainingMode=False)
            runTraining(sess, iEpoch, myValidationWriter, FLAGS, trainingMode=False)

    myTrainWriter.close()
    myValidationWriter.close()

    if FLAGS.debug>0:
        listOperations()
    if FLAGS.debug>1:
        runDebug(sess)

    saveTheModel(sess, FLAGS)
    
##############################################################################
##############################################################################
##############################################################################
def main(_):

  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)

  if tf.gfile.Exists(FLAGS.model_dir):
    tf.gfile.DeleteRecursively(FLAGS.model_dir)

  train()
##############################################################################
##############################################################################
if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--max_epoch', type=int, default=50,
                      help='Number of epochs')

  parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Initial learning rate')

  parser.add_argument('--lambda_lagrange', type=float, default=0.1,
                      help='Largange multipler for L2 loss')

  parser.add_argument('--dropout', type=float, default=0.2,
                      help='Drop probability for training dropout.')

  parser.add_argument('--train_data_file', type=str,
                      default=os.path.join(os.getenv('PWD', './'),
                                           'data/htt_features_train.pkl'),
                      help='Directory for storing training data')

  parser.add_argument('--model_dir', type=str,
                      default=os.path.join(os.getenv('PWD', './'),
                                           'model'),
                      help='Directory for storing model state')

  parser.add_argument('--log_dir', type=str,
                      default=os.path.join(os.getenv('PWD', './'),
                                           'logs'),
                      help='Summaries log directory')

  parser.add_argument('--debug', type=int, default=0,
                      help=textwrap.dedent('''\
                      Runs debug methods: 
                      0 - disabled 
                      1 - list graph operations 
                      2 - run debug method defined by the user'''))
  
  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
##############################################################################
##############################################################################
