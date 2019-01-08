from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

sys.path.append('../Common/')

import tensorflow as tf
import numpy as np
from dataManipulations import *
#from plotUtilities import *
from model import *

FLAGS = None

#deviceName = '/cpu:0'
#deviceName = '/:GPU:0'
deviceName = None
 
##############################################################################
##############################################################################
def runTraining(sess):

    train_step = tf.get_default_graph().get_operation_by_name("model/train/Adam")
    dropout_prob = tf.get_default_graph().get_operation_by_name("model/dropout_prob").outputs[0]
    trainingMode = tf.get_default_graph().get_operation_by_name("model/trainingMode").outputs[0]

    counter = 0
    while True:
        try:
            counter+=1
            sess.run([train_step], feed_dict={dropout_prob: FLAGS.dropout, trainingMode: True})
                                                         
        except tf.errors.OutOfRangeError:
            break
    print("Counter:",counter)       
##############################################################################
##############################################################################
def runValidation(sess, iEpoch, myWriter):

    #Fetch operations
    dataIter = tf.get_default_graph().get_operation_by_name("IteratorGetNext").outputs
    x = tf.get_default_graph().get_operation_by_name("x-input").outputs[0]
    y = tf.get_default_graph().get_operation_by_name("model/output/Identity").outputs[0]
 
    dropout_prob = tf.get_default_graph().get_operation_by_name("model/dropout_prob").outputs[0]
    trainingMode = tf.get_default_graph().get_operation_by_name("model/trainingMode").outputs[0]
    
    pull_mean = tf.get_default_graph().get_operation_by_name("model/performance/mean/count").outputs[0]
    pull_mean_update_op = tf.get_default_graph().get_operation_by_name("model/performance/mean/update_op").outputs[0]

    pull_variance = tf.get_default_graph().get_operation_by_name("model/performance/root_mean_squared_error/count").outputs[0]
    pull_variance_update_op = tf.get_default_graph().get_operation_by_name("model/performance/root_mean_squared_error/update_op").outputs[0]

    response = tf.get_default_graph().get_operation_by_name("model/performance/Reshape").outputs[0]

    loss = tf.get_default_graph().get_operation_by_name("model/train/total_loss").outputs[0]
    lossL2 = tf.get_default_graph().get_operation_by_name("model/train/get_regularization_penalty").outputs[0]

    mergedSummary = tf.get_default_graph().get_operation_by_name("monitor/Merge/MergeSummary").outputs[0]

    init_local = tf.local_variables_initializer()
    sess.run([init_local])

    counter = 0
    while True:
        try:
            counter+=1
            sess.run([pull_mean_update_op, pull_variance_update_op], feed_dict={dropout_prob: 0.0, trainingMode: False})
            result = sess.run([pull_variance, pull_mean, mergedSummary, loss, lossL2, dataIter], feed_dict={dropout_prob: 0.0, trainingMode: False})
            rms = result[0]
            mean = result[1]
            trainSummary = result[2]
            modelLoss = result[3]
            l2Loss = result[4]
            '''
            print("pull mean:", mean,
                  "pull RMS:", rms,
                  "L2 loss:",l2Loss,
                  "total loss:",modelLoss
            )
            '''
            #print("yTrue:",result[4][0])
            #print("x:",result[4][1])
            myWriter.add_summary(trainSummary, iEpoch)
                                              
        except tf.errors.OutOfRangeError:
            break

    print("Counter:",counter)    
    result = sess.run([pull_variance, pull_mean])
    print("pull mean:", result[0], "pull RMS:", result[1])
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
    batchSize = 1
    fileName = FLAGS.train_data_file
    nLabelBins = 512
    myDataManipulations = dataManipulations(fileName, nFolds, nEpochs, batchSize, nLabelBins,  smearMET=False)
    aDataIterator  = myDataManipulations.dataIterator.get_next()
    numberOfFeatures = myDataManipulations.numberOfFeatures
    nNeurons = [numberOfFeatures, 16, 16]
    nOutputNeurons = nLabelBins

    with tf.name_scope('model'): 
        myModel = Model(aDataIterator, nNeurons, nOutputNeurons, FLAGS.learning_rate, FLAGS.lambda_lagrange)

    init_global = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    sess.run([init_global, init_local])

    # Merge all the summaries and write them out to
    with tf.name_scope('monitor'): 
        merged = tf.summary.merge_all()
        
    myTrainWriter = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    myValidationWriter = tf.summary.FileWriter(FLAGS.log_dir + '/validation', sess.graph)
    ###############################################
    '''
    ops = tf.get_default_graph().get_operations()
    for op in ops:
        print(op.name)    
    
    print("tf.GraphKeys.UPDATE_OPS:")
        
    ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    for op in ops:
        print(op.name)
    '''
    #exit(0)    
    ###############################################
    for iEpoch in range(0,FLAGS.max_epoch):
        print("Epoch:",iEpoch)
        myDataManipulations.initializeDataIteratorForCVFold(sess, aFold=0, trainingMode=True)
        runTraining(sess)

        if iEpoch%1==0:
            myDataManipulations.initializeDataIteratorForCVFold(sess, aFold=0, trainingMode=False)
            runValidation(sess, iEpoch, myTrainWriter)


    myTrainWriter.close()
    myValidationWriter.close()
    ###############################################
    
    # Save the model to disk.
    x = tf.get_default_graph().get_operation_by_name("x-input").outputs[0]
    y = tf.get_default_graph().get_operation_by_name("model/output/Identity").outputs[0]
    yTrue = tf.get_default_graph().get_operation_by_name("y-input").outputs[0]
    
    tf.saved_model.simple_save(sess, FLAGS.model_dir,
                               inputs={"x": x, "yTrue": yTrue},
                               outputs={"y": y})
    print("Model saved in file: %s" % FLAGS.model_dir)    
       
    return
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
  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
##############################################################################
##############################################################################
