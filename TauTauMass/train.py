from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow as tf
import numpy as np
from dataManipulations import *
from plotUtilities import *
from model import *

FLAGS = None

#deviceName = '/cpu:0'
#deviceName = '/:GPU:0'
deviceName = None
 
##############################################################################
##############################################################################
##############################################################################
def runCVFold(sess, iFold, myDataManipulations, myTrainWriter, myValidationWriter):
    #Fetch operations
    x = tf.get_default_graph().get_operation_by_name("input/x-input").outputs[0]
    y = tf.get_default_graph().get_operation_by_name("model/output/Identity").outputs[0]
 
    yTrue = tf.get_default_graph().get_operation_by_name("input/y-input").outputs[0]
    keep_prob = tf.get_default_graph().get_operation_by_name("model/dropout/Placeholder").outputs[0]

    train_step = tf.get_default_graph().get_operation_by_name("model/train/Adam")

    pull_mean = tf.get_default_graph().get_operation_by_name("model/performance/moments/mean").outputs[0]
    pull_variance = tf.get_default_graph().get_operation_by_name("model/performance/moments/variance").outputs[0]

    loss = tf.get_default_graph().get_operation_by_name("model/train/total_loss").outputs[0]
    lossL2 = tf.get_default_graph().get_operation_by_name("model/train/get_regularization_penalty").outputs[0]

    mergedSummary = tf.get_default_graph().get_operation_by_name("monitor/Merge/MergeSummary").outputs[0]

    aTrainIterator, aValidationIterator = myDataManipulations.getCVFold(sess, iFold)
    numberOfBatches = myDataManipulations.numberOfBatches

    #Train
    iBatch = -1
    iEpoch = 0
    while True:
        try:
            xs, ys = makeFeedDict(sess, aTrainIterator)
            iBatch+=1
            iEpoch = (int)(iBatch/numberOfBatches)

            sess.run([train_step], feed_dict={x: xs, yTrue: ys, keep_prob: FLAGS.dropout})

            #Evaluate training performance
            if(iEpoch%10==0 and iBatch%numberOfBatches==0):
                result = sess.run([pull_variance, mergedSummary, loss], feed_dict={x: xs, yTrue: ys, keep_prob: 1.0})
                iStep = iEpoch + iFold*FLAGS.max_epoch
                variance = result[0][0]
                trainSummary = result[1]
                modelLoss = result[2]
                myTrainWriter.add_summary(trainSummary, iStep)
                print("Epoch number:",iEpoch,
                      "pull RMS:", np.sqrt(variance),
                      "total loss:",modelLoss)
                                              
        except tf.errors.OutOfRangeError:
            break
    #########################################
    #Evaluate performance on validation data
    try:
        xs, ys = makeFeedDict(sess, aValidationIterator)
        result = sess.run([pull_mean, pull_variance,  mergedSummary],
                          feed_dict={x: xs, yTrue: ys, keep_prob: 1.0})
        mean = result[0]
        variance = result[1]
        validationSummary = result[2]
        iStep = (iFold+1)*FLAGS.max_epoch - 1
        myValidationWriter.add_summary(validationSummary, iStep)
        
        print("Validation. Fold:",iFold,
              "Epoch:",iEpoch,
              "pull mean:", mean,
              "pull RMS:", np.sqrt(variance))
        
        result = sess.run([y, yTrue], feed_dict={x: xs, yTrue: ys, keep_prob: 1.0})
        modelResult = result[0]
        labels = result[1]            
        plotDiscriminant(modelResult, labels, "Validation")
    except tf.errors.OutOfRangeError:
        print("OutOfRangeError")
##############################################################################
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
    batchSize = 128
    fileName = FLAGS.train_data_file
    myDataManipulations = dataManipulations(fileName, nFolds, nEpochs, batchSize, smearMET=True)        
    numberOfFeatures = myDataManipulations.numberOfFeatures
    nNeurons = [numberOfFeatures, 128, 128]

    # Input placeholders
    with tf.name_scope('input'): 
        x = tf.placeholder(tf.float32, name='x-input')
        yTrue = tf.placeholder(tf.float32, name='y-input')

    with tf.name_scope('model'): 
        myModel = Model(x, yTrue, nNeurons, FLAGS.learning_rate, FLAGS.lambda_lagrange)

    init = tf.global_variables_initializer()
    sess.run(init)
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
    '''
    ###############################################
    sess.run(init)
    iFold = 0
    runCVFold(sess, iFold, myDataManipulations, myTrainWriter, myValidationWriter)

    myTrainWriter.close()
    myValidationWriter.close()
    # Save the model to disk.
    y = tf.get_default_graph().get_operation_by_name("model/output/Identity").outputs[0]
    
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

  parser.add_argument('--dropout', type=float, default=0.9,
                      help='Keep probability for training dropout.')

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
