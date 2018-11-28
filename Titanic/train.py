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
    #y = tf.get_default_graph().get_operation_by_name("model/output/Identity").outputs[0]
 
    yTrue = tf.get_default_graph().get_operation_by_name("input/y-input").outputs[0]
    keep_prob = tf.get_default_graph().get_operation_by_name("model/dropout/Placeholder").outputs[0]
    trainingMode = tf.get_default_graph().get_operation_by_name("model/Placeholder").outputs[0]

    train_step = tf.get_default_graph().get_operation_by_name("model/train/Adam")

    loss = tf.get_default_graph().get_operation_by_name("model/train/total_loss").outputs[0]
    lossL2 = tf.get_default_graph().get_operation_by_name("model/train/get_regularization_penalty").outputs[0]
    accuracy = tf.get_default_graph().get_operation_by_name("model/performance/Mean").outputs[0]
    y = tf.get_default_graph().get_operation_by_name("model/performance/Sigmoid").outputs[0]

    mergedSummary = tf.get_default_graph().get_operation_by_name("monitor/Merge/MergeSummary").outputs[0]

    aTrainIterator, aValidationIterator = myDataManipulations.getCVFold(sess, iFold)
    numberOfBatches = myDataManipulations.numberOfBatches
    accuracyValue = 0
    
    #Train
    iBatch = -1
    iEpoch = 0
    while True:
        try:
            xs, ys = makeFeedDict(sess, aTrainIterator)
            iBatch+=1
            iEpoch = (int)(iBatch/numberOfBatches)

            sess.run([train_step], feed_dict={x: xs, yTrue: ys, keep_prob: FLAGS.dropout, trainingMode: True})

            #Evaluate training performance
            if(iEpoch%100==0 and iBatch%numberOfBatches==0):
                iStep = iEpoch + iFold*FLAGS.max_epoch
                resultTrain = sess.run([mergedSummary, accuracy, lossL2, loss, y, yTrue], feed_dict={x: xs, yTrue: ys, keep_prob: 1.0, trainingMode: False})
                
                xs, ys = makeFeedDict(sess, aValidationIterator)                
                resultValidation = sess.run([mergedSummary,accuracy],
                                            feed_dict={x: xs, yTrue: ys, keep_prob: 1.0, trainingMode: False})
                
                myTrainWriter.add_summary(resultTrain[0], iStep)
                myValidationWriter.add_summary(resultValidation[0], iStep)
                print("Epoch, bunch number:",iEpoch,iBatch)
                print("     Train accuracy:", resultTrain[1],
                      "regularisation loss",resultTrain[2],
                      "total loss:",resultTrain[3])
                print("Validation accuracy:", resultValidation[1])
                                              
        except tf.errors.OutOfRangeError:
            break
    #########################################
    #Evaluate performance on validation data
    try:
        xs, ys = makeFeedDict(sess, aValidationIterator)
        result = sess.run([accuracy,  mergedSummary,  y, yTrue],
                          feed_dict={x: xs, yTrue: ys, keep_prob: 1.0, trainingMode: False})
        accuracyValue = result[0]
        validationSummary = result[1]
        iStep = (iFold+1)*FLAGS.max_epoch - 1
        myValidationWriter.add_summary(validationSummary, iStep)
                
        print("Validation. Fold:",iFold,
              "Epoch:",iEpoch,
              "accuracy:", accuracyValue)
        
        plotDiscriminant(result[2], result[3], "Validation", doBlock=True)
        
    except tf.errors.OutOfRangeError:
        print("OutOfRangeError")

    return accuracyValue
##############################################################################
##############################################################################
##############################################################################
def train():

    sess = tf.Session()

    print("Available devices:")
    devices = sess.list_devices()
    for d in devices:
        print(d.name)

    nFolds = 2
    nEpochs = FLAGS.max_epoch
    batchSize = 64
    fileName = FLAGS.train_data_file
    myDataManipulations = dataManipulations(fileName, nFolds, nEpochs, batchSize)
    
    numberOfFeatures = myDataManipulations.numberOfFeatures
    nNeurons = [numberOfFeatures, 32, 32, 32]

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
    ops = tf.get_default_graph().get_operations()
    for op in ops:
        print(op.name)    
    ###############################################
    accuracyTable = np.array([])
    lossTable = np.array([])

    for iFold in range(0, 1):
        sess.run(init)
        aAccuracy = runCVFold(sess, iFold, myDataManipulations, myTrainWriter, myValidationWriter)
        accuracyTable = np.append(accuracyTable, aAccuracy)

    print("Mean accuracy: %0.2f 95CL: (%0.2f - %0.2f)" % (accuracyTable.mean(),
                                                             accuracyTable.mean()-2*accuracyTable.std(),
                                                             accuracyTable.mean()+2*accuracyTable.std()))
    ###############################################

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

  parser.add_argument('--dropout', type=float, default=1.0,
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
