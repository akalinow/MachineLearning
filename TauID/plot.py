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
from sklearn.metrics import roc_curve, roc_auc_score

FLAGS = None

#deviceName = '/cpu:0'
#deviceName = '/:GPU:0'
deviceName = None
##############################################################################
##############################################################################
def makePlots(sess, myDataManipulations):
    #Fetch operations
    x = tf.get_default_graph().get_operation_by_name("input/x-input").outputs[0]
    y = tf.get_default_graph().get_operation_by_name("model/performance/Sigmoid").outputs[0]
    yTrue = tf.get_default_graph().get_operation_by_name("input/y-input").outputs[0]
    keep_prob = tf.get_default_graph().get_operation_by_name("model/dropout/Placeholder").outputs[0]
    trainingMode = tf.get_default_graph().get_operation_by_name("model/Placeholder").outputs[0]

    features = myDataManipulations.features
    labels = myDataManipulations.labels
    result = sess.run([y, yTrue], feed_dict={x: features, yTrue: labels, keep_prob: 1.0, trainingMode: False})
    modelResult = result[0]
    modelResult = np.reshape(modelResult,(1,-1))[0]
    modelResult = 1 - modelResult
    
    modelResults = {"training": modelResult,
                    "DPFv1":features[:,-1],
                    "deepTau":features[:,0],
                    "MVA2017v2":features[:,-4]}

    indexesS = labels==1
    signalResponse = modelResult[indexesS]

    indexesB = labels==0
    backgroundResponse = modelResult[indexesB]

    plt.figure(1)
    plt.hist(signalResponse, label="true tau")
    plt.hist(backgroundResponse, label="fake tau")
    plt.legend(loc=2)
    plt.show(block=False)
    
    plt.figure(2)
    for model, aResult in modelResults.items():
        print('ROC AUC score for {} model: '.format(model), 1.0 - roc_auc_score(labels, aResult))
        fpr, tpr, thr = roc_curve(labels, aResult, pos_label=1)
        plt.semilogy(tpr, fpr, label=model)
        #plt.xlim((0.2, 1.0))
        plt.ylabel('False positive rate')
        plt.xlabel('True positive rate')
    
    plt.legend(loc=2)
    plt.show()

##############################################################################
##############################################################################
def plot():

    with tf.Session(graph=tf.Graph()) as sess:

        print("Available devices:")
        devices = sess.list_devices()
        for d in devices:
            print(d.name)

        nEpochs = 1
        batchSize = 100000
        nFolds = 2
        fileName = FLAGS.test_data_file

        myDataManipulations = dataManipulations(fileName, nFolds, nEpochs, batchSize)

        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], FLAGS.model_dir)
        
        makePlots(sess, myDataManipulations)            
##############################################################################
##############################################################################
##############################################################################
def main(_):

  plot()
##############################################################################
##############################################################################
##############################################################################
if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--test_data_file', type=str,
      default=os.path.join(os.getenv('PWD', './'),
                           'data/htt_features.pkl'),
      help='Directory for storing training data')

  parser.add_argument('--model_dir', type=str,
      default=os.path.join(os.getenv('PWD', './'),
                           'model'),
      help='Directory for storing model state')

  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
##############################################################################
##############################################################################
##############################################################################
