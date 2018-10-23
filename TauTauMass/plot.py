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
def makePlots(sess, myDataManipulations):
    #Fetch operations
    x = tf.get_default_graph().get_operation_by_name("input/x-input").outputs[0]
    y = tf.get_default_graph().get_operation_by_name("model/output/Identity").outputs[0]    
    yTrue = tf.get_default_graph().get_operation_by_name("input/y-input").outputs[0]
    keep_prob = tf.get_default_graph().get_operation_by_name("model/dropout/Placeholder").outputs[0]

    iFold = 0 
    aTrainIterator, aValidationIterator = myDataManipulations.getCVFold(sess, iFold)
    numberOfBatches = myDataManipulations.numberOfBatches
   
    xs, ys = makeFeedDict(sess, aTrainIterator)
   
    result = sess.run([x, y, yTrue], feed_dict={x: xs, yTrue: ys, keep_prob: 1.0})
    modelInput = result[0]
    modelResult = result[1]
    model_fastMTT = modelInput[:,1]
    model_fastMTT = np.reshape(model_fastMTT,(-1,1))
    labels = result[2]

    print(len(labels))

    pull = (modelResult - labels)/labels
    print("Model: NN",
          "mean pull:", np.mean(pull),
          "pull RMS:", np.std(pull, ddof=1))

    plotDiscriminant(modelResult, labels, "Training", doBlock=False)

    model_fastMTT = myDataManipulations.fastMTT
    labels = myDataManipulations.labels

    print(len(labels))

    pull = ( model_fastMTT - labels)/labels
    print("Model: fastMTT",
          "mean pull:", np.mean(pull),
          "pull RMS:", np.std(pull, ddof=1))

    plotDiscriminant(model_fastMTT, labels, "fastMTT", doBlock=True)

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

        myDataManipulations = dataManipulations(fileName, nFolds, nEpochs, batchSize, smearMET=False)

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
