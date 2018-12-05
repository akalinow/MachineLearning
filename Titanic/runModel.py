from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import numpy as np
from dataManipulations import *
from plotUtilities import *

FLAGS = None

##############################################################################
##############################################################################
##############################################################################
def runModel(myDataManipulations):

    sess = tf.Session()
    
    # Add ops to save and restore all the variables.
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], FLAGS.model_dir)

    '''
    ops = tf.get_default_graph().get_operations()
    for op in ops:
        print(op.name)
    '''

    x = tf.get_default_graph().get_operation_by_name("input/x-input").outputs[0]
    y = tf.get_default_graph().get_operation_by_name("model/performance/Sigmoid").outputs[0]
    yTrue = tf.get_default_graph().get_operation_by_name("input/y-input").outputs[0]
    dropout_prob = tf.get_default_graph().get_operation_by_name("model/dropout_prob").outputs[0]
    trainingMode = tf.get_default_graph().get_operation_by_name("model/trainingMode").outputs[0]

    labels = myDataManipulations.labels
    features = myDataManipulations.features
    passengerId = myDataManipulations.passengerId
    passengerId = np.reshape(passengerId,(-1,1))

    result = sess.run([x, y], feed_dict={x: features, yTrue: labels, dropout_prob: 0.0, trainingMode: False})
    features = result[0]
    modelResult = result[1]

    isSurvived = modelResult>0.5
    

    print("passengerId.shape: {}, isSurvived.shape: {}".format(passengerId.shape, isSurvived.shape))
    isSurvived = np.concatenate([passengerId, isSurvived], axis=1)
    
    myHeader = "PassengerId,Survived"
    np.savetxt('modelResult.csv', isSurvived, header = myHeader, delimiter=',', fmt='%d', comments='')
##############################################################################
##############################################################################
##############################################################################
def test():

    fileName = FLAGS.test_data_file
    nFolds = 2
    nEpochs = 1
    batchSize = 900

    myDataManipulations = dataManipulations(fileName, nFolds, nEpochs, batchSize)

    runModel(myDataManipulations)

##############################################################################
##############################################################################
##############################################################################
def main(_):

  test()
##############################################################################
##############################################################################
##############################################################################
if __name__ == '__main__':
  parser = argparse.ArgumentParser()


  parser.add_argument('--test_data_file', type=str,
      default=os.path.join(os.getenv('PWD', './'),
                           'data/test/test.csv'),
      help='Directory for storing training data')

  parser.add_argument('--model_dir', type=str,
      default=os.path.join(os.getenv('PWD', './'),
                           'model/1'),
      help='Directory for storing model state')


  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
