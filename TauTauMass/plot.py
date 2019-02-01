from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse, textwrap
import os
import sys

sys.path.append('../Common/')

import tensorflow as tf
import numpy as np
from sklearn import metrics

from dataManipulations import *
from plotUtilities import *
from modelUtilities import listOperations

FLAGS = None

#deviceName = '/cpu:0'
#deviceName = '/:GPU:0'
deviceName = None
 
##############################################################################
##############################################################################
def getModelResult(sess, myDataManipulations):

    initializeIterator(sess, myDataManipulations)

    dropout_prob = tf.get_default_graph().get_operation_by_name("model/dropout_prob").outputs[0]
    trainingModeFlag = tf.get_default_graph().get_operation_by_name("model/trainingMode").outputs[0]

    dataIter = tf.get_default_graph().get_operation_by_name("IteratorGetNext").outputs
    response = tf.get_default_graph().get_operation_by_name("model/performance/response").outputs[0]
     
    ops = [dataIter, response]
    labels = []
    features = []
    modelResult = []
    while True:
        try:
            [[a, b], c] = sess.run(ops, feed_dict={dropout_prob: 0.0, trainingModeFlag: False})
            labels.extend(a)
            features.extend(b)
            modelResult.extend(c)
        except tf.errors.OutOfRangeError:
            break

    labels = np.concatenate(labels)
    features = np.stack(features)
    modelResult = np.concatenate(modelResult)

    return labels, features, modelResult
    
##############################################################################
##############################################################################
def getMassRange(sess, myDataManipulations, lowMass, highMass, modelType):

    if modelType=="training":        
        labels, features, result = getModelResult(sess, myDataManipulations)
        stepSize = 250/FLAGS.nLabelBins
        result *= stepSize
        labels *= stepSize
    elif modelType=="fastMTT":
        labels = myDataManipulations.labelsRaw
        result = myDataManipulations.fastMTT
        result = np.reshape(result, (-1,1))
    elif modelType=="caMass":
        labels = myDataManipulations.labelsRaw
        result = myDataManipulations.caMass
        result = np.reshape(result, (-1,1))
    elif modelType=="visMass":
        labels = myDataManipulations.labelsRaw
        result = myDataManipulations.visMass
        result = np.reshape(result, (-1,1))

    index = (labels>=lowMass)*(labels<highMass)*(result>0)*(result<300)
    result_range = result[index]
    labels_range = labels[index]

    return labels_range, result_range
##############################################################################
##############################################################################
def testTheModel(sess, myDataManipulations, modelType):

    labelsZ90, resultZ90 = getMassRange(sess, myDataManipulations, 80, 100, modelType)
    labelsH125, resultH125 = getMassRange(sess, myDataManipulations, 130, 140, modelType)

    pullZ90 = (resultZ90 - labelsZ90)/labelsZ90
    pullH125 = (resultH125 - labelsH125)/labelsH125

    print("Model:",modelType)
    print("Mass range: Z90",
          "mean pull:", np.mean(pullZ90),
          "pull RMS:", np.std(pullZ90, ddof=1))
    print("Mass range: H125",
          "mean pull:", np.mean(pullH125),
          "pull RMS:", np.std(pullH125, ddof=1))

    plotDiscriminant(resultZ90, labelsZ90, modelType+" Z90", doBlock=False)
    plotDiscriminant(resultH125, labelsH125, modelType+" H125", doBlock=False)

    scores = np.concatenate((resultH125, resultZ90))
    labels_S = np.ones(len(resultH125))
    labels_B = np.zeros(len(resultZ90))
    labels_S_B = np.concatenate((labels_S, labels_B))
    fpr, tpr, thresholds = metrics.roc_curve(labels_S_B, scores, pos_label=1) 

    return fpr, tpr
##############################################################################
##############################################################################
def initializeIterator(sess, myDataManipulations):

    dropout_prob = tf.get_default_graph().get_operation_by_name("model/dropout_prob").outputs[0]
    trainingModeFlag = tf.get_default_graph().get_operation_by_name("model/trainingMode").outputs[0]

    features = myDataManipulations.features
    labels = myDataManipulations.labels
    
    x = tf.get_default_graph().get_operation_by_name("data/x-input").outputs[0]
    yTrue = tf.get_default_graph().get_operation_by_name("data/y-input").outputs[0]
    iterator_init_op = tf.get_default_graph().get_operation_by_name("data/make_initializer")
    sess.run([iterator_init_op], feed_dict={x: features, yTrue: labels, dropout_prob: 0.0, trainingModeFlag: False})

    myDataManipulations.dataIterator.get_next()
        
##############################################################################
##############################################################################
def test():

    sess = tf.Session()

    print("Available devices:")
    devices = sess.list_devices()
    for d in devices:
        print(d.name)

    nFolds = 10 
    batchSize = 100000
    fileName = FLAGS.test_data_file
    nLabelBins = FLAGS.nLabelBins

    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], FLAGS.model_dir)

    myDataManipulations = dataManipulations(fileName, nFolds, batchSize, nLabelBins,  smearMET=False)
    
    if FLAGS.debug>0:
         listOperations()
         

    fpr_training, tpr_training = testTheModel(sess, myDataManipulations, "training")
    fpr_fastMTT, tpr_fastMTT = testTheModel(sess, myDataManipulations, "fastMTT")
    #fpr_caMass, tpr_caMass = testTheModel(sess, myDataManipulations, "caMass")
    #fpr_visMass, tpr_visMass = testTheModel(sess, myDataManipulations, "visMass")

    fig = plt.figure(10)
    ax = fig.add_subplot(1, 1, 1, label="ROC")    
    #ax.plot([0, 1], [0, 1], 'k--')
    ax.plot(tpr_training, fpr_training, label='Training')
    ax.plot(tpr_fastMTT, fpr_fastMTT, label='fastMTT')
    #ax.plot(tpr_caMass, fpr_caMass, label='caMass')
    #ax.plot(tpr_visMass, fpr_visMass, label='visMass')
    ax.set_xlim(0.25,0.75)
    ax.set_ylim(0.0,0.005)
    plt.xlabel('True positive rate')
    plt.ylabel('False positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    
##############################################################################
##############################################################################
def main(_):

  test()
##############################################################################
##############################################################################
if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--test_data_file', type=str,
                      default=os.path.join(os.getenv('PWD', './'),
                                           'data/htt_features_train.pkl'),
                      help='Directory for storing training data')

  parser.add_argument('--model_dir', type=str,
                      default=os.path.join(os.getenv('PWD', './'),
                                           'model'),
                      help='Directory for storing model state')

  parser.add_argument('--nLabelBins', type=int, default=1,
                      help='Dimension of the input label')

  parser.add_argument('--debug', type=int, default=0,
                       help=textwrap.dedent('''\
                      Runs debug methods: 
                      0 - disabled 
                      1 - list graph operations'''))

  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
##############################################################################
##############################################################################
