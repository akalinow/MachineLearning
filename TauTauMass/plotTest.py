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
def testTheModel(sess, myDataManipulations):

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
            break
        except tf.errors.OutOfRangeError:
            break

    print(features[0:2])
    print(np.concatenate(features[0:2]))
    return

    labels = np.concatenate(labels)
    features = np.concatenate(features[0:-1],axis=0)
    modelResult = np.concatenate(modelResult)
    print(labels.shape)
    print(features.shape)
    print(modelResult.shape)
    return

    print("len(result)",len(result))
    print(result[0][0][0])
    #print(result[3][0])
    
    print("labels.shape:",labels.shape)
    return

    labels = result[0][0]
    modelInput = result[0][1]

    print("len(labels):",len(labels))
    return
    
    modelResult = result[1]

       
    model_fastMTT = modelInput[:,1]
    model_fastMTT = np.reshape(model_fastMTT,(-1,1))


    mLow_H125 = 110
    mHigh_H125 = 130

    mLow_Z90 = 80
    mHigh_Z90 = 100

    index = (labels>mLow_H125)*(labels<mHigh_H125)
    modelResult_H125 = modelResult[index]
    labels_H125 = labels[index]

    index = (labels>mLow_Z90)*(labels<mHigh_Z90)
    modelResult_Z90 = modelResult[index]
    labels_Z90 = labels[index]

    scores = np.concatenate((modelResult_H125, modelResult_Z90))
    labels_S = np.ones(len(modelResult_H125))
    labels_B = np.zeros(len(modelResult_Z90))
    labels_S_B = np.concatenate((labels_S, labels_B))
    fpr_training, tpr_training, thresholds = metrics.roc_curve(labels_S_B, scores, pos_label=1)    

    ####    
    modelResult = modelResult_H125
    labels = labels_H125
    ####
    
    pull = (modelResult - labels)/labels
    print("Model: NN",
          "mean pull:", np.mean(pull),
          "pull RMS:", np.std(pull, ddof=1))

    plotDiscriminant(modelResult, labels, "Training", doBlock=False)

    model_fastMTT = myDataManipulations.fastMTT
    model_fastMTT = np.reshape(model_fastMTT, (-1,1))
    labels = myDataManipulations.labels

    index = (labels>mLow_H125)*(labels<mHigh_H125)

    print("modelResult.shape:",modelResult.shape)
    print("model_fastMTT.shape:",model_fastMTT.shape)
    print("index.shape:",index.shape)
    
    modelResult_H125 = model_fastMTT[index]
    labels_H125 = labels[index]

    index = (labels>mLow_Z90)*(labels<mHigh_Z90)
    modelResult_Z90 = model_fastMTT[index]
    labels_Z90 = labels[index]

    scores = np.concatenate((modelResult_H125, modelResult_Z90))
    labels_S = np.ones(len(modelResult_H125))
    labels_B = np.zeros(len(modelResult_Z90))
    labels_S_B = np.concatenate((labels_S, labels_B))
    fpr_fastMTT, tpr_fastMTT, thresholds = metrics.roc_curve(labels_S_B, scores, pos_label=1)    

    ####
    model_fastMTT = modelResult_H125
    labels = labels_H125
    ####
    
    pull = ( model_fastMTT - labels)/labels
    print("Model: fastMTT",
          "mean pull:", np.mean(pull),
          "pull RMS:", np.std(pull, ddof=1))

    plotDiscriminant(model_fastMTT, labels, "fastMTT", doBlock=False)

    fig = plt.figure(3)
    ax = fig.add_subplot(1, 1, 1)    
    #ax.plot([0, 1], [0, 1], 'k--')
    ax.plot(tpr_training, fpr_training, label='Training')
    ax.plot(tpr_fastMTT, fpr_fastMTT, label='fastMTT')
    ax.set_xlim(0.4,0.6)
    ax.set_ylim(0.0,0.2)
    plt.xlabel('True positive rate')
    plt.ylabel('False positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()

    numberOfEvents = len(modelResult)
    x = modelResult
    x = np.reshape(x, (-1))
    y = labels[0:numberOfEvents]    
    ratio = np.divide(x, y)
    plotVariable(labels[0:numberOfEvents], ratio, plotTitle = "Ratio", doBlock = True)
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
    batchSize = 200000
    fileName = FLAGS.test_data_file
    nLabelBins = 1

    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], FLAGS.model_dir)

    myDataManipulations = dataManipulations(fileName, nFolds, batchSize, nLabelBins,  smearMET=False)
    initializeIterator(sess, myDataManipulations)

    if FLAGS.debug>0:
         listOperations()

    testTheModel(sess, myDataManipulations)
    
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

  parser.add_argument('--debug', type=int, default=0,
                       help=textwrap.dedent('''\
                      Runs debug methods: 
                      0 - disabled 
                      1 - list graph operations'''))

  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
##############################################################################
##############################################################################
