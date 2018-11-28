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
from sklearn import metrics

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
    '''    
    prob = tf.nn.softmax(logits=y)
    onehot_labels = tf.one_hot(tf.to_int32(yTrue), depth=10, axis=-1)
    result = sess.run([x, y, yTrue, prob, onehot_labels], feed_dict={x: xs, yTrue: ys, keep_prob: 1.0})

    p = result[3]
    onehot = result[4]
    print("Target:",onehot[0:3])
    print("Prediction:",p[0:3])
    #plt.plot(p[0],"r",p[1],"g",p[2],"b")
    plt.plot(onehot[0],"r")
    plt.show()
    return
    '''
    
    modelInput = result[0]
    modelResult = result[1]
    model_fastMTT = modelInput[:,1]
    model_fastMTT = np.reshape(model_fastMTT,(-1,1))
    labels = result[2]

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
    labels = myDataManipulations.labels

    index = (labels>mLow_H125)*(labels<mHigh_H125)
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
    model_fastMTT = modelResult_Z90
    labels = labels_Z90
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
      help='File containing test examples')


  parser.add_argument('--model_dir', type=str,
      default=os.path.join(os.getenv('PWD', './'),
                           'model'),
      help='Directory for storing model state')

  FLAGS, unparsed = parser.parse_known_args()

  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
##############################################################################
##############################################################################
##############################################################################
