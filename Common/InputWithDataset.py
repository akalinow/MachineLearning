import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn import preprocessing
from collections import OrderedDict

##############################################################################
##############################################################################
##############################################################################
class InputWithDataset:

    def getNumpyMatricesFromRawData(self):

       print("Define",__function__,"in daughter class.")

##############################################################################
##############################################################################       
    def makeCVFoldGenerator(self):

        foldSplitter = KFold(n_splits=self.nFolds)
        self.foldsIndexGenerator = foldSplitter.split(self.labels, self.features)
        self.indexList = list(enumerate(self.foldsIndexGenerator))

##############################################################################
##############################################################################
    def makeDataset(self, trainingMode=True):

        aDataset = tf.data.Dataset.from_tensor_slices((self.labels_placeholder, self.features_placeholder))

        if trainingMode: 
            aDataset = aDataset.batch(self.batchSize)

        return aDataset        

##############################################################################
##############################################################################
    def getDataIterator(self, aDataset):

        print("aDataset.output_shapes:",aDataset.output_shapes)

        aIterator = tf.data.Iterator.from_structure(aDataset.output_types, aDataset.output_shapes)
        return aIterator

##############################################################################
##############################################################################
    def initializeDataIteratorForCVFold(self, sess, aFold, trainingMode=True):

        if(aFold>=len(self.indexList)):
            print("Fold too big: ",aFold," number of folds is ",self.nFolds)
            return None

        if not trainingMode:
            validationIndexes = self.indexList[aFold][1][0]
            indexes = validationIndexes
        else:
            trainIndexes = self.indexList[aFold][1][1]
            indexes = trainIndexes

        if self.batchSize>len(indexes):
            self.batchSize = len(indexes)
        self.numberOfBatches = np.ceil(len(indexes)/(float)(self.batchSize))
        self.numberOfBatches = (int)(self.numberOfBatches)

        foldFeatures = self.features[indexes]
        foldLabels = self.labels[indexes]

        aDataset = self.makeDataset(trainingMode)
        init_op = self.dataIterator.make_initializer(aDataset)
        feed_dict={self.features_placeholder: foldFeatures, self.labels_placeholder: foldLabels}
        sess.run(init_op, feed_dict=feed_dict)

##############################################################################
##############################################################################    
    def __init__(self, fileName, nFolds, nEpochs, batchSize, nLabelBins):

        self.fileName = fileName
        self.batchSize = batchSize
        self.nFolds = nFolds
        self.nEpochs = nEpochs
        self.nLabelBins = nLabelBins
        self.numberOfFeatures = 1

        self.getNumpyMatricesFromRawData()
        self.makeCVFoldGenerator()

        self.features_placeholder = tf.placeholder(tf.float32, name='x-input', shape=(None, self.numberOfFeatures))
        self.labels_placeholder = tf.placeholder(tf.float32, name='y-input', shape=(None, self.nLabelBins))

        aDataset = self.makeDataset()
        self.dataIterator = self.getDataIterator(aDataset)
    
##############################################################################
##############################################################################
def makeFeedDict(sess, dataIter):
    aBatch = sess.run(dataIter)
    x = aBatch[0]
    y = np.reshape(aBatch[1],(-1,1))
    return x, y

##############################################################################
##############################################################################
##############################################################################
