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

       print("Define",__name__,"in daughter class.")

##############################################################################
##############################################################################       
    def makeCVFoldGenerator(self):

        foldSplitter = KFold(n_splits=self.nFolds)
        self.foldsIndexGenerator = foldSplitter.split(self.labels, self.features)
        self.indexList = list(enumerate(self.foldsIndexGenerator))

##############################################################################
##############################################################################
    def makeDataset(self):

        aDataset = tf.data.Dataset.from_tensor_slices((self.labels_placeholder, self.features_placeholder))
        aDataset = aDataset.batch(self.batchSize)
        self.aDataset = aDataset.prefetch(self.batchSize)

##############################################################################
##############################################################################
    def makeDataIterator(self):

        self.dataIterator = tf.data.Iterator.from_structure(self.aDataset.output_types, self.aDataset.output_shapes)
        self.iterator_init_op = self.dataIterator.make_initializer(self.aDataset)
        
##############################################################################
##############################################################################
    def initializeDataIteratorForCVFold(self, sess, aFold, trainingMode=True):

        if(aFold>=len(self.indexList)):
            print("Fold too big: ",aFold," number of folds is ",self.nFolds)
            return None

        if trainingMode:
            validationIndexes = self.indexList[aFold][1][0]
            indexes = validationIndexes
        else:
            trainIndexes = self.indexList[aFold][1][1]
            indexes = trainIndexes

        if self.batchSize>len(indexes):
            self.batchSize = len(indexes)

        foldFeatures = self.features[indexes]
        foldLabels = self.labels[indexes]

        feed_dict={self.labels_placeholder: foldLabels, self.features_placeholder: foldFeatures}
        sess.run(self.iterator_init_op, feed_dict=feed_dict)

##############################################################################
##############################################################################    
    def __init__(self, fileName, nFolds, batchSize, nLabelBins):

        self.fileName = fileName
        self.batchSize = batchSize
        self.nFolds = nFolds
        self.nLabelBins = nLabelBins
        self.numberOfFeatures = 1

        self.getNumpyMatricesFromRawData()

        with tf.name_scope('data'):
            self.features_placeholder = tf.placeholder(tf.float32, name='x-input', shape=(None, self.numberOfFeatures))
            self.labels_placeholder = tf.placeholder(tf.float32, name='y-input', shape=(None, self.nLabelBins))

            self.makeDataset()
            self.makeDataIterator()
            self.makeCVFoldGenerator()
        
        aFold = 0
        print("Number of examples in trainig fold", aFold, len(self.indexList[aFold][1][0]))
        print("Number of examples in validation fold", aFold, len(self.indexList[aFold][1][1]))
    
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
