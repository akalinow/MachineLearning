import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn import preprocessing

##############################################################################
##############################################################################
##############################################################################
class dataManipulations:

    def getNumpyMatricesFromRawData(self):

        data = pd.read_csv(self.fileName, sep=',',header=0)
        data.replace(to_replace=dict(female=0, male=1), inplace=True)
        data.replace(to_replace=dict(C=1,Q=2,S=3), inplace=True)
        data.fillna(value=0,inplace=True)

        features = data.values
        #np.random.shuffle(features)    

        ##Add dummy survived column for the test data
        print("shape: ",features.shape)
        if features.shape[1]==11:
            features = np.insert(features, 1, values = 99, axis=1)

        #PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
        self.passengerId =  features[:,0]
        self.labels = features[:,1]
        columnMask = np.full(features.shape[1], True)
        columnMask[0] = False #mask PassengerId
        columnMask[1] = False #mask Survived
        columnMask[3] = False #mask Name
        columnMask[8] = False #mask Ticket
        columnMask[10] = False #mask Cabin
        features = features[:,columnMask]
                       
        self.numberOfFeatures = features.shape[1]             
        self.features_placeholder = tf.placeholder(tf.float32)
        self.labels_placeholder = tf.placeholder(tf.float32)
        self.features = features

        print("Input data shape:",features.shape)
        print("Data line: ",features[0,:])


    def makeCVFoldGenerator(self):

        foldSplitter = KFold(n_splits=self.nFolds)
        self.foldsIndexGenerator = foldSplitter.split(self.labels, self.features)
        self.indexList = list(enumerate(self.foldsIndexGenerator))

    def makeDatasets(self):

        aDataset = tf.data.Dataset.from_tensor_slices((self.features_placeholder, self.labels_placeholder))
        self.trainDataset = aDataset.batch(self.batchSize)
        self.trainDataset = self.trainDataset.repeat(self.nEpochs)

        aDataset = tf.data.Dataset.from_tensor_slices((self.features_placeholder, self.labels_placeholder))
        self.validationDataset = aDataset.batch(len(self.labels))


    def getDataIteratorAndInitializerOp(self, aDataset):

        aIterator = tf.data.Iterator.from_structure(aDataset.output_types, aDataset.output_shapes)
        init_op = aIterator.make_initializer(aDataset)
        return aIterator, init_op

    def getCVFold(self, sess, aFold):

        if(aFold>=len(self.indexList)):
            print("Fold too big: ",aFold," number of folds is ",self.nFolds)
            return None

        trainIndexes = self.indexList[aFold][1][0]
        validationIndexes = self.indexList[aFold][1][1]

        if self.batchSize>len(trainIndexes):
            self.batchSize = len(trainIndexes)
        self.numberOfBatches = np.ceil(len(trainIndexes)/(float)(self.batchSize))
        self.numberOfBatches = (int)(self.numberOfBatches)

        print("Numer of training examples:",len(trainIndexes))
        print("Number of validation examples:",len(validationIndexes))
        print("Batch size:",self.batchSize)
        print("Batches/epoch:",self.numberOfBatches)

        foldFeatures = self.features[trainIndexes]
        foldLabels = self.labels[trainIndexes]
        feed_dict={self.features_placeholder: foldFeatures, self.labels_placeholder: foldLabels}
        sess.run(self.trainIt_InitOp, feed_dict=feed_dict)

        foldFeatures = self.features[validationIndexes]
        foldLabels = self.labels[validationIndexes]
        feed_dict={self.features_placeholder: foldFeatures, self.labels_placeholder: foldLabels}
        sess.run(self.validationIt_InitOp, feed_dict=feed_dict)

        return self.trainIterator.get_next(), self.validationIterator.get_next()

    def __init__(self, fileName, nFolds, nEpochs, batchSize):
        self.fileName = fileName
        self.batchSize = batchSize
        self.nFolds = nFolds
        self.nEpochs = nEpochs

        self.getNumpyMatricesFromRawData()
        self.makeCVFoldGenerator()
        self.makeDatasets()

        self.trainIterator, self.trainIt_InitOp = self.getDataIteratorAndInitializerOp(self.trainDataset)
        self.validationIterator, self.validationIt_InitOp = self.getDataIteratorAndInitializerOp(self.validationDataset)

##############################################################################
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
