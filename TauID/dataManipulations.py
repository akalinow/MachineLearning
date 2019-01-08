import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import KFold
from sklearn import preprocessing
from collections import OrderedDict

from InputWithDataset import *
##############################################################################
##############################################################################
##############################################################################
class dataManipulations(InputWithDataset):

    def __init__(self, fileName, nFolds, nEpochs, batchSize):        
        
        nLabelBins = 1
        
        InputWithDataset.__init__(self, fileName, nFolds, nEpochs, batchSize, nLabelBins)

##############################################################################
    def getNumpyMatricesFromRawData(self):

        legs, jets, global_params, properties = pd.read_pickle(self.fileName)
        properties = OrderedDict(sorted(properties.items(), key=lambda t: t[0]))

        print("no of legs: ", len(legs))
        print("no of jets: ", len(jets))
        print("global params: ", global_params.keys())
        print("object properties:",properties.keys())

        sampleType = np.array(global_params["sampleType"])
        sampleType = np.reshape(sampleType, (-1,1))
        features = np.array(list(properties.values()))
        features = np.transpose(features)
        featuresNames = list(properties.keys())

        #Redefine DPF output to be 1 for signal
        discName = "leg_2_DPFTau_2016_v1tauVSall"
        DPF_index = featuresNames.index(discName)
        features[:,DPF_index] *= -1
        features[:,DPF_index] +=  1
        indexes = features[:,DPF_index]>1
        features[indexes,DPF_index] = 0.0
        #Filter features to be usedfor training        
        columnMask = np.full(features.shape[1], True)
        oldMVA_discriminators = ["leg_2_byIsolationMVArun2v1DBnewDMwLTraw2017v2",
                                 "leg_2_DPFTau_2016_v1tauVSall",                              
                                 "leg_2_deepTau2017v1tauVSall",
                                 "leg_2_deepTau2017v1tauVSjet",
                                 ]
        for discName in oldMVA_discriminators:          
            index = featuresNames.index(discName)
            print("Enabling feature:",discName)
            columnMask[index] = True
                    
        features = features[:,columnMask]
        ########################################

        features = np.hstack((sampleType, features))
        np.random.shuffle(features)

        labels = features[:,0]
        features = features[:,1:]

        print("Input data shape:",features.shape)
        print("Number of positive examples:",(labels>0.5).sum())
        print("Number of negative examples:",(labels<0.5).sum())

        assert features.shape[0] == labels.shape[0]
              
        self.numberOfFeatures = features.shape[1]
        self.features_placeholder = tf.placeholder(tf.float32)
        self.labels_placeholder = tf.placeholder(tf.float32)
        self.features = features
        self.labels = labels

        tmp = np.array(featuresNames)
        tmp = tmp[columnMask]
        self.featuresNames = list(tmp)
                
##############################################################################
##############################################################################
##############################################################################
