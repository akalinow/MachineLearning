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

    def __init__(self, fileName, nFolds, batchSize, nLabelBins, smearMET=True):        
        
        self.smearMET = smearMET
        
        InputWithDataset.__init__(self, fileName, nFolds, batchSize, nLabelBins)       

##############################################################################
    def getNumpyMatricesFromRawData(self):

        legs, jets, global_params, properties = pd.read_pickle(self.fileName)
        properties = OrderedDict(sorted(properties.items(), key=lambda t: t[0]))

        print("no of legs: ", len(legs))
        print("no of jets: ", len(jets))
        print("global params: ", global_params.keys())
        print("object properties:",properties.keys())

        genMass = np.array(global_params["genMass"])
        fastMTT = np.array(global_params["fastMTTMass"])
        visMass = np.array(global_params["visMass"])
        caMass = np.array(global_params["caMass"])
        leg1P4 = np.array(legs[0])
        leg2P4 = np.array(legs[1])
        leg1GenP4 = np.array(legs[2])
        leg2GenP4 = np.array(legs[3])        
        leg2Properties = np.array(properties["leg_2_decayMode"])
        jet1P4 = np.array(jets[1])
        jet2P4 = np.array(jets[2])        
        met = np.array(jets[0][0:3])

        genMass = np.reshape(genMass, (-1,1))
        visMass = np.reshape(visMass, (-1,1))
        caMass = np.reshape(caMass, (-1,1))
        fastMTT = np.reshape(fastMTT, (-1,1))
        leg2Properties = np.reshape(leg2Properties, (-1,1))
        leg1P4 = np.transpose(leg1P4)
        leg2P4 = np.transpose(leg2P4)
        leg1GenP4 = np.transpose(leg1GenP4)
        leg2GenP4 = np.transpose(leg2GenP4)        
        jet1P4 = np.transpose(jet1P4)
        jet2P4 = np.transpose(jet2P4)
        met = np.transpose(met)

        leg1Pt = np.sqrt(leg1P4[:,1]**2 + leg1P4[:,2]**2)
        leg2Pt = np.sqrt(leg2P4[:,1]**2 + leg2P4[:,2]**2)
        leg1Pt = np.reshape(leg1Pt, (-1,1))
        leg2Pt = np.reshape(leg2Pt, (-1,1)) 
        metMag = met[:,0]
        #smear met
        if self.smearMET:            
            sigma = 2.6036 -0.0769104*metMag + 0.00102558*metMag*metMag-4.96276e-06*metMag*metMag*metMag
            print("Smearing metX and metY with sigma =",sigma)
            metX = met[:,1]*(1.0 + sigma*np.random.randn(met.shape[0]))
            metY = met[:,2]*(1.0 + sigma*np.random.randn(met.shape[0]))
            metMag = np.sqrt(metX**2 + metY**2)
            met = np.stack((metMag, metX, metY), axis=1)

        metMag = np.reshape(metMag, (-1,1))    

        #leg2GenEnergy = leg2GenP4[:,0]
        #leg2GenEnergy = np.reshape(leg2GenEnergy, (-1,1))
        features = np.hstack((genMass, fastMTT, leg1P4, leg2P4, met))
               
        #Select events with MET>10
        index = met[:,0]>10 
        features = features[index]

        index = features[:,0]<250 
        features = features[index]

        index = features[:,0]>50 
        features = features[index]

        index = features[:,1]<250
        #features = features[index]
        
        index = features[:,1]>2 
        #features = features[index]

        np.random.shuffle(features)

        #Quantize the output variable into self.nLabelBins
        #or leave it as a floating point number        
        labels = features[:,0]
        labels = np.reshape(labels, (-1,1))
        '''
        if self.nLabelBins>1:
            est = preprocessing.KBinsDiscretizer(n_bins=self.nLabelBins, encode='ordinal', strategy='uniform')
            tmp = np.reshape(features[:,0], (-1, 1))
            est.fit(tmp)
            labels = est.transform(tmp) + 1#Avoid bin number 0
        else:                
            labels = features[:,0]
        '''

        #Apply all transformations to fastMTT column, as we want to plot it,
        #but remove the fastMTT column from model features
        fastMTT = features[:,1]
        features = features[:,2:]

        print("Input data shape:",features.shape)
        print("Label bins:",self.nLabelBins, "labels shape:",labels.shape)

        self.numberOfFeatures = features.shape[1]
             
        assert features.shape[0] == labels.shape[0]

        self.fastMTT = fastMTT
        self.features = features
        self.labels = labels

##############################################################################
##############################################################################
##############################################################################

