import argparse
import os

import numpy as np
import pandas as pd


FLAGS = None

##############################################################################
##############################################################################
def splitData():

     legs, jets, global_params, properties = pd.read_pickle(FLAGS.input)

     print("no of legs: ", len(legs))
     print("no of jets: ", len(jets))
     print("global params: ", global_params.keys())
     print("object properties:",properties.keys())

     features = np.array(list(properties.values()))
     features = np.transpose(features)
     propertiesNames = list(properties.keys())

     print(propertiesNames.index("leg_2_deepTau2017v1tauVSall"))
     



     columnMask = np.full(features.shape[1], False)
     oldMVA_discriminators = ["leg_2_byIsolationMVArun2v1DBoldDMwLTraw",
                              "leg_2_byIsolationMVArun2v1DBoldDMwLTraw2017v2",
                              "leg_2_DPFTau_2016_v1tauVSall",                              
                              "leg_2_deepTau2017v1tauVSall"]
     for discName in oldMVA_discriminators:          
          index = propertiesNames.index(discName)
          print("Enabling feature:",discName)
          columnMask[index] = True 

     features = features[:,columnMask]

     print(features[2:])


##############################################################################
##############################################################################
if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--input', type=str,
                      default=os.path.join(os.getenv('PWD', './'),
                                           'data/htt_features_test.pkl'),
                      help='Directory for storing training data')

  FLAGS, unparsed = parser.parse_known_args()

  splitData()

##############################################################################
##############################################################################
   
