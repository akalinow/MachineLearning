import os
import sys

sys.path.append('../Common/')

import tensorflow as tf
import numpy as np
from dataManipulations import *
from model import *
from TrainFramework import *

##############################################################################
##############################################################################
def main():
    
    myFramework = TrainFramework()

    ##Add custom command line flags    
    myFramework.myParser.add_argument('--smearMET', type=bool, default=False,
                                   help='Bool controling if the generator level MET should be smeared.')

    ##Parse input parameters
    myFramework.parse_known_args()
    FLAGS = myFramework.FLAGS

    ##Create the input stream handler
    myDataManipulations = dataManipulations(FLAGS.train_data_file, FLAGS.nFolds,
                                            FLAGS.batchSize, FLAGS.nLabelBins,
                                            FLAGS.smearMET)
    
    myFramework.initializeDataStream(myDataManipulations)

    ##Create the model
    numberOfFeatures = myFramework.numberOfFeatures

    ##List defining number of neurons in each layer including the input layer,
    ##BUT excluding the output layer
    #nNeurons = [numberOfFeatures, 128, 128, 128, 128, 128]
    nNeurons = [numberOfFeatures, 23, 32, 32, 32, 32, 32, 32, 32, 32]
    nOutputNeurons = FLAGS.nLabelBins
    
    myModel = Model(myFramework.aDataIterator,
                    nNeurons, nOutputNeurons,
                    FLAGS.learning_rate, FLAGS.lambda_lagrange)

    ##Run training and validation
    myFramework.runAll()

    
##############################################################################
##############################################################################
if __name__ == '__main__':

    main()

##############################################################################
##############################################################################    
