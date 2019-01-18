import tensorflow as tf
import numpy as np
import os
import argparse, textwrap

from dataManipulations import *
from model import *

##############################################################################
##############################################################################
class TrainFramework:

    ##############################################################################
    ##############################################################################
    def initializeParser(self):

        self.myParser = argparse.ArgumentParser()

        self.myParser.add_argument('--max_epoch', type=int, default=50,
                                   help='Number of epochs')

        self.myParser.add_argument('--batchSize', type=int, default=1024,
                                   help='Number of examples taken as a single training batch.')

        self.myParser.add_argument('--nLabelBins', type=int, default=1,
                                   help='Dimension of the input label')

        self.myParser.add_argument('--nFolds', type=int, default=2,
                                   help='''Number of cross validation folds. Default value of 2
                                           means splitting the whole dataset into two equal parts 
                                           for training and validation''')

        self.myParser.add_argument('--learning_rate', type=float, default=0.001,
                                   help='Initial learning rate')
        
        self.myParser.add_argument('--lambda_lagrange', type=float, default=0.1,
                                   help='Largange multipler for L2 loss')

        self.myParser.add_argument('--dropout', type=float, default=0.2,
                                   help='Drop probability for training dropout.')

        self.myParser.add_argument('--train_data_file', type=str,
                                   default=os.path.join(os.getenv('PWD', './'),
                                                        'data/htt_features_train.pkl'),
                                   help='Directory for storing training data')

        self.myParser.add_argument('--model_dir', type=str,
                                   default=os.path.join(os.getenv('PWD', './'),
                                                        'model'),
                                   help='Directory for storing model state')
        
        self.myParser.add_argument('--log_dir', type=str,
                                   default=os.path.join(os.getenv('PWD', './'),
                                                        'logs'),
                                   help='Summaries log directory')
        
        self.myParser.add_argument('--debug', type=int, default=0,
                                   help=textwrap.dedent('''\
                                   Runs debug methods: 
                                   0 - disabled 
                                   1 - list graph operations 
                                   2 - run debug method defined by the user'''))        
    ##############################################################################
    ##############################################################################
    def parse_known_args(self):
    
        self.FLAGS, self.unparsed = self.myParser.parse_known_args()

    ##############################################################################
    ##############################################################################
    def cleanDirectories(self):
            
        if tf.gfile.Exists(self.FLAGS.log_dir):
            tf.gfile.DeleteRecursively(self.FLAGS.log_dir)
            tf.gfile.MakeDirs(self.FLAGS.log_dir)

        if tf.gfile.Exists(self.FLAGS.model_dir):
            tf.gfile.DeleteRecursively(self.FLAGS.model_dir)
            
    ##############################################################################
    ##############################################################################
    def saveTheModel(self):

        x = tf.get_default_graph().get_operation_by_name("data/x-input").outputs[0]
        y = tf.get_default_graph().get_operation_by_name("model/output/Identity").outputs[0]
        yTrue = tf.get_default_graph().get_operation_by_name("data/y-input").outputs[0]
        
        tf.saved_model.simple_save(self.mySess, self.FLAGS.model_dir,
                                   inputs={"x": x, "yTrue": yTrue},
                                   outputs={"y": y})
        print("Model saved in file: %s" % self.FLAGS.model_dir)
    ##############################################################################
    ##############################################################################
    def runOverEpoch(self, iEpoch, aWriter, isTraining):

        train_step = tf.get_default_graph().get_operation_by_name("model/train/Adam")
        dropout_prob = tf.get_default_graph().get_operation_by_name("model/dropout_prob").outputs[0]
        trainingMode = tf.get_default_graph().get_operation_by_name("model/trainingMode").outputs[0]
        mergedSummary = tf.get_default_graph().get_operation_by_name("model/monitor/Merge/MergeSummary").outputs[0]

        ops = tf.get_collection("MY_UPDATE_OPS")
        ops.append(mergedSummary)
        if isTraining:
            ops.insert(0, train_step)
            
        result = []
        while True:
            try:
                result = self.mySess.run(ops, feed_dict={dropout_prob: self.FLAGS.dropout, trainingMode: isTraining})
            except tf.errors.OutOfRangeError:
                break
        
        if iEpoch%self.printoutStep==0:
            print("Epoch:",iEpoch)
            print("{runType}".format(runType="Training" if isTraining else "Validation"))
            
            trainSummary = result[-1]
            aWriter.add_summary(trainSummary, iEpoch)  
            
            ops = tf.get_collection("MY_RUNNING_VALS")
            result = self.mySess.run(ops)

            for index, aOp in enumerate(ops):
                print(aOp.name, result[index])
    ##############################################################################
    ##############################################################################
    def runAll(self):

        init_global = tf.global_variables_initializer()
        init_local = tf.local_variables_initializer()
        self.mySess.run([init_global, init_local])

        aTrainWriter = tf.summary.FileWriter(self.FLAGS.log_dir + '/train', self.mySess.graph)
        aValidationWriter = tf.summary.FileWriter(self.FLAGS.log_dir + '/validation', self.mySess.graph)

        for iEpoch in range(0,self.FLAGS.max_epoch):
            self.mySess.run([init_local])
            self.myDataManipulations.initializeDataIteratorForCVFold(self.mySess, aFold=0, trainingMode=True)
            self.runOverEpoch(iEpoch, aTrainWriter, isTraining=True)

            if iEpoch%self.printoutStep==0:
                self.myDataManipulations.initializeDataIteratorForCVFold(self.mySess, aFold=0, trainingMode=False)
                self.runOverEpoch(iEpoch, aValidationWriter, isTraining=False)

        aTrainWriter.close()
        aValidationWriter.close()

        self.saveTheModel()
    ##############################################################################
    ##############################################################################
    def initializeDataStream(self, aDataManipulations):

        self.myDataManipulations = aDataManipulations
        self.aDataIterator  = self.myDataManipulations.dataIterator.get_next()
        self.numberOfFeatures = self.myDataManipulations.numberOfFeatures
        
    ##############################################################################
    ##############################################################################
    def __init__(self):

        self.initializeParser()
        self.parse_known_args()

        self.printoutStep = int(self.FLAGS.max_epoch/100) + 1

        self.cleanDirectories()
                      
        self.mySess = tf.Session()
        print("Available computing devices:")
        devices = self.mySess.list_devices()
        for d in devices:
            print(d.name)
    ##############################################################################
    ##############################################################################
                
##############################################################################
##############################################################################            



