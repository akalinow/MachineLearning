from modelUtilities import *

##############################################################################
##############################################################################
##############################################################################
class Model:

    def addFCLayers(self):

        for iLayer in range(1,self.nLayers):
            previousLayer = self.myLayers[iLayer-1]
            nInputs = self.nNeurons[iLayer-1]
            layerName =  'hidden'+str(iLayer)

            #Workaround for a problem with shape infering. Is is better fi the x placeholder has
            #indefinite shape. In this case we made the first layer by hand, and then on the shapes
            #are well defined.
            if iLayer < 10:
                aLayer = nn_layer(previousLayer, nInputs, self.nNeurons[iLayer], layerName, self.trainingMode, act=tf.nn.elu)
            else:    
                aLayer = tf.layers.dense(inputs = previousLayer, units = self.nNeurons[iLayer],
                                         name = layerName,
                                         activation = tf.nn.elu)
            self.myLayers.append(aLayer)

##############################################################################
##############################################################################
    def addDropoutLayer(self):

        lastLayer = self.myLayers[-1]

        with tf.name_scope('dropout'):
            aLayer = tf.layers.dropout(inputs = lastLayer, rate = self.dropout_prob, training=self.trainingMode)
            self.myLayers.append(aLayer)

            ##############################################################################
##############################################################################
    def addOutputLayer(self):

        lastLayer = self.myLayers[-1]
        #Do not apply softmax activation yet as softmax is calculated during cross enetropy evaluation
        aLayer = tf.layers.dense(inputs = lastLayer, units = self.nOutputNeurons,
                                 name = 'output',
                                 activation = tf.identity)
        self.myLayers.append(aLayer)

##############################################################################
##############################################################################
    def defineOptimizationStrategy(self):              
        with tf.name_scope('train'):
            samplesWeights = 1.0

            if self.nOutputNeurons>1:
                onehot_labels = tf.one_hot(tf.to_int32(self.yTrue), depth=self.nOutputNeurons, axis=-1)
                tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=self.myLayers[-1], weights=samplesWeights)
            else:
                #mean_squared_error = tf.losses.mean_squared_error(labels=self.yTrue, predictions=self.myLayers[-1], weights=samplesWeights)
                absolute_difference = tf.losses.absolute_difference(labels=self.yTrue, predictions=self.myLayers[-1], weights=samplesWeights)

            l2_regularizer =tf.contrib.layers.l2_regularizer(self.lambdaLagrange)
            modelParameters   = tf.trainable_variables()
            tf.contrib.layers.apply_regularization(l2_regularizer, modelParameters)
            lossFunction = tf.losses.get_total_loss()

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(lossFunction)
                                    
        with tf.name_scope('performance'):
            response = self.myLayers[-1]
            if self.nOutputNeurons>1:
                response = tf.nn.softmax(response)
                response = tf.math.argmax(response, axis=(1))
                response = tf.to_float(response)
                
            response = tf.reshape(response, (-1,1), name="response")
            labels =  self.yTrue
            pull = (response - labels)/labels

            my_update_ops = tf.get_collection("MY_UPDATE_OPS")
            my_running_vals = tf.get_collection("MY_MOVING_VALS")

            entropyLoss = tf.get_default_graph().get_operation_by_name("model/train/total_loss").outputs[0]
            l2Loss = tf.get_default_graph().get_operation_by_name("model/train/get_regularization_penalty").outputs[0]
            
            l2Loss_mean, _ = tf.metrics.mean(values=l2Loss, name="mean_l2Loss", updates_collections=["MY_UPDATE_OPS"])
            entropyLoss_mean, _ = tf.metrics.mean(values=entropyLoss, name="mean_entropyLoss", updates_collections=["MY_UPDATE_OPS"])            
            pull_mean, _ = tf.metrics.mean(values=pull, name="mean_pull", updates_collections=["MY_UPDATE_OPS"])
            pull_rms, _ = tf.metrics.root_mean_squared_error(labels=self.yTrue/self.yTrue - 1, predictions=pull, name="pull_rms", updates_collections=["MY_UPDATE_OPS"])

            tf.add_to_collection("MY_RUNNING_VALS",l2Loss_mean)
            tf.add_to_collection("MY_RUNNING_VALS",entropyLoss_mean)
            tf.add_to_collection("MY_RUNNING_VALS",pull_mean)
            tf.add_to_collection("MY_RUNNING_VALS",pull_rms)
            
            tf.summary.scalar('entropyLoss_mean', entropyLoss_mean)
            tf.summary.scalar('pull_rms', pull_rms)
            tf.summary.scalar('pull_mean', pull_mean)
                
##############################################################################
##############################################################################
    def __init__(self, inputIterator, nNeurons, nOutputNeurons, learning_rate, lambdaLagrange):

        with tf.name_scope('model'):

            self.myLayers = [inputIterator[1]]
            self.yTrue = inputIterator[0]
                
            self.nNeurons = nNeurons
            self.nOutputNeurons = nOutputNeurons
        
            self.nLayers = len(self.nNeurons)

            self.learning_rate = learning_rate
            self.lambdaLagrange = lambdaLagrange

            self.trainingMode = tf.placeholder(tf.bool, name="trainingMode")
            self.dropout_prob = tf.placeholder(tf.float32, name="dropout_prob")

            self.addFCLayers()
            self.addDropoutLayer()
            self.addOutputLayer()
            self.defineOptimizationStrategy()
            
            # Merge all the summaries
            with tf.name_scope('monitor'): 
                merged = tf.summary.merge_all()
        
##############################################################################
##############################################################################
##############################################################################
