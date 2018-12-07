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

            #Workaround for a problem with shape infering. Is is better if the x placeholder has
            #indefinite shape. In this case we made the first layer by hand, and then on the shapes
            #are well defined.
            if iLayer < 10:
                aLayer = nn_layer(previousLayer, nInputs, self.nNeurons[iLayer], layerName, self.trainingMode, act=tf.nn.elu)
            else:    
                aLayer = tf.layers.dense(inputs = previousLayer, units = self.nNeurons[iLayer],
                                         name = layerName,
                                         activation = tf.nn.elu)
            self.myLayers.append(aLayer)

    def addDropoutLayer(self):

        lastLayer = self.myLayers[-1]

        with tf.name_scope('dropout'):
            aLayer = tf.layers.dropout(inputs = lastLayer, rate = self.dropout_prob, training=self.trainingMode)
            self.myLayers.append(aLayer)

    def addOutputLayer(self):

        lastLayer = self.myLayers[-1]
        #Do not apply softmax activation yet as softmax is calculated during cross enetropy evaluation
        aLayer = tf.layers.dense(inputs = lastLayer, units = 1,
                                     name = 'output',
                                     activation = tf.identity)
        self.myLayers.append(aLayer)

    def defineOptimizationStrategy(self):              
        with tf.name_scope('train'):
            #samplesWeights = tf.to_float(self.yTrue>0.5) + 100
            samplesWeights = 1.0
            sigmoid_cross_entropy = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.yTrue, logits=self.myLayers[-1], weights=samplesWeights)
            l2_regularizer =tf.contrib.layers.l2_regularizer(self.lambdaLagrange)
            modelParameters   = tf.trainable_variables()
            tf.contrib.layers.apply_regularization(l2_regularizer, modelParameters)
            lossFunction = tf.losses.get_total_loss()

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(lossFunction)

        with tf.name_scope('performance'):
            y = tf.nn.sigmoid(self.myLayers[-1])
            accuracy = tf.metrics.accuracy(self.yTrue, y>0.5, weights=samplesWeights)
            #accuracy = tf.reduce_mean(is_correct_prediction)

        tf.summary.scalar('loss', lossFunction)
        #tf.summary.scalar('accuracy', accuracy)

    def __init__(self, x, yTrue, nNeurons, learning_rate, lambdaLagrange):

        self.nNeurons = nNeurons
        
        self.nLayers = len(self.nNeurons)

        self.myLayers = [x]

        self.yTrue = yTrue

        self.learning_rate = learning_rate
        self.lambdaLagrange = lambdaLagrange

        self.trainingMode = tf.placeholder(tf.bool, name="trainingMode")
        self.dropout_prob = tf.placeholder(tf.float32, name="dropout_prob")

        self.addFCLayers()
        self.addDropoutLayer()
        self.addOutputLayer()
        self.defineOptimizationStrategy()
##############################################################################
##############################################################################
##############################################################################
