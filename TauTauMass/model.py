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
            if iLayer == 1:
                aLayer = nn_layer(previousLayer, nInputs, self.nNeurons[iLayer], layerName, act=tf.nn.elu)
            else:    
                aLayer = tf.layers.dense(inputs = previousLayer, units = self.nNeurons[iLayer],
                                         name = layerName,
                                         activation = tf.nn.elu)
            self.myLayers.append(aLayer)

    def addDropoutLayer(self):

        lastLayer = self.myLayers[-1]

        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
            aLayer = tf.layers.dropout(inputs = lastLayer, rate = keep_prob)
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
            #absolute_difference = tf.losses.absolute_difference(labels=self.yTrue, predictions=self.myLayers[-1])
            mean_squared_error = tf.losses.mean_squared_error(labels=self.yTrue, predictions=self.myLayers[-1])
            l2_regularizer =tf.contrib.layers.l2_regularizer(self.lambdaLagrange)
            modelParameters   = tf.trainable_variables()
            tf.contrib.layers.apply_regularization(l2_regularizer, modelParameters)
            #underestimateLoss = tf.to_float(tf.less(self.myLayers[-1], self.yTrue))
            #tf.losses.add_loss(underestimateLoss)
            lossFunction = tf.losses.get_total_loss()
            train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(lossFunction)

        with tf.name_scope('performance'):
            y = self.myLayers[-1]
            pull = (y - self.yTrue)/self.yTrue
            pull_mean, pull_variance = tf.nn.moments(pull, axes=[0])            

        tf.summary.scalar('loss', lossFunction)
        tf.summary.scalar('pull_rms', tf.sqrt(pull_variance[0]))

    def __init__(self, x, yTrue, nNeurons, learning_rate, lambdaLagrange):

        self.nNeurons = nNeurons
        
        self.nLayers = len(self.nNeurons)

        self.myLayers = [x]

        #batchNormalized = tf.layers.batch_normalization(x)
        #self.myLayers.append(batchNormalized)
        
        self.yTrue = yTrue

        self.learning_rate = learning_rate
        self.lambdaLagrange = lambdaLagrange

        self.addFCLayers()
        self.addDropoutLayer()
        self.addOutputLayer()
        self.defineOptimizationStrategy()
##############################################################################
##############################################################################
##############################################################################
