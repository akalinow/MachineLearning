import tensorflow as tf
import pandas as pd
import seaborn as sns
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve
from functools import partial

###################################################
###################################################
params = {'legend.fontsize': 'xx-large',
          'figure.figsize': (10, 7),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
plt.rcParams.update(params)
###################################################
###################################################
cumulativePosteriorCut = 0.70
testIndex = 0
###################################################
###################################################
def plotPosterior(ptGen, labels, predictions):
    
    fig, axes = plt.subplots(1, 2, figsize = (10, 5))
    indices = np.logical_and(labels>ptGen-0.1, labels<ptGen+0.1)
        
    predictions = predictions[indices]
    
    ###TEST
    predictions = predictions[testIndex] 
    predictions = tf.reshape(predictions, (1,-1))
    ######
    
    predictions = np.mean(predictions, axis=0)
    maxPosterior = tf.math.reduce_max(predictions)
    scaleFactor = int(0.8/maxPosterior + 0.5)
    axes[0].plot(label2Pt(np.arange(predictions.shape[0])), scaleFactor*predictions, label="{}xposterior".format(scaleFactor))
    axes[0].plot(label2Pt(np.arange(predictions.shape[0])), np.cumsum(predictions), linestyle='-.',label="cumulative posterior")
    axes[1].plot(label2Pt(np.arange(predictions.shape[0])), scaleFactor*predictions, label="{}xposterior".format(scaleFactor))
    
    predictions = np.cumsum(predictions, axis=0)>cumulativePosteriorCut
    predictions = np.argmax(predictions, axis=0)
    ptRec = label2Pt(predictions)
    print("Pt gen = {}, Pt rec {} cumulative posterior: {}".format(ptGen, cumulativePosteriorCut, ptRec))
    axes[0].axvline(ptGen, linestyle='-', color="olivedrab", label=r'$p_{T}^{GEN} \pm 1 [GeV/c]$')
    axes[0].axvline(ptRec, linestyle='--', color="r", label=r'$p_{T}^{REC} @ cum~post.=$'+str(cumulativePosteriorCut))
    
    axes[0].set_xlabel(r'$p_{T} [GeV/c]$')
    axes[0].set_ylabel('Value')
    axes[0].set_xlim([0, 2*ptGen])
    axes[0].set_ylim([1E-3,1.05])
    
    axes[0].legend(bbox_to_anchor=(2.5,1), loc='upper left')
    axes[1].set_xlabel(r'$p_{T} [GeV/c]$')
    axes[1].set_ylabel('Value')
    axes[1].set_xlim([0,201])
    axes[1].set_ylim([1E-3,1.05])
    plt.subplots_adjust(bottom=0.15, left=0.05, right=0.95, wspace=0.3)
    plt.savefig("fig_png/Posterior_ptGen_{}.png".format(ptGen), bbox_inches="tight")
###################################################
###################################################    
def plotTurnOn(dataset, ptCut):
    ptMax = ptCut+50
    nPtBins = int(ptMax*2.0)
    ptHistoBins = range(0,nPtBins+1)

    num = np.zeros(nPtBins)
    numML = np.zeros(nPtBins)
    denom = np.zeros(nPtBins)
    
    count =0
    for aBatch in dataset.as_numpy_iterator():
        labels = aBatch[1][0]
        omtfPredictions = aBatch[2]
        count += labels.shape[0]
        predictions = model.predict(aBatch[0], use_multiprocessing=True)
        predictions = predictions[0]
        predictions = predictions[:,0:45] #TEST
        predictions = np.cumsum(predictions, axis=1)>cumulativePosteriorCut
        predictions = np.argmax(predictions, axis=1)   
        predictions = label2Pt(predictions)
        
        tmp,_ = np.histogram(labels, bins=ptHistoBins)    
        denom +=tmp
        tmp,_ = np.histogram(labels[omtfPredictions>=ptCut], bins=ptHistoBins)
        num += tmp
        tmp,_ = np.histogram(labels[predictions>=ptCut], bins=ptHistoBins)
        numML += tmp
        
    fig, axes = plt.subplots(1, 3)
    ratio = np.divide(num, denom, out=np.zeros_like(denom), where=denom>0)
    ratioML = np.divide(numML, denom, out=np.zeros_like(denom), where=denom>0)
    axes[0].plot(ptHistoBins[:-1],num, label="OMTF")
    axes[0].plot(ptHistoBins[:-1],numML, label="ML")
    axes[0].set_xlim([0,2.0*ptCut])
    #axes[0].set_ylim([0,1.0])
    axes[0].set_xlabel(r'$p_{T}^{GEN}$')
    axes[0].set_ylabel('Events passing pT cut')
    axes[0].legend(loc='upper left')
    
    axes[1].plot(ptHistoBins[:-1],ratio, label="OMTF")
    axes[1].plot(ptHistoBins[:-1],ratioML, label="ML")
    axes[1].grid()
    axes[1].set_yscale("log")
    axes[1].set_xlim([0,ptMax])
    axes[1].set_ylim([1E-3,1.05])
    axes[1].set_xlabel(r'$p_{T}^{GEN}$')
    axes[1].set_ylabel('Efficiency')

    axes[2].plot(ptHistoBins[:-1],ratio, label="OMTF")
    axes[2].plot(ptHistoBins[:-1],ratioML, label="ML")
    axes[2].grid()
    axes[2].axhline(y=0.5)
    axes[2].axhline(y=0.85)
    axes[2].axvline(x=ptCut)
    axes[2].set_xlim([0,ptMax])
    axes[2].set_ylim([0.0,1.05])
    axes[2].set_xlabel(r'$p_{T}^{GEN}$')
    axes[2].set_ylabel('Efficiency')
    plt.subplots_adjust(bottom=0.15, left=0.05, right=0.95, wspace=0.5)
    plt.savefig("fig_png/TurnOn_ptCut_{}.png".format(ptCut), bbox_inches="tight")    
###################################################
###################################################   
def plotPull(labels, predictions, omtfPredictions):
    
    minX = -1
    maxX = 2
    nBins = 50
    predictions = np.cumsum(predictions, axis=1)>cumulativePosteriorCut
    predictions = np.argmax(predictions, axis=1)   
    predictions = label2Pt(predictions)   
    error = (predictions - labels)/labels
    omtfError = (omtfPredictions - labels)/labels    
    fig, axes = plt.subplots(1, 2, figsize = (12, 5))  
    axes[0].hist(error, range=(minX, maxX), bins = nBins, color="deepskyblue", label = "NN")
    axes[0].hist(omtfError, range=(minX, maxX), bins = nBins, color="tomato", label="OMTF")
    axes[0].set_xlabel("(Model - True)/True")
    axes[0].legend(loc='upper right')
    axes[0].set_xlim([minX, maxX])
    #axes[0].set_ylim([-2,2])
    
    axes[1].hist(omtfError, range=(minX, maxX), bins = nBins, color="tomato", label="OMTF")
    axes[1].hist(error, range=(minX, maxX), bins = nBins, color="deepskyblue", label = "NN")
    axes[1].set_xlabel("(Model - True)/True")
    axes[1].legend(loc='upper right')
    axes[1].set_xlim([minX, maxX])
    plt.savefig("fig_png/Pull.png", bbox_inches="tight")
###################################################
################################################### 
def plotCM(labels, predictions, omtfPredictions):
    
    fig, axes = plt.subplots(2, 2, figsize = (10, 10))  
    
    ptMax =  ptBins.shape[0]  
    vmax = 1.0
    ptPredictions = np.cumsum(predictions[0], axis=1)>cumulativePosteriorCut
    ptPredictions = np.argmax(ptPredictions, axis=1)   
    cm = tf.math.confusion_matrix(pT2Label(labels[0]), ptPredictions)
    cm = tf.cast(cm, dtype=tf.float32)
    cm = tf.math.divide_no_nan(cm, tf.math.reduce_sum(cm, axis=1)[:, np.newaxis])
    cm = tf.transpose(cm)
    #vmax = tf.math.reduce_max(cm)
    
    myPalette = sns.color_palette("YlGnBu", n_colors=20)
    myPalette[0] = (1,1,1)
    
    vmax = 0.1 #TEST
    sns.heatmap(cm, ax = axes[0,0], vmax = vmax, annot=False, xticklabels=4, yticklabels=4, cmap=myPalette)
    axes[0,0].set_ylabel(r'$p_{T}^{NN} \rm{[bin ~number]}$');
    axes[0,0].set_xlabel(r'$p_{T}^{GEN} \rm{[bin ~number]}$');
    axes[0,0].grid()
    axes[0,0].set_ylim([0,ptMax])
    axes[0,0].set_xlim([0,ptMax])
    axes[0,0].set_aspect(aspect='equal')
    axes[0,0].set_title("NN")
    
    cm = tf.math.confusion_matrix(pT2Label(labels[0]), pT2Label(omtfPredictions[0]))
    cm = tf.cast(cm, dtype=tf.float32)
    cm = tf.math.divide_no_nan(cm, tf.math.reduce_sum(cm, axis=1)[:, np.newaxis])
    cm = tf.transpose(cm)
    #vmax = tf.math.reduce_max(cm)
    sns.heatmap(cm, ax = axes[0,1], vmax = vmax, annot=False, xticklabels=4, yticklabels=4, cmap=myPalette)
    axes[0,1].grid()
    axes[0,1].set_title("OMTF")
    axes[0,1].set_xlim([0,ptMax])
    axes[0,1].set_ylim([0,ptMax])
    axes[0,1].set_aspect(aspect='equal')
    axes[0,1].set_ylabel(r'$p_{T}^{OMTF} \rm{[bin ~number]}$')
    axes[0,1].set_xlabel(r'$p_{T}^{GEN} \rm{[bin ~number]}$') 
        
    chPredictions =tf.map_fn(lambda x: x>0.5, predictions[1], dtype=tf.bool)
    chPredictions = tf.reshape(chPredictions, (-1,1))
    chLabels =tf.map_fn(lambda x: x>0.5, labels[1], dtype=tf.bool)
    #chLabels = tf.reshape(chLabels, (-1,1))
    
    vmax = 1.0
    vmin = 0.0
    cm = tf.math.confusion_matrix(chLabels, chPredictions)
    cm = tf.cast(cm, dtype=tf.float32)
    cm = tf.math.divide_no_nan(cm, tf.math.reduce_sum(cm, axis=1)[:, np.newaxis])
    cm = tf.transpose(cm)
    sns.heatmap(cm, ax = axes[1,0], vmax = vmax, annot=True, cmap=myPalette)
    axes[1,0].set_title("NN")
    axes[1,0].set_aspect(aspect='equal')
    axes[1,0].set_ylabel(r'$q^{NN}$')
    axes[1,0].set_xlabel(r'$q^{GEN}$') 
    
    chPredictions =tf.map_fn(lambda x: x>0.5, omtfPredictions[1], dtype=tf.bool)  
    chPredictions = tf.reshape(chPredictions, (-1,1))
    cm = tf.math.confusion_matrix(chLabels, chPredictions)
    cm = tf.cast(cm, dtype=tf.float32)
    cm = tf.math.divide_no_nan(cm, tf.math.reduce_sum(cm, axis=1)[:, np.newaxis])
    cm = tf.transpose(cm)
    sns.heatmap(cm, ax = axes[1,1], vmax = vmax, annot=True, cmap=myPalette, linewidths=0.01)
    axes[1,1].set_title("OMTF")
    axes[1,1].set_aspect(aspect='equal')
    axes[1,1].set_ylabel(r'$q^{NN}$')
    axes[1,1].set_xlabel(r'$q^{GEN}$') 
    
    plt.subplots_adjust(bottom=0.15, left=0.05, right=0.95, wspace=0.25, hspace=0.35)
    plt.savefig("fig_png/CM.png", bbox_inches="tight")
###################################################
###################################################

###################################################
###################################################