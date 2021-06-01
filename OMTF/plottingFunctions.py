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
def plotPosterior(ptGen, labels, predictions, label2Pt):
    
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
def plotTurnOn(df, ptCut):
    
    ptMax = ptCut+50
    nPtBins = int(ptMax*2.0)
    ptHistoBins = range(0,nPtBins+1)
    
    denominator, _ = np.histogram(df["genPt"], bins=ptHistoBins) 
    numerator_OMTF, _ = np.histogram(df[df["OMTF_pt"]>=ptCut]["genPt"], bins=ptHistoBins)
    numerator_NN, _ = np.histogram(df[df["NN_pt"]>=ptCut]["genPt"], bins=ptHistoBins)
      
    ratio_OMTF = np.divide(numerator_OMTF, denominator, out=np.zeros(denominator.shape), where=denominator>0)
    ratio_NN = np.divide(numerator_NN, denominator, out=np.zeros(denominator.shape), where=denominator>0)
    
    fig, axes = plt.subplots(1, 3)
    axes[0].plot(ptHistoBins[:-1],numerator_OMTF, label="OMTF")
    axes[0].plot(ptHistoBins[:-1],numerator_NN, label="NN")
    axes[0].set_xlim([0,2.0*ptCut])
    axes[0].set_xlabel(r'$p_{T}^{GEN}$')
    axes[0].set_ylabel('Events passing pT cut')
    axes[0].legend(loc='upper left')
    
    axes[1].plot(ptHistoBins[:-1],ratio_OMTF, label="OMTF")
    axes[1].plot(ptHistoBins[:-1],ratio_NN, label="NN")
    axes[1].grid()
    axes[1].set_yscale("log")
    axes[1].set_xlim([0,ptMax])
    axes[1].set_ylim([1E-3,1.05])
    axes[1].set_xlabel(r'$p_{T}^{GEN}$')
    axes[1].set_ylabel('Efficiency')

    axes[2].plot(ptHistoBins[:-1],ratio_OMTF, label="OMTF")
    axes[2].plot(ptHistoBins[:-1],ratio_NN, label="NN")
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
def plotPull(df, minX=-1, maxX=2, nBins=50):
       
    pull_OMTF = (df["OMTF_pt"] - df["genPt"])/df["genPt"]
    pull_NN = (df["NN_pt"] - df["genPt"])/df["genPt"]

    fig, axes = plt.subplots(1, 2, figsize = (12, 5))  
    axes[0].hist(pull_NN, range=(minX, maxX), bins = nBins, color="deepskyblue", label = "NN")
    axes[0].hist(pull_OMTF, range=(minX, maxX), bins = nBins, color="tomato", label="OMTF")
    axes[0].set_xlabel("(Model - True)/True")
    axes[0].legend(loc='upper right')
    axes[0].set_xlim([minX, maxX])
    
    axes[1].hist(pull_OMTF, range=(minX, maxX), bins = nBins, color="tomato", label="OMTF")
    axes[1].hist(pull_NN, range=(minX, maxX), bins = nBins, color="deepskyblue", label = "NN")
    axes[1].set_xlabel("(Model - True)/True")
    axes[1].legend(loc='upper right')
    axes[1].set_xlim([minX, maxX])
    plt.savefig("fig_png/Pull.png", bbox_inches="tight")
###################################################
################################################### 
def plotSingleCM(gen_labels, model_labels, modelName, palette, annot, axis):
    
    vmax = 1.0
    cm = tf.math.confusion_matrix(gen_labels, model_labels)
    cm = tf.cast(cm, dtype=tf.float32)
    cm = tf.math.divide_no_nan(cm, tf.math.reduce_sum(cm, axis=1)[:, np.newaxis])
    cm = tf.transpose(cm)
    
    sns.heatmap(cm, ax = axis, vmax = vmax, annot=annot, xticklabels=5, yticklabels=5, cmap=palette)
    axis.set_title(modelName)
    axis.set_ylabel(r'$p_{T}^{REC} \rm{[bin ~number]}$');
    axis.set_xlabel(r'$p_{T}^{GEN} \rm{[bin ~number]}$');
    axis.grid()
    
    max_label = np.amax([gen_labels,model_labels])+1
    axis.set_ylim([0,max_label])
    axis.set_xlim([0,max_label])
    axis.set_aspect(aspect='equal')
    axis.set_title(modelName)    
###################################################
###################################################
def plotCM(df, pT2Label):
     
    gen_labels = pT2Label(df["genPt"])  
    NN_labels = pT2Label(df["NN_pt"])
    OMTF_labels = pT2Label(df["OMTF_pt"])
 
    fig, axes = plt.subplots(2, 2, figsize = (10, 10))  
    myPalette = sns.color_palette("YlGnBu", n_colors=20)
    myPalette[0] = (1,1,1)
     
    gen_labels = pT2Label(df["genPt"])  
    NN_labels = pT2Label(df["NN_pt"])
    OMTF_labels = pT2Label(df["OMTF_pt"])
    
    plotSingleCM(gen_labels, NN_labels, "NN", myPalette, False, axes[0,0])
    plotSingleCM(gen_labels, OMTF_labels, "OMTF", myPalette, False, axes[0,1])
          
    gen_labels = df["genCharge"]  
    NN_labels = df["NN_charge"]
    OMTF_labels = df["OMTF_charge"]
    
    plotSingleCM(gen_labels, NN_labels, "NN", myPalette, True, axes[1,0])
    plotSingleCM(gen_labels, OMTF_labels, "OMTF", myPalette, True, axes[1,1])
     
    plt.subplots_adjust(bottom=0.15, left=0.05, right=0.95, wspace=0.25, hspace=0.35)
    plt.savefig("fig_png/CM.png", bbox_inches="tight")
###################################################
###################################################

###################################################
###################################################