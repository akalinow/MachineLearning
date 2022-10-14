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
cumulativePosteriorCut = 0.7
###################################################
###################################################
def plotEvent(element, label2Pt):
  
  image = element[0][0]
  labels = element[1]  
  pt = label2Pt(labels[0][0])[0]
  charge =   labels[1][0]*2-1
  image = plt.matshow(image, aspect = 5.0, origin = "lower", cmap="gnuplot2")
  plt.title("Gen: pt: {:+.1f}, charge: {}".format(pt,charge))

###################################################
###################################################
def plotPosterior(ptGen, labels, predictions, label2Pt, testIndex=0):
    
    fig, axes = plt.subplots(1, 2, figsize = (10, 5))
            
    if testIndex==0:
        indices = np.logical_and(labels>ptGen-0.1, labels<ptGen+0.1)
        predictions = predictions[indices]
        
    predictions = predictions[testIndex] 
    ###
    #x = np.roll(predictions, 1, axis=0)
    #predictions -=x
    ###
    
    predictions = tf.reshape(predictions, (1,-1))    
    predictions = np.mean(predictions, axis=0)
    maxPosterior = tf.math.reduce_max(predictions)
    scaleFactor = int(0.8/maxPosterior + 0.5)
    axes[0].plot(label2Pt(np.arange(predictions.shape[0])), scaleFactor*predictions, label="{}xposterior".format(scaleFactor))
    axes[0].plot(label2Pt(np.arange(predictions.shape[0])), np.cumsum(predictions), linestyle='-.',label="cumulative posterior")
    axes[1].plot(label2Pt(np.arange(predictions.shape[0])), scaleFactor*predictions, label="{}xposterior".format(scaleFactor))
    
    predictions = np.cumsum(predictions, axis=0)>cumulativePosteriorCut
    predictions = np.argmax(predictions, axis=0)
    ptRec = label2Pt(predictions)
    print("Pt gen = {:+.1f}, Pt rec {} cumulative posterior cut: {}".format(ptGen, ptRec, cumulativePosteriorCut))
    axes[0].axvline(ptGen, linestyle='-', color="olivedrab", label=r'$p_{T}^{GEN} \pm 1 [GeV/c]$')
    
    axes[0].set_xlabel(r'$p_{T} [GeV/c]$')
    axes[0].set_ylabel('Value')
    axes[0].set_xlim([0, 2*ptGen])
    axes[0].set_ylim([1E-3,1.05])
    
    axes[0].legend(bbox_to_anchor=(2.5,1), loc='upper left')
    axes[1].set_xlabel(r'$p_{T} [GeV/c]$')
    axes[1].set_ylabel('Value')
    axes[1].set_xlim([0,201])
    #axes[1].set_ylim([1E-3,1.05])
    plt.subplots_adjust(bottom=0.15, left=0.05, right=0.95, wspace=0.3)
    plt.savefig("fig_png/Posterior_ptGen_{}.png".format(ptGen), bbox_inches="tight")
###################################################
###################################################    
def plotTurnOn(df, ptCut):
    
    ptMax = ptCut+10
    nPtBins = int(ptMax*2.0)
    ptHistoBins = range(0,nPtBins+1)
    
    denominator, _ = np.histogram(df["genPt"], bins=ptHistoBins) 
    numerator_OMTF, _ = np.histogram(df[df["OMTF_pt"]>=ptCut]["genPt"], bins=ptHistoBins)
    numerator_NN, _ = np.histogram(df[(df["NN_pt"]>=ptCut)&(df["NN_prob"]>0.07)]["genPt"], bins=ptHistoBins)
      
    ratio_OMTF = np.divide(numerator_OMTF, denominator, out=np.zeros(denominator.shape), where=denominator>0)
    ratio_NN = np.divide(numerator_NN, denominator, out=np.zeros(denominator.shape), where=denominator>0)
    
    fig, axes = plt.subplots(1, 3)
    axes[0].plot(ptHistoBins[:-1],numerator_OMTF, "r", label="OMTF",linewidth=2)
    axes[0].plot(ptHistoBins[:-1],numerator_NN, "b", label="NN", linewidth=2)
    axes[0].set_xlim([0,2.0*ptCut])
    axes[0].set_xlabel(r'$p_{T}^{GEN}$')
    axes[0].set_ylabel('Events passing pT cut')
    axes[0].legend(loc='upper left')
    
    axes[1].plot(ptHistoBins[:-1],ratio_OMTF, "ro", label="OMTF", linewidth=2)
    axes[1].plot(ptHistoBins[:-1],ratio_NN, "bo", label="NN")
    axes[1].grid()
    axes[1].set_yscale("log")
    axes[1].set_xlim([0,ptMax])
    axes[1].set_ylim([1E-3,1.05])
    axes[1].set_xlabel(r'$p_{T}^{GEN}$')
    axes[1].set_ylabel('Efficiency')
    
    axes[2].plot(ptHistoBins[:-1],ratio_OMTF, "r",label="OMTF")
    axes[2].plot(ptHistoBins[:-1],ratio_NN, "b", label="NN")
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
    
    if cm.shape[0]==2:
        vmax = 1.0
        sns.heatmap(cm, ax = axis, vmax = vmax, annot=annot, xticklabels=("-1", "1"), yticklabels=("-1", "1"), cmap=palette)
        axis.set_ylabel(r'$q^{REC}$')
        axis.set_xlabel(r'$q^{GEN}$')
    else:
        vmax = 1.0
        sns.heatmap(cm, ax = axis, vmax = vmax, annot=annot, xticklabels=5, yticklabels=5, cmap=palette)
        axis.set_ylabel(r'$p_{T}^{REC} \rm{[bin ~number]}$')
        axis.set_xlabel(r'$p_{T}^{GEN} \rm{[bin ~number]}$')
        axis.grid()
        
    axis.set_title(modelName)   
    
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
def getVxMuRate(x):
    
    #Some newer parametriation do not remember source
    params = np.array([-0.235801, -2.82346, 17.162])
    integratedRate = np.power(x,params[0]*np.log(x) + params[1])*np.exp(params[2])  
    differentialRate = -np.power(x,params[0]*np.log(x) + params[1] -1)*np.exp(params[2])*(2*params[0]*np.log(x)+params[1])
    
    ##RPCConst.h parametrisation from CMS-TN-1995/150
    dabseta = 1.23 - 0.8
    lum = 1.0
    dpt = 1.0;
    afactor = 1.0e-34*lum*dabseta*dpt
    a  = 2*1.3084E6;
    mu=-0.725;
    sigma=0.4333;
    s2=2*sigma*sigma;
   
    ptlog10 = np.log10(x);
    ex = (ptlog10-mu)*(ptlog10-mu)/s2;
    rate = (a * np.exp(-ex) * afactor); 
    ######
    
    return differentialRate
###################################################
###################################################
def getVsMuRateWeight(x, hist, bins):
       
    weightToFlatSpectrum = np.divide(1.0, hist, out=np.zeros(hist.shape), where=hist>0)  
    binNumber = np.digitize(x,bins) -1  
    weight = getVxMuRate(x)*weightToFlatSpectrum[binNumber]    
    return weight
###################################################
###################################################
def plotRate(df):
    
    from matplotlib.gridspec import GridSpec
        
    ptHistoBins = np.concatenate((np.arange(2,201,1), [9999]))  
    genPtHist, bin_edges = np.histogram(df["genPt"], bins=ptHistoBins) 
    weights = getVsMuRateWeight(df["genPt"], genPtHist, bin_edges)
       
    genPtHist_weight, bin_edges = np.histogram(df["genPt"], bins=ptHistoBins, weights=weights) 
    genPtHist_weight = np.sum(genPtHist_weight) - np.cumsum(genPtHist_weight)
    
    omtfPtHist_weight, bin_edges = np.histogram(df["OMTF_pt"], bins=ptHistoBins, weights=weights) 
    omtfPtHist_weight = np.sum(omtfPtHist_weight) - np.cumsum(omtfPtHist_weight)
    
    nnPtHist_weight, bin_edges = np.histogram(df["NN_pt"], bins=ptHistoBins, weights=weights) 
    nnPtHist_weight = np.sum(nnPtHist_weight) - np.cumsum(nnPtHist_weight)
        
    #fig, axes = plt.subplots(2, 1, sharex=True)
    #fig.subplots_adjust(hspace=0.1)
    
    fig = plt.figure()
    gs = GridSpec(6, 6, figure=fig)
    axes = [0,0]
    axes[0] = fig.add_subplot(gs[0:4, :])
    axes[1] = fig.add_subplot(gs[5:, :])
    
    axes[0].step(ptHistoBins[:-1], genPtHist_weight, label="Vxmurate", linewidth=3, color="black", where='post')
    axes[0].step(ptHistoBins[:-1], omtfPtHist_weight, label="OMTF", linewidth=3, color="r", where='post')
    axes[0].step(ptHistoBins[:-1], nnPtHist_weight, label="NN", linewidth=3, color="b", where='post')
    axes[0].set_xlim([2,60])
    axes[0].set_ylim([10,1E6])
    axes[0].set_ylabel('Rate [arb. units]')
    axes[0].legend(loc='upper right')
    axes[0].grid()
    axes[0].set_yscale("log")
      
    ratio = np.divide(omtfPtHist_weight, nnPtHist_weight, out=np.zeros_like(nnPtHist_weight), where=nnPtHist_weight>0)  
    axes[1].step(ptHistoBins[:-1], ratio, label="OMTF/NN", linewidth=3, color="black", where='post')
    axes[1].set_xlim([2,60])
    axes[1].set_ylim([0.9,2.1])
    axes[1].set_xlabel(r'$p_{T}^{cut}$')
    axes[1].set_ylabel('OMTF/NN')
    #axes[1].legend(loc='upper right')
    axes[1].grid()
    
    #plt.subplots_adjust(bottom=0.15, left=0.05, right=0.95, wspace=0.5)
    plt.savefig("fig_png/Rate.png", bbox_inches="tight")
###################################################
###################################################