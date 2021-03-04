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
def plotPull(df, minX = -1, maxX=2, nBins=50):
    pull_nn = (df["NN"] - df["genMass"])/df["genMass"]
    pull_fastMTT = (df["fastMTT"] - df["genMass"])/df["genMass"]
    
    fig, axes = plt.subplots(1, 3, figsize = (15, 5))  
    
    axes[0].hist(pull_nn, range=(minX, maxX), bins = nBins, color="deepskyblue", label = "NN")
    axes[0].hist(pull_fastMTT, range=(minX, maxX), bins = nBins, color="tomato", label="fastMTT")      
    axes[0].set_xlabel("(Model - True)/True")
    axes[0].legend(loc='upper right')
    axes[0].set_xlim([minX, maxX])
    axes[0].set_title("fastMTT in foreround")
    
    axes[1].hist(pull_fastMTT, range=(minX, maxX), bins = nBins, color="tomato", label="fastMTT")
    axes[1].hist(pull_nn, range=(minX, maxX), bins = nBins, color="deepskyblue", label = "NN")
    axes[1].set_xlabel("(Model - True)/True")
    axes[1].legend(loc='upper right')
    axes[1].set_xlim([minX, maxX])
    axes[1].set_title("fastMTT in background")
    
    axes[2].hist(pull_fastMTT, range=(minX, maxX), bins = nBins, color="tomato", label="fastMTT")
    axes[2].hist(pull_nn, range=(minX, maxX), bins = nBins, color="deepskyblue", label = "NN")
    axes[2].set_xlabel("(Model - True)/True")
    axes[2].legend(loc='upper right')
    axes[2].set_xlim([minX, maxX])
    axes[2].set_title("fastMTT in background")
    axes[2].semilogy()
    
    plt.subplots_adjust(bottom=0.15, left=0.05, right=0.95, wspace=0.3)
    plt.savefig("fig_png/Pull.png", bbox_inches="tight")
###################################################
###################################################
def plotCM(df, mass2Label, vmax=1.0):
        
    cm = tf.math.confusion_matrix(mass2Label(df["genMass"]), mass2Label(df["NN"]))
    cm = tf.cast(cm, dtype=tf.float32)
    cm = tf.math.divide_no_nan(cm, tf.math.reduce_sum(cm, axis=1)[:, np.newaxis])
    cm = tf.transpose(cm)
    
    myPalette = sns.color_palette("YlGnBu", n_colors=20)
    myPalette[0] = (1,1,1)
      
    massMax = mass2Label([9999.0])   
    fig, axes = plt.subplots(1, 2, figsize = (12, 5))  
    sns.heatmap(cm, ax = axes[0], vmax = vmax, annot=False, xticklabels=10, yticklabels=10, cmap=myPalette)
    axes[0].set_ylabel(r'$mass^{NN} \rm{[bin ~number]}$');
    axes[0].set_xlabel(r'$mass^{GEN} \rm{[bin ~number]}$');
    axes[0].grid()
    axes[0].set_ylim([0,massMax])
    axes[0].set_xlim([0,massMax])
    axes[0].set_aspect(aspect='equal')
    axes[0].set_title("NN")
    
    cm = tf.math.confusion_matrix(mass2Label(df["genMass"]), mass2Label(df["fastMTT"]))
    cm = tf.cast(cm, dtype=tf.float32)
    cm = tf.math.divide_no_nan(cm, tf.math.reduce_sum(cm, axis=1)[:, np.newaxis])
    cm = tf.transpose(cm)
    sns.heatmap(cm, ax = axes[1], vmax = vmax, annot=False, xticklabels=10, yticklabels=10, cmap=myPalette)
    axes[1].grid()
    axes[1].set_title("fastMTT")
    axes[1].set_xlim([0,massMax])
    axes[1].set_ylim([0,massMax])
    axes[1].set_aspect(aspect='equal')
    axes[1].set_ylabel(r'$mass^{fastMTT} \rm{[bin ~number]}$')
    axes[1].set_xlabel(r'$mass^{GEN} \rm{[bin ~number]}$') 
           
    plt.subplots_adjust(bottom=0.15, left=0.05, right=0.95, wspace=0.25, hspace=0.35)
    plt.savefig("fig_png/CM.png", bbox_inches="tight")
###################################################
###################################################
def compareDYandH125(df):
        
    df_Z90 = df[(df["genMass"]>88) & (df["genMass"]<92)] 
    df_H125 = df[(df["genMass"]>123) & (df["genMass"]<127)]
        
    pull_NN_Z90 = (df_Z90["NN"] - df_Z90["genMass"])/df_Z90["genMass"]
    pull_NN_H125 = (df_H125["NN"] - df_H125["genMass"])/df_H125["genMass"]
      
    pull_fastMTT_Z90 = (df_Z90["fastMTT"] - df_Z90["genMass"])/df_Z90["genMass"]
    pull_fastMTT_H125 = (df_H125["fastMTT"] - df_H125["genMass"])/df_H125["genMass"]
    
    print("fastMTT:")
    print("Mass range: Z90",
          "mean pull: {0:3.3f}".format(np.mean(pull_fastMTT_Z90)),
          "pull RMS: {0:3.3f} RMS/90: {1:3.4f}".format(np.std(pull_fastMTT_Z90, ddof=1), np.std(pull_fastMTT_Z90, ddof=1)/90.0)
         )
    print("Mass range: H125",
          "mean pull: {0:3.3f}".format(np.mean(pull_fastMTT_H125)),
          "pull RMS: {0:3.3f} RMS/125: {1:3.4f}".format(np.std(pull_fastMTT_H125, ddof=1), np.std(pull_fastMTT_H125, ddof=1)/125.0)
         )  
    print("NN:")
    print("Mass range: Z90",
          "mean pull: {0:3.3f}".format(np.mean(pull_NN_Z90)),
          "pull RMS: {0:3.3f} RMS/90: {1:3.4f}".format(np.std(pull_NN_Z90, ddof=1), np.std(pull_NN_Z90, ddof=1)/90.0)
         )
    print("Mass range: H125",
          "mean pull: {0:3.3f}".format(np.mean(pull_NN_H125)),
          "pull RMS: {0:3.3f} RMS/125: {1:3.4f}".format(np.std(pull_NN_H125, ddof=1), np.std(pull_NN_H125, ddof=1)/125.0)
         )
     
    minX = 0
    maxX = 200
    #maxY = 0.5*np.maximum(df_Z90., _H125.shape[0])
    nBins = 40
    fig, axes = plt.subplots(1, 3, figsize = (15, 5))  
    axes[0].hist(df_Z90["NN"], range=(minX, maxX), bins = nBins, color="deepskyblue", label = "m=90")
    axes[0].hist(df_H125["NN"], range=(minX, maxX), bins = nBins, color="tomato", label="m=125")
    axes[0].set_xlabel("Mass")
    axes[0].set_title("NN")
    axes[0].legend(loc='upper right')
    axes[0].set_xlim([minX, maxX]) 
    axes[0].minorticks_on()
    #axes[0].set_ylim([0, maxY])  
    
    axes[1].hist(df_Z90["fastMTT"], range=(minX, maxX), bins = nBins, color="deepskyblue", label = "m=90")
    axes[1].hist(df_H125["fastMTT"], range=(minX, maxX), bins = nBins, color="tomato", label="m=125")
    axes[1].set_xlabel("Mass")
    axes[1].set_title("fastMTT")
    axes[1].legend(loc='upper right')
    axes[1].set_xlim([minX, maxX]) 
    axes[1].minorticks_on()
    #axes[1].set_ylim([0, maxY])
    
    scores = np.concatenate((df_Z90["fastMTT"],df_H125["fastMTT"]))
    labels_S = np.ones(df_H125["fastMTT"].shape)
    labels_B = np.zeros(df_Z90["fastMTT"].shape)
    labels_S_B = np.concatenate((labels_B, labels_S))
    fpr_fastMTT, tpr_fastMTT, thresholds_fastMTT = roc_curve(labels_S_B, scores, pos_label=1) 
    
    scores = np.concatenate((df_Z90["NN"],df_H125["NN"]))
    labels_S = np.ones(df_H125["NN"].shape)
    labels_B = np.zeros(df_Z90["NN"].shape)
    labels_S_B = np.concatenate((labels_B, labels_S))
    fpr_NN, tpr_NN, thresholds_NN = roc_curve(labels_S_B, scores, pos_label=1) 
     
    axes[2].plot(tpr_NN, fpr_NN, label='NN')
    axes[2].plot(tpr_fastMTT, fpr_fastMTT, label='fastMTT')
    for x, y, txt in zip(tpr_NN[::1], fpr_NN[::1], thresholds_NN[::1]):
        if x>0.9 and x<0.995:
            axes[2].annotate(np.round(txt,2), (x, y-0.04), color="brown")
            
    step = 10        
    for x, y, txt in zip(tpr_fastMTT[::step], fpr_fastMTT[::step], thresholds_fastMTT[::step]):
        if x>0.9 and x<0.995:
            axes[2].annotate(np.round(txt,2), (x, y+0.04), color="blue")       
    
    axes[2].set_xlim(0.90,1.0)
    #axes[2].set_ylim(0.0,1.0)
    axes[2].set_xlabel('True positive rate')
    axes[2].set_ylabel('False positive rate')
    axes[2].set_title('ROC curve')
    axes[2].legend(loc='best')
    
    plt.subplots_adjust(bottom=0.15, left=0.05, right=0.95, wspace=0.4, hspace=0.35)
    plt.savefig("fig_png/ROC.png", bbox_inches="tight")
###################################################
###################################################