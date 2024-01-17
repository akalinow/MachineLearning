import tensorflow as tf
import pandas as pd
import seaborn as sns
import pandas as pd
import numpy as np
import scipy
from scipy import ndimage
from skimage import io, transform
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import utility_functions as utils


params = {'legend.fontsize': 'xx-large',
          'figure.figsize': (14, 10),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large',
         #'xticks':'major_ticks_top'
         }

plt.rcParams.update(params)


def plotEndpoints(data, iProj, axis, label, color):

        scale = 100
        uvwt =  utils.XYZtoUVWT(scale*data[0:3])
        axis.plot(uvwt[3], uvwt[iProj], marker='.', markersize=20, alpha=0.8, color=color, label=label)
        
        uvwt =  utils.XYZtoUVWT(scale*data[3:6])
        axis.plot(uvwt[3], uvwt[iProj], marker='.', markersize=20, alpha=0.8, color=color)
        
        uvwt =  utils.XYZtoUVWT(scale*data[6:9])
        axis.plot(uvwt[3], uvwt[iProj], marker='.', markersize=20, alpha=0.8, color=color)


def plotEvent(data, model, axlim = False, uvwt_mode=False):
    #data indexing: data[features/label][element in batch][index in features/label]
    projNames = ("U", "V", "W")
    fig, axes = plt.subplots(1,3, figsize=(28,10))
    if uvwt_mode:
      data = utils.batch_to_xyz(data)

    iEvent = 0
    projections = data[0]
    labels = data[1]
    
    for iProj in range(0,3):
        axis = axes[iProj] 
        data = projections[iEvent][:,:,iProj]
                
        im = axis.imshow(data, origin='lower', aspect='auto')            
        plotEndpoints(labels[iEvent], iProj, axis, color="red", label="true")         
        
        rois = find_ROIs(data, thr=0.1, size_thr=10)
        #plot_ROIs(rois, axis)
        sy, sx = rois[0]['slice']
        
        if model!=None:
          if uvwt_mode:
            projections_uvwt = np.transpose(projections, (3, 1, 2, 0))
            modelResponse = utils.UVWTtoXYZ(model(projections_uvwt))[iEvent]
          else:
            modelResponse = model(projections)[iEvent]
          plotEndpoints(modelResponse, iProj, axis, color="blue", label="NN")          
        axis.set_xlabel("time bin")
        axis.set_ylabel(projNames[iProj]+" strip")
        if axlim:
            axis.set_xlim(axlim[iProj][0][0], axlim[iProj][0][1])
            axis.set_ylim(axlim[iProj][1][0], axlim[iProj][1][1])
        else:
            axis.set_xlim(sx.start-5, sx.stop+5)
            axis.set_ylim(sy.start-5, sy.stop+5)
        axis.legend()
        
        divider = make_axes_locatable(axis)
        cax = divider.append_axes("right", size="5%", pad=0.4)
        fig.colorbar(im, cax=cax)
        
        plt.subplots_adjust(bottom=0.15, left=0.05, right=0.95, wspace=0.3)
        plt.savefig("fig_png/event.png", bbox_inches="tight")


def find_ROIs(img, thr=10, size_thr=0):
    s = ndimage.generate_binary_structure(2,2) #(2,1)
    x = tf.math.greater(img, thr)
    x = ndimage.binary_fill_holes(x)
    #x = ndimage.binary_opening(x, structure=s)
    x = ndimage.binary_dilation(x, iterations=3)
    labels, nl = ndimage.label(x,structure=s)
    objects_slices = ndimage.find_objects(labels)
    masks = [labels[obj_slice] == idx for idx, obj_slice in enumerate(objects_slices, start=1)]
    sizes = [mask.sum() for mask in masks]
    
    result = [{'idx': idx, 'slice': s, 'mask': m, 'size': size} for
               idx, (s, m, size) in enumerate(zip(objects_slices, masks, sizes), start=1) \
                   if size>size_thr
               ] 
    result = sorted(result,key=lambda x: x["size"], reverse=True)

    return result


def plot_ROIs(rois, axis):
     import matplotlib.transforms as transforms
     for roi in rois:
        sy, sx = roi['slice']
        #print("ROI: ({},{}) - ({},{})".format(sx.start,sy.start, sx.stop,sy.stop))
        
        scalex = 1.0
        scaley = 1.0
        
        width = (sx.stop-sx.start)*scalex
        height = (sy.stop-sy.start)*scaley
        (startx,starty) = (sx.start*scalex,sy.start*scaley)
    
        axis.add_patch(Rectangle((startx,starty), width, height,
                                 linewidth=1, edgecolor='r',facecolor='none'
                                 ))
        break #plot only the first ROI

 
def crop_ROIs(image, rois):
     width = 64
     threshold = 0.05   
        
     for roi in rois:
        sy, sx = roi['slice']
        mask = roi['mask']
        print("ROI: ({},{}) - ({},{})".format(sx.start,sy.start, sx.stop,sy.stop))
        cropped = image[sy.start:sy.stop, sx.start:sx.stop]
        cropped = cropped[0:width, 0:width]
        cropped = mask[0:width, 0:width]
        xPad = tf.cast((width-cropped.shape[0])/2, tf.int32)
        yPad = tf.cast((width-cropped.shape[1])/2, tf.int32)
        cropped = tf.pad(cropped, ((xPad,xPad),(yPad,yPad)))
        xPad = tf.cast((width-cropped.shape[0]), tf.int32)
        yPad = tf.cast((width-cropped.shape[1]), tf.int32)
        cropped = tf.pad(cropped, ((xPad,0),(yPad,0)))
        return cropped
     return np.zeros((width,width))

  
def cropROI(item):
    
    params = {
        'thr': 0.2, #seed pixel magnitude 
        'size_thr': 10 #number of pixels in patch
        }
    image = item[:,:,0]
    rois = find_ROIs(image, **params)
    cropped = crop_ROIs(image, rois)
    return cropped


def plotTrainHistory(history):
    
    fig, axes = plt.subplots(1,2, figsize=(10,3))
    axes[0].plot(history.history['loss'], label = 'train')
    axes[0].plot(history.history['val_loss'], label = 'val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss function')
    axes[0].legend(loc='upper right')
    
    axes[1].plot(history.history['loss'], label = 'train')
    axes[1].plot(history.history['val_loss'], label = 'val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss function')
    axes[1].legend(loc='upper right')
    axes[1].set_yscale('log')
    
    plt.subplots_adjust(bottom=0.02, left=0.02, right=0.98, hspace=0.5)
    plt.savefig("fig_png/training_history.png", bbox_inches="tight")


def plotLengthPull(df, partIdx):
    
    d_GEN = np.sqrt((df["GEN_StopPosX_Part"+str(partIdx)] - df["GEN_StartPosX"])**2 + 
                    (df["GEN_StopPosY_Part"+str(partIdx)] - df["GEN_StartPosY"])**2 + 
                    (df["GEN_StopPosZ_Part"+str(partIdx)] - df["GEN_StartPosZ"])**2)
    
    d_RECO = np.sqrt((df["RECO_StopPosX_Part"+str(partIdx)] - df["RECO_StartPosX"])**2 + 
                    (df["RECO_StopPosY_Part"+str(partIdx)] - df["RECO_StartPosY"])**2 + 
                    (df["RECO_StopPosZ_Part"+str(partIdx)] - df["RECO_StartPosZ"])**2)          
    
    pull = (d_RECO-d_GEN)
    df["d_GEN_part"+str(partIdx)] = d_GEN
    df["d_RECO_part"+str(partIdx)] = d_RECO
    df["pull_part"+str(partIdx)] = pull
    
    mean = pull.mean()
    std = pull.std()

    fig, axes = plt.subplots(3,2, figsize=(10,10))
    label = "$\mu = {:.3f}$\n$\sigma = {:.2f}$".format(mean, std)
    axes[0,0].hist(pull, bins=40, label=label);
    axes[0,0].set_xlabel("RECO-GEN [mm]")

    axes[0,1].hist(pull, bins=40, label=label);
    axes[0,1].set_xlabel("RECO-GEN [mm]")
    axes[0,1].set_yscale('log')
    axes[0,1].legend(bbox_to_anchor=(1.1,1), loc='upper left')
    
    xBins = np.linspace(0,80,40)
    yBins = np.linspace(-5,5,20)
    axes[1,0].hist2d(d_GEN, pull, bins=(xBins, yBins), cmin=10, label="length")
    axes[1,0].set_xlabel('particle range [mm]')
    axes[1,0].set_ylabel('RECO-GEN')
    
    yBins = np.linspace(-0.5,0.5,20)
    axes[1,1].hist2d(d_GEN, pull/d_GEN, bins=(xBins, yBins), cmin=10, label="length")
    axes[1,1].set_xlabel(' particle range')
    axes[1,1].set_ylabel('(RECO-GEN)/GEN')
    
    axes[2,0].plot(d_GEN, d_RECO, "bo")
    axes[2,0].plot((d_GEN.min(), d_GEN.max()), (d_RECO.min(), d_RECO.max()), color="black")
    axes[2,0].set_xlabel('range GEN [mm]')
    axes[2,0].set_ylabel('range RECO [mm]')
    
    axes[2,1].plot(d_GEN, d_RECO, "bo")
    axes[2,1].plot((d_GEN.min(), d_GEN.max()), (d_RECO.min(), d_RECO.max()), color="black")
    axes[2,1].set_xlabel('range GEN [mm]')
    axes[2,1].set_ylabel('range RECO [mm]')
    axes[2,1].set_xlim((0,20))
    axes[2,1].set_ylim((0,20))
       
    fig.suptitle("particle "+str(partIdx)+" track length resolution")

    plt.subplots_adjust(bottom=0.05, left=0.05, right=0.95, hspace=0.3, wspace=0.3) 
    plt.savefig("fig_png/length_pull_part_"+str(partIdx)+".png", bbox_inches="tight")


def plotLengthPullEvolution(df):
    
    fig, axes = plt.subplots(3,1, figsize=(6,9))

    partIdx = 1
    binWidth = 1 
    bins = np.linspace(1,100,100)
    for partIdx in range(1,3):
        
        label = ""
        if partIdx==1:
            label = r"$\alpha$"
        elif partIdx==2:
            label = r"$^{12}_{6}$C"
        
        axes[0].hist(df["d_GEN_part"+str(partIdx)], bins=bins, density=True, label=label+" GEN")
        axes[0].hist(df["d_RECO_part"+str(partIdx)], bins=bins, density=True, label=label+" RECO", alpha=0.6)
        axes[0].set_xlabel("length [mm]")
        axes[0].set_ylabel("#events")
        axes[0].set_xlim(-5,100)
        axes[0].legend(bbox_to_anchor=(1.1,1), loc='upper left')
        
        df_grouped = df.groupby(by=binWidth*(df["d_GEN_part"+str(partIdx)]/binWidth).astype(int))
        x = df_grouped["d_GEN_part"+str(partIdx)].mean()
        y = df_grouped["pull_part"+str(partIdx)].mean()

        axes[1].plot(x, y, ".", label=label)
        axes[1].plot((x.min(), x.max()), (0,0), color='black')
        axes[1].set_xlabel("GEN length [mm]")
        axes[1].set_ylabel("RECO-GEN [mm]")
        axes[1].set_xlim(-5,100)
        axes[1].set_ylim(-5,5)
        
        df_grouped = df.groupby(by=binWidth*(df["GEN_StartPosX"]/binWidth).astype(int))
        x = df_grouped["GEN_StartPosX"].mean()
        y = df_grouped["pull_part"+str(partIdx)].mean()
        axes[2].set_xlabel("GEN vertex X [mm]")
        axes[2].set_ylabel("RECO-GEN [mm]")
        axes[2].plot(x, y, ".", label=label)
        axes[2].plot((x.min(), x.max()), (0,0), color='black')
    
    plt.subplots_adjust(bottom=0.05, left=0.05, right=0.95, hspace=0.5, wspace=0.3) 
    plt.savefig("fig_png/length_pull_vs_gen.png", bbox_inches="tight")


def plotEndPointRes(df, edge, partIdx):
    
    fig, axes = plt.subplots(2,2, figsize=(8,8))

    for index, coordName in enumerate(["X", "Y", "Z"]):
            axis = axes.flatten()[index]  
            varName1 = "GEN_"+edge+"Pos"+coordName
            varName2 = "RECO_"+edge+"Pos"+coordName
            if edge=="Stop":
                varName1+="_Part"+str(partIdx)
                varName2+="_Part"+str(partIdx)
            mean = (df[varName2] - df[varName1]).mean()
            std = (df[varName2] - df[varName1]).std()
            label = "$\mu_{} = {:.3f}$\n$\sigma_{} = {:.2f}$".format(coordName, mean, coordName, std)
            (df[varName2] - df[varName1]).hist(ax=axis, bins=40, label=label)
            axis.set_xlabel(coordName+" [mm]")
            axis.set_ylabel("")  
            axis.grid(False)
            axis.legend()
            if coordName=="X":
                axis.legend(bbox_to_anchor=(1.5,-0.3), loc='upper left')
            elif coordName=="Y":
                axis.legend(bbox_to_anchor=(0.2,-0.6), loc='upper left')
            elif coordName=="Z":
                axis.legend(bbox_to_anchor=(1.5, 0.4), loc='upper left')   
                

    fig.suptitle("track "+str(partIdx)+" "+edge+" resolution")       
    axes[1,1].set_visible(False)        
    plt.subplots_adjust(bottom=0.05, left=0.05, right=0.95, hspace=0.3, wspace=0.3) 
    plt.savefig("fig_png/"+edge+"_endpoint_resolution_part_"+str(partIdx)+".png", bbox_inches="tight")          

   
def controlPlots(df):
    
    fig, axes = plt.subplots(2,2, figsize=(8,8))

    for index, coordName in enumerate(["X", "Y", "Z"]):
            axis = axes.flatten()[index]  
            varName = "GEN_StartPos"+coordName
            df.hist(varName, ax=axis, bins=40)
            axis.set_xlabel(coordName)
            axis.set_ylabel("")  
            axis.grid(False)

    axes[1,1].set_visible(False)    

    plt.subplots_adjust(bottom=0.05, left=0.05, right=0.95, hspace=0.3, wspace=0.3)  
    plt.savefig("fig_png/gen_startPos.png", bbox_inches="tight")
    
    fig, axes = plt.subplots(2,2, figsize=(8,8))
    for index, coordName in enumerate(["X", "Y", "Z"]):
            axis = axes.flatten()[index]  
            varName = "GEN_StopPos"+coordName
            df.hist(varName, ax=axis, bins=40)
            axis.set_xlabel(coordName)
            axis.set_ylabel("")
            axis.grid(False)

    fig.suptitle("GEN track "+edge+" position")       
    axes[1,1].set_visible(False)    

    plt.subplots_adjust(bottom=0.05, left=0.05, right=0.95, hspace=0.3, wspace=0.3)   
    plt.savefig("fig_png/gen_endPos.png", bbox_inches="tight")


def plotOpeningAngleCos(df):
    
    fig, axes = plt.subplots(1,2, figsize=(10,5))

    GEN_cosAlpha = utils.getOpeningAngleCos(df, algoType="GEN")
    RECO_cosAlpha = utils.getOpeningAngleCos(df, algoType="RECO")

    axes[0].hist(RECO_cosAlpha, bins=np.linspace(-1, -0.95, 40), alpha=0.5, label="NN");
    axes[0].hist(GEN_cosAlpha, bins=np.linspace(-1, -0.95, 40), alpha=0.8, label="true");
    axes[0].set_xlabel(r'$cos(\alpha)$')
    axes[0].legend()

    mean = ((GEN_cosAlpha-RECO_cosAlpha)/(-1-GEN_cosAlpha)).mean()
    std = ((GEN_cosAlpha-RECO_cosAlpha)/(-1-GEN_cosAlpha)).std()
    label = "$\mu = {:.3f}$\n$\sigma = {:.2f}$".format(mean, std)

    axes[1].hist((GEN_cosAlpha-RECO_cosAlpha)/(-1-GEN_cosAlpha), bins=np.linspace(-2, 2, 40), label=label);
    axes[1].set_xlabel(r'$\frac{cos(\alpha^{RECO}) - cos(\alpha^{GEN})}{-1-cos(\alpha^{GEN})}$')
    axes[1].legend()