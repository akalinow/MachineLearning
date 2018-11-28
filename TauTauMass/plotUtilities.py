import numpy as np
import matplotlib.pyplot as plt

#####################################################################
#####################################################################
def plotVariable(x, y, plotTitle, doBlock=True):

    print (x.shape, y.shape)
    fig, (axis0) = plt.subplots(nrows=1, ncols=1, figsize=(9, 4), num = plotTitle)
    axis0.hist2d(x, y, bins=[50,50], range = [[50, 250], [0.8, 1.2]])
    axis0.set_title("")
    axis0.set_xlabel("Target")
    axis0.set_ylabel("$NN/m_{vis}$")

    plt.show(block=doBlock)
#####################################################################
#####################################################################
def plotDiscriminant(modelResult, labels, plotTitle, doBlock=True):

    fig, (axis0, axis1) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), num = plotTitle)
    
    x = np.array(modelResult)
    y = np.array(labels)
    x = np.reshape(x,(-1))
    y = np.reshape(y,(-1))
    axis0.hist2d(x, y, bins=40, cmap = plt.cm.rainbow)
    #axis0.scatter(modelResult, labels, c="b")
    axis0.set_title("")
    axis0.set_xlabel("Prediction")
    axis0.set_ylabel("Target")
    '''
    error = (modelResult - labels)/labels    
    axis1.hist(error, bins=60, range=(-2,2))    
    axis1.set_xlabel("(Prediction - Target)/Target")
    axis1.set_title("Pull")
    '''

    xMax = 250
    nBins = (int)(xMax/5)
    aHist = axis1.hist(modelResult, bins=nBins, range=(0,xMax))
    #axis1.set_xlim((0, 250))
    axis1.set_xlabel("Prediction")
    axis1.set_title("")
    
    
    plt.show(block=doBlock)
#####################################################################
#####################################################################
