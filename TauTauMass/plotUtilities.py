import numpy as np
import matplotlib.pyplot as plt


def plotHistogram(xSurvive, xDied, xAll, name, nBins, normed, axisRange):

    # the histogram of the data
    fig = plt.figure()
    nPass, bins, patches = plt.hist(xSurvive, bins = nBins, range = axisRange, normed = normed, facecolor='g', alpha=0.75)
    nAll, bins, patches = plt.hist(xAll, bins = nBins, range = axisRange, normed = normed, facecolor='g', alpha=0.75)
    plt.close(fig)

    print(name)
    print("Pass",nPass)
    print("All",nAll)

    nPass = nPass/nAll
    nPass = np.nan_to_num(nPass)
    x = range(0,len(nPass))
    yerr = nPass*(1-nPass)/nAll
    yerr = np.nan_to_num(yerr)
    yerr = np.sqrt(yerr)

    fig = plt.figure(name)
    plt.errorbar(x, nPass, yerr=yerr, fmt='o')

    print("P: ",nPass)
    plt.title(name)
    plt.xlabel('Feature')
    plt.ylabel('Probability')
    plt.grid(True)
    plt.show(block=False)
#####################################################################
#####################################################################
#####################################################################
def plotVariable(x, y):

    print (x.shape, y.shape)

    y = np.broadcast_to(y,x.shape)

    survivedIndexes = y[:,0]==1.0
    diedIndexes = y[:,0]==0.0
    allIndexes = y[:,0]>=0.0

    survivedFeatures = x[survivedIndexes]
    diedFeatures = x[diedIndexes]
    allFeatures = x[allIndexes]

    featuresNames = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"]
    #featuresRanges = [(0,4), (-2,2), (0,100), (0,10), (0,10), (0,100), (-16,8), (0,5)]
    featuresRanges = [(-1,2), (-1,2), (-1,2), (-1,2), (-1,2), (-1,2), (-1,2), (-1,2)]
    nFeatures = 8

    for iFeature in range(0, nFeatures):
        plotHistogram(xSurvive = survivedFeatures[:,iFeature],
                      xDied = diedFeatures[:,iFeature],
                      xAll = allFeatures[:,iFeature],
                      name = featuresNames[iFeature],
                      nBins = 21, normed = 0,axisRange=featuresRanges[iFeature])


    plt.show(block=True)
#####################################################################
#####################################################################
def plotDiscriminant(modelResult, labels, plotTitle, doBlock=True):

    fig, (axis0, axis1) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), num = plotTitle)

    axis0.scatter(modelResult, labels, c="b")    
    axis0.set_title("")
    axis0.set_xlabel("Prediction")
    axis0.set_ylabel("Target")
    
    error = (modelResult - labels)/labels
    axis1.hist(error, bins=60, range=(-1,1))
    axis1.set_xlabel("(Prediction - Target)/Target")
    axis1.set_title("Pull")
    
    plt.show(block=doBlock)
#####################################################################
#####################################################################
