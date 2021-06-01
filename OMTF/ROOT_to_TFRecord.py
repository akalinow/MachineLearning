#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import random
import pathlib
import glob
import time
import numpy as np
import ROOT
import pandas as pd
from root_pandas import read_root
from root_numpy import root2array

nLayers = 18


# ## Definitions of functions used in this notebook.

# In[3]:


columns = np.array(['muonPt', 'muonEta', 'muonPhi', 'muonCharge', 'omtfPt', 'omtfEta',
       'omtfPhi', 'omtfCharge', 'omtfScore', 'omtfQuality', 'omtfRefLayer',
       'omtfProcessor', 'omtfFiredLayers', 'phiDist_0', 'phiDist_1',
       'phiDist_2', 'phiDist_3', 'phiDist_4', 'phiDist_5', 'phiDist_6',
       'phiDist_7', 'phiDist_8', 'phiDist_9', 'phiDist_10', 'phiDist_11',
       'phiDist_12', 'phiDist_13', 'phiDist_14', 'phiDist_15', 'phiDist_16',
       'phiDist_17'])

def decodeUnion(raw, unionFormat):     
    rawData = int(raw)
    layer    = rawData &               0xff 
    quality = (rawData &             0xff00) >> 8
    z       = (rawData &           0xff0000) >> 16 
    eta   = 0
    valid = 1
    phiDist = 0     
    if unionFormat=="new":
        valid   = (rawData &         0xff000000) >> 24
        eta     = (rawData &     0xffff00000000) >> 32
        phiDist = (rawData & 0xffff000000000000) >> 48
    else:
        eta   = (rawData &         0xff000000) >> 24
        phiDist     = (rawData &     0xffff00000000) >> 32 
    if phiDist>=2**15 -1:
        phiDist -= 2**16
    return np.array([layer, quality, z, valid, eta, phiDist], dtype=np.int16)

def decodeHits(hits, unionFormat):
    phiDistArray = np.full(nLayers, 9999, dtype=np.int16)
    for aHit in hits:
        decodedUnion = decodeUnion(aHit, unionFormat)
        np.put(phiDistArray, decodedUnion[0], decodedUnion[5])    
    return phiDistArray  

def transformColumns(df, unionFormat):
    df["omtfFiredLayers"] = df["omtfFiredLayers"].transform(lambda x: np.binary_repr(x,18).count("1"))
    columnNames = ["phiDist_{}".format(iLayer) for iLayer in range(0, nLayers)]                         
    for iLayer in range(0, nLayers):
            df["phiDist_{}".format(iLayer)] = df["hits"].transform(lambda x: decodeHits(x, unionFormat)[iLayer]).astype('int16',copy=False)

def loadDatasetFromParquet(parquetFile):
    df = pd.read_parquet(parquetFile)
    df = df.drop(columns="hits")
    df = df.sample(frac=1.0)
    df.info(memory_usage='deep')
    dataset = tf.data.Dataset.from_tensor_slices(df.values)
    return dataset

def saveDatasetToTFRecord(dataset, fileName):  
    dataset = dataset.map(tf.io.serialize_tensor)
    writer = tf.data.experimental.TFRecordWriter(fileName, compression_type="GZIP")
    writer.write(dataset)
    
def parse_tensor(tensor):
    return tf.io.parse_tensor(tensor, out_type=tf.float64)  

def benchmark(dataset, num_epochs=1):
    start_time = time.perf_counter()
    count = 0
    for epoch_num in range(num_epochs):
        for sample in dataset:
            count+=sample.shape[0]
            # Performing a training step
            time.sleep(1E-10)
    tf.print("Number of examples: ",count)       
    tf.print("Execution time:", time.perf_counter() - start_time) 
    
def convertROOT_2_Parquet_2_TFRecord(fileNames):
    for fileName in fileNames: 
        print("Processing file:",fileName)
        label = fileName.split("/")[-1].split(".")[0]
        label = label.lstrip("omtfHits_omtfAlgo0x0006_v1")
        path = str(pathlib.Path(fileName).parent)
        path = path.rstrip("omtfHits_omtfAlgo0x0006_v1")
        path = path.replace("ROOT","Python/")
        for iChunk, dfChunk in enumerate(read_root(fileName, chunksize=int(15E6))):
            print("\tProcessing chunk: {}".format(iChunk))
            transformColumns(dfChunk, unionFormat="new")  
            parquetFile = path+'df.parquet_{}_chunk_{}.gzip'.format(label, iChunk)
            dfChunk.to_parquet(parquetFile, compression='gzip')
            dataset = loadDatasetFromParquet(parquetFile)
            dataset = dataset.map(tf.io.serialize_tensor)
            tfrecordFileName = path+'{}_chunk_{}.tfrecord.gzip'.format(label,iChunk)
            writer = tf.data.experimental.TFRecordWriter(tfrecordFileName, compression_type="GZIP")
            writer.write(dataset)
            print("Chunk done.")
            break
        print("File done.")
         
def convertParquet_2_TFRecord(fileNames, isTrain, doFilter):
    for parquetFile in fileNames: 
        print("Processing file:",parquetFile)
        label = parquetFile.split("/")[-1]
        label = label.lstrip("df.parquet_")
        label = label.rstrip(".gzip")
        path = str(pathlib.Path(parquetFile).parent)+"/"
        dataset = loadDatasetFromParquet(parquetFile)
        if doFilter:
            print("Filtering.")
            dataset = filterDataset(dataset, isTrain)
            label = label+"_filtered"
            path = path+"/filtered/"
        tfrecordFileName = path+label+'.tfrecord.gzip'
        print("Saving to TFRecord file.")
        saveDatasetToTFRecord(dataset, tfrecordFileName) 
        print("File done.")
        
def test(fileNames, isTrain, doFilter):
    parquetFile = fileNames[0]
    label = parquetFile.split("/")[-1]
    label = label.lstrip("omtfHits_omtfAlgo0x0006_v1df.parquet_")
    label = label.rstrip(".gzip")
    path = str(pathlib.Path(parquetFile).parent)+"/"
    if doFilter:
        label = label+"_filtered"
        path = path+"/filtered/"
    tfrecordFileName = path+'{}.tfrecord_TEST.gzip'.format(label)
    writer = tf.data.experimental.TFRecordWriter(tfrecordFileName, compression_type="GZIP")
       
    for parquetFile in fileNames: 
        print("Processing file:",parquetFile)
        print("Loading parquet file.")
        dataset = loadDatasetFromParquet(parquetFile).take(10)
        if doFilter:
            print("Filtering.")
            dataset = filterDataset(dataset, isTrain)
        dataset = dataset.map(tf.io.serialize_tensor)
        for item in dataset:
            writer.write(item)
        print("File done.")        
        
                
def filterDataset(dataset, isTrain):      
    #Select positive muons (has to be done before batching)
    #columnIndex = np.where(columns == "omtfFiredLayers")[0][0]
    #dataset = dataset.filter(lambda x: x[columnIndex] != 1027)  
    #Select muon with OMTF quality==12
    #columnIndex = np.where(columns == "omtfQuality")[0][0]       
    #dataset = dataset.filter(lambda x: x[columnIndex]>=12)
    if isTrain:
        #Select muon basing on generated pT
        columnIndex = np.where(columns == "muonPt")[0][0]
        dataset = dataset.filter(lambda x: x[columnIndex]<100)  
    return dataset       


# ##  Import ROOT files into Pandas DataFrame, and save into a parquet format, then transform into TFRecord
# This step should be executed only once. Later the data should be read from parquet or TFRecord files.

# In[4]:


fileNames = glob.glob('/home/user1/scratch/akalinow/CMS/OverlapTrackFinder/Python/omtfHits/ROOT/omtfHits_omtfAlgo0x0006_v1/*.root')
#fileNames = glob.glob('/home/user1/scratch/akalinow/CMS/OverlapTrackFinder/Python/omtfHits/ROOT/omtfHits_omtfAlgo0x0006_v1/*oldSample_files_*.root')

print(fileNames)

convertROOT_2_Parquet_2_TFRecord(fileNames)        


# ## Write TFRecord from parquet files

# In[ ]:


fileNames = glob.glob('/home/user1/scratch/akalinow/CMS/OverlapTrackFinder/Python/omtfHits/Python/omtfHits_omtfAlgo0x0006_v1/df.parquet_*.gzip')
fileNames = glob.glob('/home/user1/scratch/akalinow/CMS/OverlapTrackFinder/Python/omtfHits/Python/omtfHits_omtfAlgo0x0006_v1/*df.parquet_*oldSample_files_1_10*.gzip')

#convertParquet_2_TFRecord(fileNames, isTrain = False, doFilter = True)       
#test(fileNames, isTrain = False, doFilter = False)     


# ## Test: read Pandas df from parquet file.

# In[ ]:


fileNames = glob.glob('/home/user1/scratch/akalinow/CMS/OverlapTrackFinder/Python/omtfHits/Python/df.parquet_OMTFHits_pats0x0003_oldSample_files_35_45_chunk_*.gzip')
#df = pd.read_parquet(fileNames[0])
#print(df)


# ## Test: read TFRecord from TFRecord file.

# In[ ]:


fileNames = glob.glob('/home/user1/scratch/akalinow/CMS/OverlapTrackFinder/Python/omtfHits/Python/omtfHits_omtfAlgo0x0006_v1/OMTFHits_pats0x0003_oldSample_files_1_10_chunk_6.tfrecord_TEST.gzip')

raw_dataset = tf.data.TFRecordDataset(fileNames, compression_type="GZIP")
dataset = raw_dataset.map(parse_tensor, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.batch(10000, drop_remainder=False)

#benchmark(raw_dataset.map(parse_tensor,num_parallel_calls=tf.data.experimental.AUTOTUNE))
#benchmark(raw_dataset.map(parse_tensor))
benchmark(dataset)
#benchmark(dataset.prefetch(tf.data.experimental.AUTOTUNE))
#benchmark(tf.data.Dataset.range(2).interleave(dataset))

#for element in dataset.take(1): 
#  print(element)


# In[ ]:




