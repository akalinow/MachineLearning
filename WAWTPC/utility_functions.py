import os, glob

import tensorflow as tf
import pandas as pd
import numpy as np


df = pd.DataFrame(columns=["GEN_StartPosU", "GEN_StartPosV", "GEN_StartPosW", "GEN_StartPosT",
                           "GEN_StopPosU", "GEN_StopPosV", "GEN_StopPosW", "GEN_StopPosT",
                           "RECO_StartPosU", "RECO_StartPosV", "RECO_StartPosW", "RECO_StartPosT",
                           "RECO_StopPosU", "RECO_StopPosV", "RECO_StopPosW", "RECO_StopPosT",
                            ])

df = pd.DataFrame(columns=["GEN_StartPosX", "GEN_StartPosY", "GEN_StartPosZ",
                           "GEN_StopPosX_Part1", "GEN_StopPosY_Part1", "GEN_StopPosZ_Part1",
                           "GEN_StopPosX_Part2", "GEN_StopPosY_Part2", "GEN_StopPosZ_Part2",
                           #
                           "RECO_StartPosX", "RECO_StartPosY", "RECO_StartPosZ", 
                           "RECO_StopPosX_Part1", "RECO_StopPosY_Part1", "RECO_StopPosZ_Part1",
                           "RECO_StopPosX_Part2", "RECO_StopPosY_Part2", "RECO_StopPosZ_Part2",
                            ])

'''
df = pd.DataFrame(columns=["GEN_X", "GEN_Y", "GEN_Z",
                           "GEN_U", "GEN_V", "GEN_W", "GEN_T",
                           "RECO_U", "RECO_V", "RECO_W", "RECO_T"
                            ])  
'''                            
def fillPandasDataset(aBatch, df, model, uvwt_mode = False):   
    
    scale = 100
    
    if uvwt_mode:
      modelAnswer = UVWTtoXYZ(model(aBatch[0]))*scale
      features, labels = batch_to_xyz(aBatch)
      labels *= scale
    else:
      features = aBatch[0]
      labels = aBatch[1]*scale
      modelAnswer = model(features)*scale
    
    batch_df = pd.DataFrame(data=np.column_stack((labels,modelAnswer)),
                            columns = df.columns)
                               
    return pd.concat((df, batch_df), ignore_index=True).astype('float32')

def UVWTtoXYZ(data):
    referencePoint = np.array([-138.9971, 98.25])
    phi = np.pi/6.0
    stripPitch = 1.5
    f = 1.0/25*6.46
    data = data.reshape(3, 3, 2)

    # Reverse the scaling by stripPitch
    u = data[0, :, 1] * stripPitch
    v = data[1, :, 1] * stripPitch
    w = data[2, :, 1] * stripPitch
    t = data[0, :, 0]

    # Reverse the operations to compute u, v, w, t
    y = -u + 99.75
    x = (v + w - 98.75) / (2 * np.cos(phi)) + referencePoint[0]
    z = (t - 256) * f
    data_xyz = np.vstack([x, y, z]).T.flatten() / 100
    return data_xyz.reshape(1, -1)

def batch_to_xyz(batch):
  image, target = batch
  target_xyz = UVWTtoXYZ(target) 
  image = np.transpose(image, (3, 1, 2, 0))
  return (image, target_xyz)



def XYZtoUVWT(data):
    referencePoint = np.array([-138.9971, 98.25])
    phi = np.pi/6.0
    stripPitch = 1.5
    f = 1.0/25*6.46   
    u = -(data[1]-99.75)
    v = (data[0]-referencePoint[0])*np.cos(phi) - (data[1]-referencePoint[1])*np.sin(phi)
    w = (data[0]-referencePoint[0])*np.cos(-phi) - (data[1]-referencePoint[1])*np.sin(-phi) + 98.75
    t = data[2]/f + 256
    u/=stripPitch
    v/=stripPitch
    w/=stripPitch
    return np.array((u,v,w,t)).T


def getOpeningAngleCos(df, algoType):
    
    start = df[[algoType+"_StartPosX", algoType+"_StartPosY", algoType+"_StartPosZ"]].to_numpy()
    stop_part1 = df[[algoType+"_StopPosX_Part1", algoType+"_StopPosY_Part1", algoType+"_StopPosZ_Part1"]].to_numpy()
    stop_part2 = df[[algoType+"_StopPosX_Part2", algoType+"_StopPosY_Part2", algoType+"_StopPosZ_Part2"]].to_numpy()

    track1 = stop_part1-start
    norm = np.sqrt(np.sum(track1*track1, axis=1, keepdims=True))
    track1 /=norm

    track2 = stop_part2-start
    norm = np.sqrt(np.sum(track2*track2, axis=1, keepdims=True))
    track2 /=norm

    cosAlpha = np.sum(track1*track2, axis=1)
    return cosAlpha