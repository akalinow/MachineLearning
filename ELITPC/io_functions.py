import tensorflow as tf
import numpy as np
from scipy import ndimage
################################################
################################################
featuresShape = (512, 92, 3)
crop_shape = (64, 64, 1)
labelsShape = (2,)
################################################
################################################
feature_description = {
    'UVW_data': tf.io.FixedLenFeature(featuresShape, tf.float32),
    'label': tf.io.FixedLenFeature(labelsShape, tf.int64),
}
################################################
################################################
def _parse_function(example_proto):
  return tf.io.parse_single_example(example_proto, feature_description)
################################################
################################################
def readTFRecordFile(fileNames):
    raw_dataset = tf.data.TFRecordDataset(fileNames)
    return raw_dataset.map(_parse_function)
################################################
################################################
def preprocessProjections(item, projection=0):
    x = item["UVW_data"]
    x = tf.reshape(x[:,:,projection], (512,92,1))
    x = tf.pad(x, ((0,0),(0,36),(0,0)))
    return x
################################################
################################################
def find_ROIs(data, thr=0.2, size_thr=10):
    
    data = data[:,:,0]
    s = ndimage.generate_binary_structure(2,2) #(2,1)
    x = tf.math.greater(data, tf.constant(thr))
    x = ndimage.binary_fill_holes(x)
    x = ndimage.binary_opening(x, structure=s)
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
################################################
################################################
def crop_ROI(data, roi, shape):

    if roi==None:
        return np.zeros(crop_shape), np.zeros(crop_shape)
    sy, sx = roi['slice']
    mask = roi['mask']
    #print("ROI: ({},{}) - ({},{})".format(sx.start,sy.start, sx.stop,sy.stop))
    mask = tf.reshape(mask, mask.shape+(1,))
    cropped = data[sy.start:sy.stop, sx.start:sx.stop,:]
    cropped = tf.image.resize_with_crop_or_pad(cropped, shape[0], shape[1])
    mask = tf.image.resize_with_crop_or_pad(mask, shape[0], shape[1])
    return cropped, mask
################################################
################################################
def cropped_images_generator(dataset):
    params = {
        'thr': 0.2, #seed pixel magnitude 
        'size_thr': 10 #number of pixels in patch
        }
    
    for data in dataset:    
        rois = find_ROIs(data, **params)
        cropped, _ = crop_ROI(data, rois[0], shape=(64,64)) 
        yield cropped
################################################
def generateCircle(xx, yy, center_x, center_y):
    radius = tf.random.uniform(shape=[1], minval=10, maxval=32)
    circle_equation = tf.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    circle = tf.math.logical_and(circle_equation<radius, True)
    circle = tf.reshape(circle, crop_shape)
    return circle
################################################
def generateRectangle(xx, yy, center_x, center_y):
    a,b = tf.random.uniform(shape=[2], minval=10, maxval=50)
    rectangle = tf.math.logical_and(np.abs(xx - center_x)<a/2, np.abs(yy - center_y)<b/2)
    rectangle = tf.reshape(rectangle, crop_shape)
    return rectangle
################################################
def generateLine(xx, yy, center_x, center_y):
    phi = tf.random.uniform(shape=[1], minval=0, maxval=np.pi)
    tanPhi = tf.tan(phi)
    lenght = tf.random.uniform(shape=[1], minval=5, maxval=40)
    width = tf.random.uniform(shape=[1], minval=5, maxval=10)
    circle_equation = tf.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    line_equation = (yy-center_y) - tanPhi*(xx-center_x)
    rectangle = tf.math.logical_and(tf.abs(line_equation)<width, circle_equation<lenght)
    rectangle = tf.reshape(rectangle, crop_shape)
    return rectangle
################################################
def generateEmpty(center_x, center_y):
    return np.zeros(crop_shape)
################################################
def shapes_images_generator():
    xx, yy = tf.meshgrid(tf.range(0,crop_shape[0], dtype=tf.float32), tf.range(0,crop_shape[1], dtype=tf.float32))
    for number in range(0,32000):
        aRandom = tf.random.uniform(shape=[1], minval=0, maxval=1)    
        image = generateLine(xx, yy, center_x=32,center_y=32)
        yield image
################################################ 
def saveVAE(vae, tag):      
    path = "training/encoder_"+tag
    vae.encoder.save(path, save_format='tf')
    path = "training/decoder_"+tag
    vae.decoder.save(path, save_format='tf')
################################################
def loadEncoderDecoder(tag):   
    checkpoint_path = "training/encoder_"+tag
    encoder = tf.keras.models.load_model(filepath=checkpoint_path)
    checkpoint_path = "training/decoder_"+tag
    decoder = tf.keras.models.load_model(filepath=checkpoint_path)
    return encoder, decoder
################################################