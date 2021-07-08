import tensorflow as tf
import numpy as np
from scipy import ndimage
import skimage
from skimage import feature
from skimage.draw import ellipse
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
    return x
################################################
################################################
def find_ROIs(data, threshold_abs=0.2, min_distance=0, roi_size_thr=10):
    
    data = np.reshape(data, featuresShape[0:2])
    data = feature.peak_local_max(data, min_distance=min_distance, indices=False, threshold_abs=threshold_abs)
    
    s = ndimage.generate_binary_structure(2,3)
    x = tf.math.greater(data, tf.constant(threshold_abs))
    x = ndimage.binary_fill_holes(x)
    #x = ndimage.binary_dilation(x, iterations=3)
    #x = ndimage.binary_opening(x, structure=s)
    #x = ndimage.grey_opening(x, structure=s)
    labels, nl = ndimage.label(x,structure=s)
    objects_slices = ndimage.find_objects(labels)
         
    masks = [labels[obj_slice] == idx for idx, obj_slice in enumerate(objects_slices, start=1)]
    sizes = [mask.sum() for mask in masks]
    
    result = [{'idx': idx, 'slice': s, 'mask': m, 'size': size} for
               idx, (s, m, size) in enumerate(zip(objects_slices, masks, sizes), start=1) \
                   if size>roi_size_thr
               ] 
    result = sorted(result,key=lambda x: x["size"], reverse=True)  
    
    return result
################################################
################################################
def crop_ROI(data, roi, shape):

    if roi==None:
        return np.zeros(crop_shape), np.zeros(crop_shape)
    row, column = roi['slice']
    mask = roi['mask']
    #print("ROI (x,y): ({},{}) - ({},{})".format(column.start,column.start, row.stop,row.stop))
    loc = roi['slice'] + (slice(0,1),)
    
    mask = tf.reshape(mask, mask.shape+(1,))
    cropped = data[loc]
    cropped = tf.image.resize_with_crop_or_pad(cropped, shape[0], shape[1])
    mask = tf.image.resize_with_crop_or_pad(mask, shape[0], shape[1])
    return cropped, mask
################################################
################################################
def cropped_images_generator(dataset, **params):
    for data in dataset:    
        rois = find_ROIs(data, **params)
        cropped, _ = crop_ROI(data, rois[0], shape=crop_shape)
        yield cropped
################################################
def generateCircle(center_x, center_y):
    xx, yy = tf.meshgrid(tf.range(0,crop_shape[0], dtype=tf.float32), tf.range(0,crop_shape[1], dtype=tf.float32))
    radius = tf.random.uniform(shape=[1], minval=10, maxval=32)
    circle_equation = tf.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    circle = tf.math.logical_and(circle_equation<radius, True)
    circle = tf.reshape(circle, crop_shape)
    return circle
################################################
def generateEllipse():    
    center_x, center_y = tf.random.uniform(shape=[2], minval=32-10, maxval=32+10)
    minorAxis =  tf.random.uniform(shape=[1], minval=1, maxval=5)
    majorAxis =  tf.random.uniform(shape=[1], minval=5, maxval=40)
    phi = tf.random.uniform(shape=[1], minval=0, maxval=np.pi)
    rr, cc = skimage.draw.ellipse(r=center_x, c=center_y, r_radius=minorAxis[0], c_radius=majorAxis[0], shape=crop_shape, rotation=phi[0])
    ellipse = np.zeros(crop_shape, dtype=np.uint8)
    ellipse[rr, cc] = 1
    return ellipse
################################################
def generateRectangle(center_x, center_y):
    xx, yy = tf.meshgrid(tf.range(0,crop_shape[0], dtype=tf.float32), tf.range(0,crop_shape[1], dtype=tf.float32))
    a,b = tf.random.uniform(shape=[2], minval=10, maxval=50)
    rectangle = tf.math.logical_and(np.abs(xx - center_x)<a/2, np.abs(yy - center_y)<b/2)
    rectangle = tf.reshape(rectangle, crop_shape)
    return rectangle
################################################
def generateLine():
    xx, yy = tf.meshgrid(tf.range(0,crop_shape[0], dtype=tf.float32), tf.range(0,crop_shape[1], dtype=tf.float32))
    phi = tf.random.uniform(shape=[1], minval=0, maxval=np.pi)
    tanPhi = tf.tan(phi)
    center_x, center_y = tf.random.uniform(shape=[2], minval=32-10, maxval=32+10)
    lenght = tf.random.uniform(shape=[1], minval=5, maxval=40)
    width = tf.random.uniform(shape=[1], minval=1, maxval=5)
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
    for number in range(0,32000):
        aRandom = tf.random.uniform(shape=[1], minval=0, maxval=1)    
        #image = generateLine()
        image = generateEllipse()
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

################################################