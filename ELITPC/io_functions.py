import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from scipy import ndimage
import skimage
from skimage import feature
from skimage.draw import ellipse

from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_float
################################################
################################################
featuresShape = (512, 92, 3)
crop_shape = (128, 128, 1)
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
def denoise_and_mask(data):
    
    noisy = img_as_float(data)
    patch_kw = dict(patch_size=5,  # 5x5 patches
                patch_distance=6,  # 13x13 search area
                )
    # slow algorithm
    data = denoise_nl_means(noisy, fast_mode=False,
                               **patch_kw)
    threshold = 0.1
    mask = tf.greater(data, tf.constant(threshold))
    mask = tf.cast(mask,tf.float32)
    
    return mask
################################################
################################################
def find_ROIs(data, threshold_abs=0.2, min_distance=0, roi_size_thr=10):
    
    data = denoise_and_mask(data)
    
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
        return np.zeros(crop_shape), np.zeros(crop_shape), np.zeros(2)
    row, column = roi['slice']
    mask = roi['mask']
    #print("ROI (x,y): ({},{}) - ({},{})".format(column.start,column.stop, row.stop,row.stop))
    loc = roi['slice'] + (slice(0,1),)
    
    mask = tf.reshape(mask, mask.shape+(1,))
    cropped = data[loc]
    cropped = tf.image.resize_with_crop_or_pad(cropped, shape[0], shape[1])
    mask = tf.image.resize_with_crop_or_pad(mask, shape[0], shape[1])
      
    roi_centroid_col = column.start + tf.math.reduce_mean(tf.math.reduce_sum(cropped, axis=1))
    roi_centroid_row = row.start + tf.math.reduce_mean(tf.math.reduce_sum(cropped, axis=0))
    roi_centroid = (roi_centroid_row, roi_centroid_col)
    
    return cropped, mask, roi_centroid
################################################
################################################
def cropped_images_generator(dataset, **params):
    for data in dataset:  
        
        data = denoise_and_mask(data) #TEST
        
        rois = find_ROIs(data, **params)
        cropped, _, _ = crop_ROI(data, rois[0], shape=crop_shape)
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
    minorAxis =  tf.random.uniform(shape=[1], minval=1, maxval=7)
    majorAxis =  tf.random.uniform(shape=[1], minval=5, maxval=40)
    phi = tf.random.uniform(shape=[1], minval=0, maxval=np.pi)
    rr, cc = skimage.draw.ellipse(r=center_x, c=center_y, r_radius=minorAxis[0], c_radius=majorAxis[0], shape=crop_shape, rotation=phi[0])
    ellipse = np.zeros(crop_shape, dtype=np.uint8)
    ellipse[rr, cc] = 1
    return ellipse
################################################
def generateRectangle():
    center_x, center_y = tf.random.uniform(shape=[2], minval=32-10, maxval=32+10)
    xx, yy = tf.meshgrid(tf.range(0,crop_shape[0], dtype=tf.float32), tf.range(0,crop_shape[1], dtype=tf.float32))
    a = tf.random.uniform(shape=[1], minval=4, maxval=5)
    b = tf.random.uniform(shape=[1], minval=10, maxval=50)
    phi = tf.random.uniform(shape=[1], minval=0, maxval=np.pi)
    rectangle = tf.math.logical_and(np.abs(xx - center_x)<a/2, np.abs(yy - center_y)<b/2)
    rectangle = tf.cast(rectangle,tf.float32)
    rectangle = tf.reshape(rectangle, crop_shape)
    rectangle = tfa.image.rotate(rectangle, tf.constant(phi))
    rectangle = tf.cast(rectangle,tf.bool)
    return rectangle
################################################
################################################
def generateEmpty(center_x, center_y):
    return np.zeros(crop_shape)
################################################
def shapes_images_generator():
    for number in range(0,64000): 
        image = generateRectangle()
        #image = generateEllipse()
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