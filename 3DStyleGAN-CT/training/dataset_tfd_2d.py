# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Multi-resolution input data pipeline."""

import os
import glob
import numpy as np
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

import dnnlib
import dnnlib.tflib as tflib

# Own Tensorflow dataset
from pydicom import dcmread
from scipy import ndimage

#----------------------------------------------------------------------------
# Dataset class that loads data from tfrecords files.
dtypeGlob = tf.float32 
dtypeGlob_np = np.float32 


class TFRecordDataset:
    def __init__(self,
        tfrecord_dir,               # Directory containing a collection of tfrecords files.
        resolution      = None,     # Dataset resolution, None = autodetect.
        label_file      = None,     # Relative path of the labels file, None = autodetect.
        max_label_size  = 0,        # 0 = no labels, 'full' = full labels, <int> = N first label components.
        max_images      = None,     # Maximum number of images to use, None = use all images.
        repeat          = True,     # Repeat dataset indefinitely?
        shuffle_mb      = 4096,     # Shuffle data within specified window (megabytes), 0 = disable shuffling.
        prefetch_mb     = 2048,     # Amount of data to prefetch (megabytes), 0 = disable prefetching.
        buffer_mb       = 256,      # Read buffer size (megabytes).
        num_threads     = 4,        # Number of concurrent threads.
        base_size = [ 8, 8 ]     # Size of Base Layer
        ):       

        self.tfrecord_dir       = tfrecord_dir
        self.resolution         = None
        self.resolution_log2    = None
        self.shape              = []        # [channels, height, width]
        self.dtype              = 'float32'
        self.dynamic_range      = [0, 1.0]
        self.label_file         = label_file
        self.label_size         = None      # components
        self.label_dtype        = None
        self._np_labels         = None
        self._tf_minibatch_in   = None
        self._tf_labels_var     = None
        self._tf_labels_dataset = None
        self._tf_datasets       = dict()
        self._tf_iterator       = None
        self._tf_init_ops       = dict()
        self._tf_minibatch_np   = None
        self._cur_minibatch     = -1
        self._cur_lod           = -1
        
        self.base_size = base_size


        # List all dcm files
        assert os.path.isfile(self.tfrecord_dir)
        with open(self.tfrecord_dir) as f:
            img_files = f.read().splitlines()
        self.img_files = img_files
        img_shape = [s*32 for s in self.base_size] # e.g. 256x256 
        img_shape.insert(0, 1) # 1x256x256
        
        # Autodetect label filename.
        if self.label_file is None:
            guess = sorted(glob.glob(os.path.join(self.tfrecord_dir, '*.labels')))
            if len(guess):
                self.label_file = guess[0]
        elif not os.path.isfile(self.label_file):
            guess = os.path.join(self.tfrecord_dir, self.label_file)
            if os.path.isfile(guess):
                self.label_file = guess

        # Determine shape and resolution.
        max_shape = img_shape
        print("#imgs:",len(img_files))
        print(self.base_size)
        print(img_shape)
        print(max_shape)
        max_res = np.max( max_shape[ 1: ] )

        self.resolution = max_res
        self.resolution_log2 = int(np.log2(self.resolution))
        self.shape = max_shape


        # Load labels.
        assert max_label_size == 'full' or max_label_size >= 0
        self._np_labels = np.zeros([1<<30, 0], dtype=np.float16)
        if self.label_file is not None and max_label_size != 0:
            self._np_labels = np.load(self.label_file)
            assert self._np_labels.ndim == 2
        if max_label_size != 'full' and self._np_labels.shape[1] > max_label_size:
            self._np_labels = self._np_labels[:, :max_label_size]
        if max_images is not None and self._np_labels.shape[0] > max_images:
            self._np_labels = self._np_labels[:max_images]
        self.label_size = self._np_labels.shape[1]
        self.label_dtype = self._np_labels.dtype.name


        ## load dicom image series from folder
        def load_dicom_convert(filename, vmin=-250, vmax=650):
            if "ENVcon" in filename: # for contrast CTCA imgs
                vmin = -350
                vmax = 1000
            ds = dcmread(filename)
            image = ds.pixel_array
            image = image * ds.RescaleSlope + ds.RescaleIntercept
            image = np.clip(image, a_min=vmin, a_max=vmax)
            #image = 2*(image-vmin)/(vmax-vmin) - 1.0
            image = 1.*(image-vmin)/(vmax-vmin)     # [0.0, 1.0] consistent with MRI example
            #if image.ndim == 2:
            #    image = image[:, :, np.newaxis]
            #image = image.transpose(2, 0, 1) # HWC => CHW
            image = ndimage.zoom(image, self.shape[1]/image.shape[1]) 
            return image.astype(dtypeGlob_np)

        ## numpy function for dataset map of each item
        def map_image_np(t: np.ndarray):
            img_batch = np.empty((0, self.shape[1], self.shape[2]), dtype=dtypeGlob_np)
            if np.issubdtype(type(t), int):
                t = np.array([t])
            for i in range(t.shape[0]):
                #print(t[i], self.img_folders[t[i]])
                img = load_dicom_convert(self.img_files[t[i]]) # 256x256
                img_np = img[np.newaxis, :, :] # 1x256x256
                img_batch = np.concatenate((img_batch,img_np), axis=0).astype(dtypeGlob_np)

            return np.expand_dims(img_batch, axis=1)

    
        # Build TF expressions.
        with tf.name_scope('Dataset'), tf.device('/cpu:0'):
            ## labels dataset
            self._tf_labels_var = tflib.create_var_with_large_initial_value(self._np_labels, name='labels_var')
            self._tf_labels_dataset = tf.data.Dataset.from_tensor_slices(self._tf_labels_var)

            ## CTCA images dataset
            dset = tf.data.Dataset.range(len(self.img_files))
            dset = dset.shuffle(len(self.img_files))
            if repeat:
                print( "=================================================" )
                print( " Dataset Repeated" )
                print( "=================================================" )
                dset = dset.repeat()
            #dset = dset.prefetch(8)
            self._tf_minibatch_in = tf.compat.v1.placeholder(tf.int64, name='minibatch_in', shape=[])
            dset = dset.batch(self._tf_minibatch_in)

            # self._tf_minibatch_in = tf.keras.Input(name='minibath_in', shape=(), dtype=tf.dtypes.int64)
            #dataset = tf.data.Dataset.zip((dataset, self._tf_labels_dataset))
            dset = dset.map(lambda x: tf.numpy_function(func=map_image_np, inp=[x], Tout=[dtypeGlob]), num_parallel_calls=num_threads)

            #self._tf_iterator = dataset.make_initializable_iterator()
            #self._tf_iterator = tf.data.Iterator.from_structure(, tf.TensorShape([]))
            self._tf_iterator = tf.data.Iterator.from_structure(dset.output_types, dset.output_shapes)
            self.iterator_initializer = self._tf_iterator.make_initializer(dset)


    def close(self):
        pass

    # Use the given minibatch size and level-of-detail for the data returned by get_minibatch_tf().
    def configure(self, minibatch_size, lod=0):
        lod = int(np.floor(lod))
        assert minibatch_size >= 1 
        #print("bs:", minibatch_size)
        if self._cur_minibatch != minibatch_size or self._cur_lod != lod:
            #with tf.Session() as sess:
            #    print("Initialize iterator 3")
            #    sess.run(self.iterator_initializer, {self._tf_minibatch_in: minibatch_size})
            tflib.run(self.iterator_initializer, {self._tf_minibatch_in: minibatch_size})
            self._cur_minibatch = minibatch_size
            self._cur_lod = lod

    # Get next minibatch as TensorFlow expressions.
    def get_minibatch_tf(self): # => images, labels
        return self._tf_iterator.get_next()[0], tf.zeros([self._cur_minibatch, 0])

    # Get next minibatch as NumPy arrays.
    def get_minibatch_np(self, minibatch_size, lod=0): # => images, labels
        self.configure(minibatch_size, lod)
        with tf.name_scope('Dataset'):
            if self._tf_minibatch_np is None:
                print( "tf_minibatch_np : None - Using get_minibatch_tf" )
                self._tf_minibatch_np = self.get_minibatch_tf()
            #print( self._tf_minibatch_np )
            return tflib.run(self._tf_minibatch_np)

    # Get random labels as TensorFlow expression.
    def get_random_labels_tf(self, minibatch_size): # => labels
        with tf.name_scope('Dataset'):
            if self.label_size > 0:
                with tf.device('/cpu:0'):
                    return tf.gather(self._tf_labels_var, tf.random_uniform([minibatch_size], 0, self._np_labels.shape[0], dtype=tf.int32))
            return tf.zeros([minibatch_size, 0], self.label_dtype)

    # Get random labels as NumPy array.
    def get_random_labels_np(self, minibatch_size): # => labels
        if self.label_size > 0:
            return self._np_labels[np.random.randint(self._np_labels.shape[0], size=[minibatch_size])]
        return np.zeros([minibatch_size, 0], self.label_dtype)

    # Parse individual image from a tfrecords file into TensorFlow expression.
    @staticmethod
    def parse_tfrecord_tf(record):
        # features = tf.parse_single_example(record, features={ 'data': tf.io.FixedLenFeature([], tf.string), 'shape': tf.io.FixedLenFeature([4], tf.int64)} )
        features = tf.io.parse_single_example(record, features={
            'shape': tf.io.FixedLenFeature([4], tf.int64),
            'data': tf.io.FixedLenFeature([], tf.string)})
        data = tf.io.decode_raw(features['data'], tf.float32)

        data = tf.cast( data, dtypeGlob )
        return tf.reshape(data, features['shape'])

    # Parse individual image from a tfrecords file into NumPy array.
    @staticmethod
    def parse_tfrecord_np(record):
        ex = tf.train.Example()
        ex.ParseFromString(record)
        shape = ex.features.feature['shape'].int64_list.value # pylint: disable=no-member
        data = ex.features.feature['data'].bytes_list.value[0] # pylint: disable=no-member
        data = np.fromstring( data, np.float32 ) 
        data = data.astype( dtypeGlob_np )

        return data.reshape(shape)

#----------------------------------------------------------------------------
# Helper func for constructing a dataset object using the given options.

def load_dataset(class_name=None, data_dir=None, verbose=False, **kwargs):
    kwargs = dict(kwargs)
    if 'tfrecord_dir' in kwargs:
        if class_name is None:
            class_name = __name__ + '.TFRecordDataset'
        if data_dir is not None:
            kwargs['tfrecord_dir'] = os.path.join(data_dir, kwargs['tfrecord_dir'])

    assert class_name is not None
    if verbose:
        print('Streaming data using %s...' % class_name)
    dataset = dnnlib.util.get_obj_by_name(class_name)(**kwargs)
    if verbose:
        print('Dataset shape =', np.int32(dataset.shape).tolist())
        print('Dynamic range =', dataset.dynamic_range)
        print('Label size    =', dataset.label_size)
    return dataset


def load_3d_dataset(class_name=None, data_dir=None, verbose=True, **kwargs):
    kwargs = dict(kwargs)
    if 'tfrecord_dir' in kwargs:
        if class_name is None:
            class_name = __name__ + '.TFRecordDataset'
        if data_dir is not None:
            # the txt files containing all the CTCA image-series folders, e.g. datasets/SCT12bit/SCT_180260/noncontrast 
            kwargs['tfrecord_dir'] = os.path.join(data_dir, kwargs['tfrecord_dir'])

    assert class_name is not None
    if verbose:
        print('Streaming data using %s...' % class_name)
    dataset = dnnlib.util.get_obj_by_name(class_name)(**kwargs)
    if verbose:
        print('Dataset shape =', np.int64(dataset.shape).tolist())
        print('Dynamic range =', dataset.dynamic_range)
        print('Label size    =', dataset.label_size)
    return dataset

#----------------------------------------------------------------------------
