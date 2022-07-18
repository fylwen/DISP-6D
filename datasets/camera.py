# -*- coding: utf-8 -*-
import tensorflow as tf

import cv2
from datasets.dataset_utils import lazy_property
from datasets.image_augmentation_functions import *
import glob

class MultiQueue(object):

    def __init__(self, batch_size, img_shape, list_objs, aug_args):
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.num_objs = len(list_objs)
        self.next_element = None
        
        self.zoom_range = aug_args['zoom_pad']
        self.contrast_norm_range = aug_args['contrast_norm']
        self.mult_brightness = aug_args['mult_brightness']
        self.max_off_brightness = aug_args['max_off_brightness']
    
        self.bg_img_init = None
        self.next_bg_element = None

        self.obj_id_to_embed_order=np.zeros((10000,),dtype=np.int64)-1
        self.list_objs= list_objs
        for tid, oid in enumerate(self.list_objs):
            self.obj_id_to_embed_order[oid]=tid

    
    def deserialize_tfrecord(self, example_proto):
        features = {"train_x": tf.FixedLenFeature((), tf.string),
                "train_y": tf.FixedLenFeature((), tf.string),                     
                "depth_y": tf.FixedLenFeature((), tf.string),
                "obj_id": tf.FixedLenFeature((), tf.int64),
                "hsh_code":tf.FixedLenFeature((), tf.string)}

        parsed_features = tf.parse_single_example(example_proto, features)
        train_x = tf.reshape(tf.decode_raw(parsed_features['train_x'], tf.uint8),self.img_shape)
        train_y = tf.reshape(tf.decode_raw(parsed_features['train_y'], tf.uint8),self.img_shape)
        depth_y= tf.reshape(tf.decode_raw(parsed_features['depth_y'], tf.float32),self.img_shape[:2])
        
        hsh_code=tf.reshape(tf.decode_raw(parsed_features['hsh_code'], tf.float32),(1,128))
        obj_id=tf.reshape(parsed_features['obj_id'],(1,))

        return train_x,train_y,depth_y,obj_id,hsh_code

    def _tf_augmentations(self, train_x,train_y,depth_y, obj_id,hsh_code):
        train_x = zoom_image_object(train_x,np.linspace(self.zoom_range[0], self.zoom_range[1], 50).astype(np.float32),pad_value=1.0)
        zoom_x_mask=tf.where(train_x>1.0-1e-4,tf.ones_like(train_x),tf.zeros_like(train_x))

        #for camera
        train_c=tf.identity(train_x)
        train_c=tf.image.random_brightness(image=train_c, max_delta=self.max_off_brightness, seed=None)
        train_c=tf.clip_by_value(train_c,0,1)
        train_c=tf.where(zoom_x_mask>1.0-1e-4, tf.ones_like(train_c),train_c)
        
        #for others        
        train_x = random_brightness(train_x, self.max_off_brightness)
        train_x = invert_color(train_x)
        train_x=tf.where(zoom_x_mask>1.0-1e-4, tf.ones_like(train_x),train_x)
        
        train_x=self._set_for_camera(train_x, train_c, obj_id)
        return (train_x, train_y, depth_y, obj_id, hsh_code)
        
    

    def _float_cast(self, train_x,train_y,depth_y,obj_id,hsh_code):
        train_x = tf.image.convert_image_dtype(train_x,tf.float32)
        train_y = tf.image.convert_image_dtype(train_y,tf.float32)
        return train_x,train_y,depth_y, obj_id,hsh_code

    def _set_for_camera(self, train_o, train_c, obj_id):
        with tf.variable_scope('set_encoder_inputs'):
            train_o = tf.cond(tf.logical_and(tf.greater_equal(obj_id,tf.constant(399,dtype=tf.int64)),tf.less(obj_id,tf.constant(473,dtype=tf.int64))), 
                lambda:tf.identity(train_c), lambda:tf.identity(train_o))

            train_o = tf.cond(tf.logical_and(tf.greater_equal(obj_id,tf.constant(399+1085,dtype=tf.int64)),tf.less(obj_id,tf.constant(473+1085,dtype=tf.int64))), 
                lambda:tf.identity(train_c), lambda:tf.identity(train_o))
            return train_o
    

    def _normalize_trainx_trainy(self, train_x, train_y, depth_y, obj_id, hsh_code):
        with tf.variable_scope('normalize_imgs'):
            _normalized_bgr_x= tf.image.per_image_standardization(train_x)
            _normalized_bgr_y= tf.image.per_image_standardization(train_y)

            min_normalized_bgrx=tf.reduce_min(_normalized_bgr_x)
            max_normalized_bgrx=tf.reduce_max(_normalized_bgr_x)

            min_normalized_bgry=tf.reduce_min(_normalized_bgr_y)
            max_normalized_bgry=tf.reduce_max(_normalized_bgr_y)

            train_x=(_normalized_bgr_x-min_normalized_bgrx)/(max_normalized_bgrx-min_normalized_bgrx)
            train_y=(_normalized_bgr_y-min_normalized_bgry)/(max_normalized_bgry-min_normalized_bgry)

            train_x = tf.where(tf.is_nan(train_x), tf.zeros_like(train_x), train_x)
            train_y = tf.where(tf.is_nan(train_y), tf.zeros_like(train_y), train_y)
            
            return train_x, train_y, depth_y, obj_id, hsh_code

    def preprocess_pipeline(self, dataset):
        dataset = dataset.map(self.deserialize_tfrecord)       
        dataset = dataset.shuffle(buffer_size=1000)
        
        dataset = dataset.map(lambda train_x,train_y,depth_y,obj_id,hsh_code :
                                self._float_cast(train_x,train_y,depth_y,obj_id,hsh_code))
        dataset = dataset.repeat()
        dataset = dataset.map(lambda train_x,train_y,depth_y,obj_id,hsh_code :
                                self._tf_augmentations(train_x,train_y,depth_y,obj_id,hsh_code))

        dataset = dataset.map(lambda train_x,train_y,depth_y,obj_id,hsh_code:
                            self._normalize_trainx_trainy(train_x,train_y,depth_y,obj_id,hsh_code))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(1)
        return dataset

    def create_iterator(self, path_tf_records):
        tf_dataset = tf.data.TFRecordDataset(path_tf_records, compression_type = 'ZLIB')
        tf_dataset = self.preprocess_pipeline(tf_dataset)

        iterator = tf_dataset.make_initializable_iterator()
        self.next_element = iterator.get_next()
        return iterator


