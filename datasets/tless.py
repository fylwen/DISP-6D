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
                "obj_id": tf.FixedLenFeature((), tf.int64),
                "hsh_code":tf.FixedLenFeature((), tf.string)}

        parsed_features = tf.parse_single_example(example_proto, features)
        train_x = tf.reshape(tf.decode_raw(parsed_features['train_x'], tf.uint8),self.img_shape)
        train_y = tf.reshape(tf.decode_raw(parsed_features['train_y'], tf.uint8),self.img_shape)
        
        hsh_code=tf.reshape(tf.decode_raw(parsed_features['hsh_code'], tf.float32),(1,128))
        obj_id=tf.reshape(parsed_features['obj_id'],(1,))

        return train_x,train_y,obj_id,hsh_code

    def _tf_augmentations(self, train_x,train_y,obj_id,hsh_code,bg):
        train_x = zoom_image_object(train_x,np.linspace(self.zoom_range[0], self.zoom_range[1], 50).astype(np.float32),pad_value=0.)       
        train_x = add_background(train_x, bg)
        train_x = random_brightness(train_x, self.max_off_brightness)
        train_x = invert_color(train_x)
        train_x = multiply_brightness(train_x, self.mult_brightness)
        train_x = contrast_normalization(train_x, self.contrast_norm_range)

        return train_x,train_y,obj_id,hsh_code
    

    def _float_cast(self, train_x,train_y,obj_id,hsh_code):
        train_x = tf.image.convert_image_dtype(train_x,tf.float32)
        train_y = tf.image.convert_image_dtype(train_y,tf.float32)
        return train_x,train_y,obj_id,hsh_code

    def load_bg_imgs(self, path_img):
        return tf.image.resize_images(tf.image.convert_image_dtype(tf.image.decode_jpeg(tf.read_file(path_img)),tf.float32),
                    [self.img_shape[0],self.img_shape[1]])

    def create_background_image_iterator(self, path_bg_imgs):
        background_imgs_dataset = tf.data.Dataset.from_tensor_slices(path_bg_imgs)
        background_imgs_dataset = background_imgs_dataset.map(map_func=self.load_bg_imgs, num_parallel_calls = 4)
        # background_imgs_dataset = background_imgs_dataset.cache()
        background_imgs_dataset = background_imgs_dataset.shuffle(1000)
        background_imgs_dataset = background_imgs_dataset.repeat()
        background_imgs_dataset = background_imgs_dataset.prefetch(1)

        self.bg_img_init = background_imgs_dataset.make_initializable_iterator()
        self.next_bg_element = self.bg_img_init.get_next()



    def preprocess_pipeline(self, dataset):
        dataset = dataset.map(self.deserialize_tfrecord)       
        dataset = dataset.shuffle(buffer_size=10000)
        
        dataset = dataset.map(lambda train_x,train_y,obj_id,hsh_code :
                                self._float_cast(train_x,train_y,obj_id,hsh_code))
        dataset = dataset.repeat()
        dataset = dataset.map(lambda train_x,train_y,obj_id,hsh_code :
                                self._tf_augmentations(train_x,train_y,obj_id,hsh_code, self.bg_img_init.get_next()))

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(1)
        return dataset

    def create_iterator(self, path_tf_records, path_bg_imgs):
        self.create_background_image_iterator(path_bg_imgs)

        tf_dataset = tf.data.TFRecordDataset(path_tf_records, compression_type = 'ZLIB')
        tf_dataset = self.preprocess_pipeline(tf_dataset)

        iterator = tf_dataset.make_initializable_iterator()
        self.next_element = iterator.get_next()
        return iterator


