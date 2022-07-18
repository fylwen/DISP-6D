import numpy as np
import tensorflow as tf
import cv2
from network import Encoder,ConditionedBlock
from network_decoder import AdaptiveDecoder
import random,os,glob

def est_tra_w_tz(tz, Radius_render_train, K_test, center_obj_x_test, center_obj_y_test,
                 K_train, center_obj_x_train, center_obj_y_train):
    tx = center_obj_x_test *tz / K_test[0, 0] \
                       - center_obj_x_train * Radius_render_train / K_train[0, 0]
    ty = center_obj_y_test * tz / K_test[1, 1] \
                       - center_obj_y_train * Radius_render_train / K_train[1, 1]
    return np.array([tx,ty,tz]).reshape((3, 1))

def update_xy_w_z(tz,K_test,bbox_test):
    bb_x=bbox_test[0]+bbox_test[2]/2
    bb_y=bbox_test[1]+bbox_test[3]/2
    tx=tz*(bb_x-K_test[0,2])/K_test[0,0]
    ty=tz*(bb_y-K_test[1,2])/K_test[1,1]
    return np.array([tx,ty,tz]).reshape((3,1))


def rectify_rot(init_rot, est_tra):
    est_tra=est_tra.squeeze()
    d_alpha_x = - np.arctan(est_tra[0] / est_tra[2])
    d_alpha_y = - np.arctan(est_tra[1] / est_tra[2])
    R_corr_x = np.array([[1, 0, 0],
                         [0, np.cos(d_alpha_y), -np.sin(d_alpha_y)],
                         [0, np.sin(d_alpha_y), np.cos(d_alpha_y)]]).reshape((3,3))
    R_corr_y = np.array([[np.cos(d_alpha_x), 0, -np.sin(d_alpha_x)],
                         [0, 1, 0],
                         [np.sin(d_alpha_x), 0, np.cos(d_alpha_x)]]).reshape((3,3))
    return np.dot(R_corr_y, np.dot(R_corr_x, init_rot))


def convert_batch_to_image_grid(batch_image, row=4,col=8):
    if batch_image.shape[0]<row*col:
        row=batch_image.shape[0]//col
    image_size=batch_image.shape[1]
    if batch_image.ndim == 3:
        batch_image = np.expand_dims(batch_image[0:row*col, :, :], -1)
    reshaped = (batch_image[0:row*col, :, :, :].reshape(row, col, image_size, image_size, -1)
                .transpose(0, 2, 1, 3, 4)
                .reshape(row * image_size, col * image_size, -1))
    return reshaped



def add_summary(tower_summaries):
    for k in tower_summaries[0].keys():
        x = tf.identity(tower_summaries[0][k])
        for i in range(1, len(tower_summaries)):
            x += tower_summaries[i][k]
        tf.summary.scalar(k, x / len(tower_summaries))


def load_real275_depth(depth_path):
    """Generate an image from the specs of the given image ID.
    Typically this function loads the image from a file.
    """
    depth = cv2.imread(depth_path, -1)
    if len(depth.shape) == 3:
        # This is encoded depth image, let's convert
        depth16 = np.uint16(depth[:, :, 1] * 256) + np.uint16(depth[:, :, 2])  # NOTE: RGB is actually BGR in opencv
        depth16 = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':
        depth16 = depth
    else:
        assert False, '[ Error ]: Unsupported depth type.'
    return depth16


def build_encoder(inst_norm):
    LATENT_SPACE_SIZE=128
    NUM_FILTER =[128, 256, 512, 512]
    KERNEL_SIZE_ENCODER = 5
    STRIDES = [2, 2, 2, 2]
    HIDDEN_ACT = 'lrelu'
    encoder = Encoder(
        latent_space_size=LATENT_SPACE_SIZE,
        num_filters=NUM_FILTER,
        kernel_size=KERNEL_SIZE_ENCODER,
        strides=STRIDES,
        hidden_act=HIDDEN_ACT,
        inst_norm=inst_norm,
    )
    return encoder


def build_decoder(branches):
    RECONSTRUCTION_SHAPE=[128,128,3]
    NUM_FILTER =[128, 256, 512, 512]
    KERNEL_SIZE_DECODER = 5
    STRIDES =  [2, 2, 2, 2]
    HIDDEN_ACT='lrelu'
    FINAL_ACT='sigmoid'

    decoder = AdaptiveDecoder(
        reconstruction_shape=RECONSTRUCTION_SHAPE,
        num_filters=list(reversed(NUM_FILTER)),
        kernel_size=KERNEL_SIZE_DECODER,
        strides=list(reversed(STRIDES)),
        hidden_act=HIDDEN_ACT,
        final_act=FINAL_ACT,
        branches=branches,)
        
    return decoder

def build_conditioned_block():
    LANTENT_SPACE_SIZE=128
    LATENT_DIM_CONDITION=128
    LATENT_DIM_SRC=128
    LATENT_DIM_BT=128
    LATEND_DIMS_FF=[512,128]
    HIDDEN_ACT='lrelu'

    cblock=ConditionedBlock(latent_space_size=LANTENT_SPACE_SIZE,
                            latent_dim_condition=LATENT_DIM_CONDITION,
                            latent_dim_src=LATENT_DIM_SRC,
                            latent_dim_bt=LATENT_DIM_BT,
                            latent_dims_ff=LATEND_DIMS_FF,
                            hidden_act=HIDDEN_ACT, 
                            name='cond_block_f')
    return cblock


def build_dataset(dataset_tag, dir_tfrecords,dir_bg_imgs, batch_size):    
    RECONSTRUCTION_SHAPE=[128,128,3]
    AUG_ARGS={'zoom_pad':[0.8,1.2],'contrast_norm':[0.5,2.0], 'mult_brightness': [0.6, 1.4],'max_off_brightness': 0.2}
    if dataset_tag=='tless_f18':
        list_objs=list(range(1,19))
        path_tf_records = [os.path.join(dir_tfrecords,'{:02d}'.format(oid),'{:05d}.tfrecords'.format(fid)) for fid in range(0, 922) for oid in list_objs]#
        random.shuffle(path_tf_records)

        path_bg_imgs=glob.glob(os.path.join(dir_bg_imgs,'*.jpg'))

        from datasets.tless import MultiQueue
        dataset=MultiQueue(batch_size=batch_size, img_shape=RECONSTRUCTION_SHAPE, list_objs=list_objs, aug_args=AUG_ARGS)
        iterator=dataset.create_iterator(path_tf_records=path_tf_records,path_bg_imgs=path_bg_imgs)
    elif dataset_tag in ['camera_all','camera_bottle','camera_bowl','camera_camera','camera_can','camera_laptop','camera_mug']:
        obj_ids_by_categories={'bottle':list(range(0+1085,255+1085)),'bowl':list(range(255+1085,399+1085)),\
                            'camera':list(range(399, 473))+list(range(399+1085, 473+1085)),'can':list(range(473+1085,522+1085)),
                            'laptop':list(range(522+1085,913+1085)),'mug':list(range(913+1085,1085+1085))}
        
        category_tag=dataset_tag.split('_')[-1]
        list_categories=[category_tag] if category_tag!='all' else ['bottle','bowl','camera','can','laptop','mug']
        list_objs=[]
        path_tf_records=[]
        for c in list_categories:
            list_objs+=obj_ids_by_categories[c]
            path_tf_records+= glob.glob(os.path.join(dir_tfrecords,c,'*.tfrecords'))
        random.shuffle(path_tf_records)

        from datasets.camera import MultiQueue
        dataset=MultiQueue(batch_size=batch_size, img_shape=RECONSTRUCTION_SHAPE, list_objs=list_objs, aug_args=AUG_ARGS)
        iterator=dataset.create_iterator(path_tf_records=path_tf_records)


    return dataset,iterator

