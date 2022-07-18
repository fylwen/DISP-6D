import os
import numpy as np
import tensorflow as tf
import cv2

from est_pipeline_tless import pose_estimation_tless
from utils import build_encoder
from network import VectorQuantizerEMA


import ruamel.yaml as yaml
import argparse

# This is the demo code for testing on the t-less dataset (Our setting II).
parser = argparse.ArgumentParser()
parser.add_argument('--test_obj_id',default=5,type=int, choices=[5,8,26,28],help='test object id, choose from 5/8/26/28 as they are contained in the demo image')
args = parser.parse_args()
test_obj_id=int(args.test_obj_id)

# Set hyper-parameters.
path_ws='./ws/'
ckpt_dir = os.path.join(path_ws,'ckpts','tless_f18')
query_crop_size = 128
num_ites=50000

img_id=275
path_test_img=os.path.join(path_ws, 'demo_data/tless_{:04d}.png'.format(img_id))
test_cam_K=[1075.65091572, 0.0, 366.06888344, 0.0, 1073.90347929, 294.72159802, 0.0, 0.0, 1.0]

#Load 2D detection result
with open(os.path.join(path_ws,'demo_data/tless_det2d.yml'), 'r') as f:
    det_infos = yaml.load(f, Loader=yaml.CLoader)

    for item in det_infos[img_id]:
        if item['obj_id']==test_obj_id:
            det_info={'obj_bb':item['obj_bb'],'score':item['score']}#score is from 2D detection
            break



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.reset_default_graph()
with tf.variable_scope('subdiv_f18_softmax'): #ignore the name of the scope
    enc_x = tf.placeholder(tf.float32, shape=(None, query_crop_size, query_crop_size, 3))

    with tf.variable_scope('encoder'):
        encoder = build_encoder(inst_norm=False)
    z_pose = encoder(enc_x)['z_pose']
    
    network_vars = tf.trainable_variables()

    embedding_zp=np.load(os.path.join(path_ws,'embeddings/tless','embedding_zp_{:02d}.npy'.format(test_obj_id)))
    embedding_bbox2d=np.load(os.path.join(path_ws,'embeddings/tless','embedding_bbox2d_{:02d}.npy'.format(test_obj_id))).reshape((-1,4))
    embedding_rotmat=np.load(os.path.join(path_ws,'embeddings/tless','embedding_rotmat_92232.npy')).reshape((-1,3,3))

    pose_embed = VectorQuantizerEMA(embedding_dim=encoder.latent_space_size, num_embeddings=embedding_zp.shape[0])
    pose_retr = pose_embed(z_pose,decay=0.9995, temperature=0.07, encoding_1nn_indices=None, encodings=None,  is_training=False)


    embedding = tf.placeholder(tf.float32, shape=(embedding_zp.shape[1],embedding_zp.shape[0]))
    embedding_assign_op = tf.assign(pose_embed.embeddings, embedding)
   

gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.9)
config = tf.ConfigProto(gpu_options=gpu_options)

saver = tf.train.Saver(network_vars, save_relative_paths=False)
with tf.Session(config=config) as sess:
   
    saver.restore(sess, os.path.join(ckpt_dir, 'chkpt-{0}'.format(num_ites)))
    sess.run(embedding_assign_op, {embedding: embedding_zp.T})

    test_img=cv2.imread(path_test_img)
    pose_estimation_tless(sess,enc_x=enc_x,pose_retr=pose_retr,query_bgr=test_img,det_info=det_info,\
            K_test=test_cam_K,embedding_rotmat=embedding_rotmat,embedding_bbox2d=embedding_bbox2d)

'''
Estimation result
Obj-05
2D object bounding box [363, 277, 100, 148]
R_m2c [[ 0.11609715  0.9890227   0.09140875]
 [ 0.69603274 -0.01535863 -0.71784576]
 [-0.70856184  0.14696333 -0.69017527]]
t_m2c(unit: mm) [ 31.54997122  34.44011098 738.03392224]
score 0.99998283


Obj-08
2D object bounding box [154, 138, 247, 228]
R_m2c [[ 0.39667818  0.91640654 -0.05334299]
 [-0.63604064  0.31628893  0.70385625]
 [ 0.66189027 -0.24527611  0.70833672]]
t_m2c(unit: mm) [-62.09927955 -36.38939554 824.61068034]
score 0.99929094


Obj-26
2D object bounding box [195, 146, 140, 154]
R_m2c [[ 0.390951   -0.91849493  0.05936651]
 [-0.60526834 -0.30514829 -0.73521069]
 [ 0.69340288  0.25149868 -0.67523393]]
t_m2c(unit: mm) [-61.56537425 -54.91427398 700.24376105]
score 0.99763989

Obj-28
2D object bounding box [412, 173, 155, 137]
R_m2c [[-0.93559122  0.34844549  0.05705088]
 [ 0.20375144  0.66475971 -0.71873491]
 [-0.28836507 -0.66081788 -0.69293962]]
t_m2c(unit: mm) [ 88.53277712 -40.50246048 803.13567895]
score 0.99999464
'''