import os
import numpy as np
import tensorflow as tf
import cv2

import ruamel.yaml as yaml
from est_pipeline_real275 import  pose_estimation_real275
from utils import build_encoder, build_conditioned_block, build_decoder, load_real275_depth
from network import ConditionedCodebook, generate_pose_position_encoding
import argparse

# This is the demo code for testing on the real275 dataset, for Ours-all (by setting --trained_category to "all") and Ours-per (otherwise).
parser = argparse.ArgumentParser()
parser.add_argument('--trained_category',choices=["all","bottle","bowl","camera","can","mug","laptop"],help="trained category")
parser.add_argument('--test_category',choices=["bottle","bowl","camera","can","mug","laptop"],help="category to test")
parser.add_argument('--demo_img', default=258,type=int, choices=[258,300], help="demo img id, in 258 or 300, note that bowl is not involved in img 258, while bottle is not involved in img 300")
args = parser.parse_args()

# Set hyper-parameters.
path_ws='./ws/'
if args.trained_category=='all': # is ours-all
    trained_category='all'
    ckpt_dir = os.path.join(path_ws,'ckpts','real275_ours_all')
    num_ites=150000
else:# is ours-per
    trained_category=args.test_category
    ckpt_dir = os.path.join(path_ws,'ckpts','real275_ours_per',trained_category)
    num_ites=50000
    
query_crop_size = 128
decoder_branches={'depth':1.,'bgr':1.,}

img_id=int(args.demo_img)
path_test_img=os.path.join(path_ws, 'demo_data/real275_{:04d}_color.png'.format(img_id))
path_test_depth=os.path.join(path_ws, 'demo_data/real275_{:04d}_depth.png'.format(img_id))
test_cam_K=[591.0125, 0, 322.525,0, 590.16775, 244.11084,0, 0, 1]

#Load 2D detection result
nocs_labels_to_id={'bottle':1,'bowl':2,'camera':3,'can':4,'laptop':5,'mug':6}
with open(os.path.join(path_ws,'demo_data/real275_det2d.yml'), 'r') as f:
    det_infos = yaml.load(f, Loader=yaml.CLoader)

det_info=None
path_test_mask=os.path.join(path_ws,'demo_data/real275_{:04d}_mask_bin32.npz'.format(img_id))
for iid, item in enumerate(det_infos[img_id]):
    if item['obj_id']==nocs_labels_to_id[args.test_category]:
        det_info={'obj_bb':item['obj_bb'],'score':item['score']}#score is from 2D detection
        seged_mask= np.load(path_test_mask)['inst_masks'][iid]
        break

if det_info is None:
    print('This test category is not detected in the demo image!')
    exit(0)



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.reset_default_graph()
with tf.variable_scope('subdiv_camera'): #ignore the name of the scope
    enc_x = tf.placeholder(tf.float32, shape=(None, query_crop_size, query_crop_size, 3))

    _normalized_bgr_x = tf.image.per_image_standardization(enc_x)
    print('after per image per_image_standardization', _normalized_bgr_x.shape)

    min_normalized_bgrx = tf.reduce_min(_normalized_bgr_x,axis=[1,2,3],keep_dims=True)
    max_normalized_bgrx = tf.reduce_max(_normalized_bgr_x,axis=[1,2,3],keep_dims=True)

    print('min,max dims', min_normalized_bgrx.shape, max_normalized_bgrx.shape)

    enc_x = (_normalized_bgr_x - min_normalized_bgrx) / (max_normalized_bgrx - min_normalized_bgrx)
    
    with tf.variable_scope('encoder'):
        encoder=build_encoder(inst_norm=True)
        z = encoder(enc_x)

    with tf.variable_scope('gprior'):
        cond_zo_cp = build_conditioned_block()
        embedding_rot = np.load(os.path.join(path_ws,'embeddings','real275','embedding_rots_{:s}.npz'.format(trained_category)))
        embedding_rotmat,embedding_rotquat=embedding_rot["matrix"].reshape((-1,3,3)),embedding_rot["quaternion"].reshape((-1,4))

        _=cond_zo_cp(tf.ones_like(z['z_obj']),tf.ones_like(tf.reshape(z['z_pose'],[-1,1,128])))




    with tf.variable_scope('decoder'):
        decoder = build_decoder(branches=list(decoder_branches.keys()))
        dec_y = decoder(pose_code=z['z_pose'],  obj_code=z['z_obj'], is_training=False)
    network_vars=tf.global_variables()

    with tf.variable_scope('embedding'):
        pose_embeds = ConditionedCodebook(embedding_dim=encoder.latent_space_size, num_embeddings=embedding_rotmat.shape[0], func_f=cond_zo_cp)
        pose_retr = pose_embeds(inputs=z['z_pose'], conditioned_code=z['z_obj'])


gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.9)
config = tf.ConfigProto(gpu_options=gpu_options)

saver = tf.train.Saver(network_vars)
with tf.Session(config=config) as sess:
    saver.restore(sess, os.path.join(ckpt_dir, 'chkpt-{0}'.format(num_ites)))

    #Assign hsh embedding
    embedding_hsh_code = generate_pose_position_encoding(embedding_rotquat, 6, beta_bias_deg=360 / 80., verbose=False)[:, :128]
    pose_embeds.assign_w(sess,embedding_hsh_code.T)
    seeds_w=sess.run(tf.transpose(pose_embeds._w,[1,0]))

    test_img=cv2.imread(path_test_img)
    test_depth=load_real275_depth(path_test_depth)/1000
    

    pose_estimation_real275(sess=sess, enc_x=enc_x, pose_retr=pose_retr, dec_y=dec_y,\
            query_bgr=test_img, query_depth=test_depth, det_info=det_info,query_mask=seged_mask, K_test=test_cam_K, embedding_rotmat=embedding_rotmat)


'''
Ours-all
#img 258
bottle
2D object bounding box [0, 285, 79, 123]
R_m2c [[-0.66214825 -0.22484176 -0.71484676]
 [ 0.46062732 -0.87455197 -0.15159576]
 [-0.59108561 -0.42965682  0.68265133]]
t_m2c(unit: m) [-0.35696397  0.1293227   0.74541177]
s_m2c 0.13078257118712064
pred_RTs [[-0.08659745 -0.02940538 -0.0934895  -0.35696397]
 [ 0.06024203 -0.11437616 -0.01982608  0.1293227 ]
 [-0.0773037  -0.05619162  0.0892789   0.74541177]
 [ 0.          0.          0.          1.        ]]
pred_scales [0.7272354216095549, 1.1666081479461528, 0.5891852921086154]
score 0.97497368


Ours-per
#img 300
bowl
R_m2c [[-0.03326242  0.24335106 -0.96936775]
 [ 0.73643036 -0.64975496 -0.18838466]
 [-0.67569512 -0.72013802 -0.15759868]]
t_m2c(unit: m) [-0.21728545  0.25513333  0.7950374 ]
s_m2c 0.14037813435507016
pred_RTs [[-0.00466932  0.03416117 -0.13607804 -0.21728545]
 [ 0.10337872 -0.09121139 -0.02644509  0.25513333]
 [-0.09485282 -0.10109163 -0.02212341  0.7950374 ]
 [ 0.          0.          0.          1.        ]]
pred_scales [1.0443036469637628, 0.8797275456285987, 0.9522846061637475]
score 0.97604352

'''