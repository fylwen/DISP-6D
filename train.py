import os
import cv2
import numpy as np
import tensorflow as tf

import argparse
import progressbar
import signal
gentle_stop = np.array((1,), dtype=np.bool)
gentle_stop[0] = False
def on_ctrl_c(signal, frame):
    gentle_stop[0] = True

signal.signal(signal.SIGINT, on_ctrl_c)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
num_gpus_used = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

from utils import convert_batch_to_image_grid, add_summary, build_encoder, build_decoder, build_conditioned_block, build_dataset
from network import VectorQuantizerEMA, compute_reconstruction_losses

parser = argparse.ArgumentParser()
parser.add_argument('--training_data_type',default='tless_f18',\
                choices=['tless_f18','camera_all','camera_bottle','camera_bowl','camera_camera','camera_can','camera_laptop','camera_mug'], \
                help='Training dataset, tless_f18 for first 18 tless objects, camera_all for a combination of all 6 categories in CAMERA, and camera_category for the specified category in CAMERA.')
parser.add_argument('--dir_tfrecords',default='./dir/to/tfrecords',help='Path of directory for tfrecords.')
parser.add_argument('--dir_bg_imgs',default='./dir/to/background/images',help='Path of directory for background images (in jpg format).')

parser.add_argument('--normalize_encoder',action="store_true",help='Apply instance normalization in encoder.')    
parser.add_argument('--decoder_branches', default='bgr',choices=['bgr','depth','bgr+depth'],help='Decoder branches.')
parser.add_argument('--tau',default=0.07, type=float,help='Temperature for shape loss.')
parser.add_argument('--ema_decay',default=0.9995,type=float,help='EMA decay for shape embedding.')
parser.add_argument('--lambda_shape_loss',default=0.004,type=float,help='Weight for shape loss')
parser.add_argument('--lambda_pose_loss',default=0.002,type=float,help='Weight for pose loss')


parser.add_argument('--lr',default=2e-4,type=float,help='Learning rate.')
parser.add_argument('--batch_size',default=64,type=int,help='Total batch size')
parser.add_argument('--num_ites',default=50000,type=int,help='Number of iterations for training')
parser.add_argument('--dir_checkpoints',default='./ws/my_exp/',help='path to save checkpoints and summary for tensorboard.')


args = parser.parse_args()
if not os.path.exists(args.dir_checkpoints):
    os.makedirs(args.dir_checkpoints)

dir_train_figs=os.path.join(args.dir_checkpoints,'train_figures')
if not os.path.exists(dir_train_figs):
    os.makedirs(dir_train_figs)

assert num_gpus_used==1, 'todo: for multiple gpu'
tf.reset_default_graph()


with tf.variable_scope('my_exp'):
    with tf.variable_scope('data_queue'):
        dataset,iterator=build_dataset(dataset_tag=args.training_data_type,\
                            dir_tfrecords=args.dir_tfrecords,
                            dir_bg_imgs=args.dir_bg_imgs,
                            batch_size=args.batch_size)
        obj_id_to_embed_order = tf.convert_to_tensor(dataset.obj_id_to_embed_order, dtype=tf.int32)
        next_element=dataset.next_element
        if args.decoder_branches=='bgr':
            enc_x,dec_gt_bgr,obj_label,hsh_code=next_element
            dec_gt_depth=tf.constant(0.)
        elif args.decoder_branches=='bgr+depth':
            enc_x,dec_gt_bgr, dec_gt_depth,obj_label,hsh_code=next_element
        elif args.decoder_branches=='depth':
            enc_x,dec_gt_depth,obj_label,hsh_code=next_element
            dec_gt_bgr=tf.constant(0.)
        obj_label=tf.nn.embedding_lookup(obj_id_to_embed_order,obj_label)

        


    with tf.variable_scope('encoder'):
        encoder = build_encoder(inst_norm=args.normalize_encoder)
        z = encoder(enc_x)

    with tf.variable_scope('gprior'):
        embedding_dim=encoder._latent_space_size
        obj_embeds = VectorQuantizerEMA(embedding_dim=embedding_dim, num_embeddings=dataset.num_objs, name='obj_prior')
        cond_block = build_conditioned_block()

        obj_retr = obj_embeds(z['z_obj'], decay=args.ema_decay, temperature=args.tau, encoding_1nn_indices=obj_label,\
                        encodings=None, is_training=(num_gpus_used == 1))
       
        cond_pose_code=cond_block(condition_code=tf.stop_gradient(z['z_obj']),src_code=tf.reshape(hsh_code,[-1,1,embedding_dim]))
        cond_pose_code=tf.reshape(cond_pose_code,[-1,embedding_dim])
       

    with tf.variable_scope('decoder'):
        decoder_branches={}
        for db in ['bgr','depth']:
            if db in args.decoder_branches:
                decoder_branches[db]=1.
        decoder = build_decoder(branches=list(decoder_branches.keys()))
        dec_out = decoder(pose_code=z['z_pose'], obj_code=z['z_obj'], is_training=True)

    with tf.variable_scope('loss'):        
        summaries = {}
        total_loss = 0.

        dec_gt={'bgr':dec_gt_bgr,'depth':dec_gt_depth}
        recon_error = compute_reconstruction_losses(dec_out,dec_gt, decoder_branches)
        total_loss += recon_error['reconst_loss']
        for k in decoder_branches.keys():
            summaries['reconst_loss_{:s}'.format(k)] = recon_error['reconst_loss_{:s}'.format(k)]
        

        total_loss += obj_retr["loss"]*args.lambda_shape_loss
        summaries['shape_loss'] = obj_retr["loss"]


        pose_loss = tf.reduce_mean((tf.nn.l2_normalize(cond_pose_code, axis=1)-tf.nn.l2_normalize(z['z_pose'], axis=1))**2)
        total_loss+=pose_loss*args.lambda_pose_loss/2
        summaries['pose_loss'] = pose_loss

        #pose_cosine= tf.reduce_mean(tf.multiply(tf.nn.l2_normalize(cond_pose_code, axis=1),tf.nn.l2_normalize(z['z_pose'], axis=1)))
        #summaries['pose_check']=(1-pose_cosine*128.)-pose_loss*128./2


    add_summary([summaries])
    with tf.variable_scope('update_grads'):
        global_step = tf.Variable(0, dtype=tf.int64, trainable=False, name='global_step')
        optim = tf.train.AdamOptimizer(learning_rate=args.lr)
        train_op = tf.contrib.training.create_train_op(total_loss, optim, global_step=global_step)
    saver = tf.train.Saver(save_relative_paths=True, max_to_keep=50)
    merged_summary = tf.summary.merge_all()






widgets = ['Training: ', progressbar.Percentage(),
           ' ', progressbar.Bar(),
           ' ', progressbar.Counter(), ' / {:d}'.format(args.num_ites),
           ' ', progressbar.ETA(), ' ']
bar = progressbar.ProgressBar(maxval=args.num_ites, widgets=widgets)

gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.9)
config = tf.ConfigProto(gpu_options=gpu_options)

with tf.Session(config=config) as sess:    
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer)
    if not (dataset.bg_img_init is None):
        sess.run(dataset.bg_img_init.initializer)
       
    sess.graph.finalize()
    summary_writer = tf.summary.FileWriter(args.dir_checkpoints, sess.graph)
    bar.start()

    for i in range(0, args.num_ites):
        bar.update(i)
        if i % 1000 == 0:
            results = sess.run([merged_summary,enc_x,dec_gt,dec_out,train_op, global_step])
            gs = results[-1]
            summary_writer.add_summary(results[0], gs)
            cv2.imwrite(os.path.join(dir_train_figs, '{:05d}_x.png'.format(i)), (convert_batch_to_image_grid(results[1]) * 255.).astype(np.uint8))

            if 'bgr' in args.decoder_branches:
                cv2.imwrite(os.path.join(dir_train_figs, '{:05d}_y_gt.png'.format(i)), (convert_batch_to_image_grid(results[2]['bgr']) * 255.).astype(np.uint8))
                cv2.imwrite(os.path.join(dir_train_figs, '{:05d}_y_out.png'.format(i)), (convert_batch_to_image_grid(results[3]['x_bgr']) * 255.).astype(np.uint8))
            if 'depth' in args.decoder_branches:
                cv2.imwrite(os.path.join(dir_train_figs, '{:05d}_yd_gt.png'.format(i)), (convert_batch_to_image_grid(results[2]['depth']) * 255.).astype(np.uint8))
                cv2.imwrite(os.path.join(dir_train_figs, '{:05d}_yd_out.png'.format(i)), (convert_batch_to_image_grid(results[3]['x_depth']) * 255.).astype(np.uint8))
        elif i%100==0:
            results=sess.run([merged_summary,train_op,global_step])
            gs = results[-1]
            summary_writer.add_summary(results[0], gs)
        else:
            results = sess.run([train_op, global_step])   
            gs = results[-1]        

        if (i + 1) % 10000 == 0 or i == 0:
            path_ckpt=os.path.join(args.dir_checkpoints,'chkpt')
            saver.save(sess, path_ckpt, global_step=gs+1)

        if gentle_stop[0]:
            break


