import numpy as np
import tensorflow as tf
import sonnet as snt
from sonnet.python.modules import base
from tensorflow.python.training import moving_averages

from pysixd import transform
from scipy.special import gamma, sph_harm, eval_gegenbauer
import math

from network_decoder import instance_norm


def lrelu_02(x):
    return tf.nn.leaky_relu(x, alpha=0.2)



class Encoder(snt.AbstractModule):
    def __init__(self, latent_space_size, num_filters, kernel_size, strides, inst_norm, hidden_act, name='encoder'):
        super(Encoder, self).__init__(name=name)
        self._latent_space_size = latent_space_size
        self._num_filters = num_filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._instance_normalization = inst_norm

        self._hidden_act =  hidden_act
        if self._hidden_act == 'relu':
            self._hidden_act_func = tf.nn.relu
        elif self._hidden_act == 'lrelu':
            self._hidden_act_func = lrelu_02


    @property
    def latent_space_size(self):
        return self._latent_space_size

    @property
    def encoder_layers(self):
        layers = []
        x = self._input
        layers.append(x)

        for filters, stride in zip(self._num_filters, self._strides):
            padding = 'same'
            x = tf.layers.conv2d(
                inputs=x,
                filters=filters,
                kernel_size=self._kernel_size,
                strides=stride,
                padding=padding,
                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                activation=self._hidden_act_func, )

            if self._instance_normalization:
                x = instance_norm(x)
                
            layers.append(x)
        return layers

    @property
    def encoder_out(self):
        x = self.encoder_layers[-1]
        encoder_out = tf.contrib.layers.flatten(x)
        return encoder_out

    @property
    def z(self):
        x = self.encoder_out

        z_obj = tf.layers.dense(x, self._latent_space_size, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='z_obj')
        z_pose = tf.layers.dense(x, self._latent_space_size, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='z_pose')
        if self._instance_normalization:
            z_obj = tf.nn.l2_normalize(z_obj, dim=1)
            z_pose = tf.nn.l2_normalize(z_pose, dim=1)
        return {'z_obj': z_obj, 'z_pose': z_pose}
        

    def _build(self, x, is_training=False):
        self._input = x
        self._is_training = is_training
        return self.z



##==========Embedding Block=========##
class VectorQuantizerEMA(base.AbstractModule):
    """
    decay: float, decay for the moving averages.
    epsilon: small float constant to avoid numerical instability.
    w: is a matrix with an embedding in each column. When training, the embedding is assigned to be the average of all inputs assigned to that embedding.
    """

    def __init__(self, embedding_dim, num_embeddings, epsilon=1e-5, name='vq_center'):
        super(VectorQuantizerEMA, self).__init__(name=name)
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._epsilon = epsilon

        with self._enter_variable_scope():
            initializer = tf.random_normal_initializer()
            self._w = tf.get_variable('embedding', [embedding_dim, num_embeddings], initializer=initializer, use_resource=True)
            self._ema_cluster_size = tf.get_variable('ema_cluster_size', [num_embeddings], initializer=tf.constant_initializer(1), use_resource=True)  # 0
            self._ema_w = tf.get_variable('ema_dw', initializer=self._w.initialized_value(), use_resource=True)

    def _build(self, inputs, decay, temperature, encoding_1nn_indices, encodings, is_training):
        with tf.control_dependencies([inputs]):
            w = self._w.read_value()


        input_shape = tf.shape(inputs)

        # Flat inputs
        with tf.control_dependencies([tf.Assert(tf.equal(input_shape[-1], self._embedding_dim), [input_shape])]):
            flat_inputs = tf.reshape(inputs, [-1, self._embedding_dim])

        # Test stage: compute the index with minimal cosine error
        if encoding_1nn_indices is None:
            distances = -tf.matmul(tf.nn.l2_normalize(flat_inputs, axis=1), tf.nn.l2_normalize(w, axis=0))
            encoding_1nn_indices = tf.argmax(-distances, 1)

        if encodings is None:
            encodings = tf.squeeze(tf.one_hot(encoding_1nn_indices, self._num_embeddings))

        # Reshape Encoding 1nn indices for loss computation
        encoding_1nn_indices = tf.reshape(encoding_1nn_indices, tf.shape(inputs)[:-1])
        quantized_1nn = self.quantize(encoding_1nn_indices)


        normalized_inputs =  tf.nn.l2_normalize(flat_inputs, axis=1)
        normalized_w = tf.nn.l2_normalize(w, axis=0)
        e_multiply = tf.matmul(normalized_inputs, tf.stop_gradient(normalized_w)) / temperature

        flat_encodings = tf.reshape(encodings, [-1, self._num_embeddings])
        e_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(flat_encodings), logits=e_multiply))

        dema_cluster_size = tf.reduce_sum(encodings, 0)
        dw = tf.matmul(normalized_inputs, flat_encodings, transpose_a=True) 

        if is_training:
            updated_ema_cluster_size = moving_averages.assign_moving_average(self._ema_cluster_size, dema_cluster_size, decay, zero_debias=False)
            updated_ema_w = moving_averages.assign_moving_average(self._ema_w, dw, decay, zero_debias=False)
            normalised_updated_ema_w = (updated_ema_w / tf.reshape(updated_ema_cluster_size, [1, -1]))
            with tf.control_dependencies([e_loss]):
                update_w = tf.assign(self._w, normalised_updated_ema_w)
                with tf.control_dependencies([update_w]):
                    loss = tf.identity(e_loss)
        else:
            loss = e_loss

        return_dict = {'quantize_1nn': quantized_1nn,  # (-1,128)
                       'loss': loss,
                       'encodings': encodings,  # (-1,nc)
                       'encoding_indices': encoding_1nn_indices,  # (-1,)
                       'dema_cluster_size': dema_cluster_size,  # (nc,)
                       'dw': dw}  # (128,nc)

        return return_dict

    def update_w_from_list(self, tower_dema_cluster_size, tower_dw, decay):
        dema_cluster_size = tf.identity(tower_dema_cluster_size[0])
        dw = tf.identity(tower_dw[0])
        for i in range(1, len(tower_dema_cluster_size)):
            dema_cluster_size += tower_dema_cluster_size[i]
            dw += tower_dw[i]
        updated_ema_cluster_size = moving_averages.assign_moving_average(self._ema_cluster_size, dema_cluster_size, decay, zero_debias=False)
        updated_ema_w = moving_averages.assign_moving_average(self._ema_w, dw, decay, zero_debias=False)
        normalised_updated_ema_w = (updated_ema_w / tf.reshape(updated_ema_cluster_size, [1, -1]))
        return tf.assign(self._w, normalised_updated_ema_w)

    @property
    def embeddings(self):
        return self._w

    def quantize(self, encoding_indices):
        with tf.control_dependencies([encoding_indices]):
            w = tf.transpose(self.embeddings.read_value(), [1, 0])
        return tf.nn.embedding_lookup(w, encoding_indices, validate_indices=False)



class ConditionedBlock(base.AbstractModule):
    def __init__(self, latent_space_size, latent_dim_condition, latent_dim_src, latent_dim_bt, latent_dims_ff, \
                    hidden_act, name='conditioned_block'):
        super(ConditionedBlock, self).__init__(name=name)
        self._latent_space_size = latent_space_size
        self._latent_dim_condition = latent_dim_condition
        self._latent_dim_src = latent_dim_src
        self._latent_dim_bt = latent_dim_bt
        self._latent_dims_ff = latent_dims_ff
        self._hidden_act = hidden_act
        if self._hidden_act == 'relu':
            self._hidden_act_func = tf.nn.relu
        elif self._hidden_act == 'lrelu':
            self._hidden_act_func = lrelu_02
        
        self._name = name

    def _build(self, condition_code, src_code):
        c_code = tf.reshape(condition_code, [-1, self._latent_space_size])

        with tf.variable_scope('fc_zo'):                 
            c_code = tf.layers.dense(inputs=c_code, units=self._latent_dim_condition, activation=self._hidden_act_func, kernel_initializer=tf.contrib.layers.xavier_initializer())
            c_code = tf.nn.l2_normalize(c_code, axis=-1)
            

        s_code = src_code
        is_shared_scode = (src_code.shape.ndims == 2)
        ori_scode_shape = s_code.shape 
        s_code = tf.reshape(s_code, [-1, ori_scode_shape[-1]])

        with tf.variable_scope('fc_cp'):
            s_code= tf.layers.dense(inputs=s_code, units=self._latent_dim_src, activation=self._hidden_act_func, kernel_initializer=tf.contrib.layers.xavier_initializer())
            s_code = tf.nn.l2_normalize(s_code, axis=-1)

        if is_shared_scode:
            s_code = tf.transpose(s_code, [1, 0])
        else:
            s_code = tf.reshape(s_code, [-1, ori_scode_shape[1], s_code.shape[-1]])
            s_code = tf.transpose(s_code, [2, 1, 0])
            
        with tf.variable_scope('bilinear_transformation'):
            with tf.variable_scope('BilinearTransformation-0'):
                c_code_dim = c_code.shape[-1]
                s_code_dim = s_code.shape[0]

                bt_weight = tf.get_variable('weight', shape=[c_code_dim, self._latent_dim_bt, s_code_dim], initializer=tf.contrib.layers.xavier_initializer())
                
                mul_ccode_btw = tf.tensordot(c_code, bt_weight, axes=1)
                if is_shared_scode:
                    cs_code = tf.tensordot(mul_ccode_btw, s_code, axes=1)
                else:
                    cs_code = tf.einsum('aji,ika->ajk', mul_ccode_btw, s_code)
                
                cs_code = self._hidden_act_func(cs_code)
                
                cs_code = tf.transpose(cs_code, [0, 2, 1])
                cs_code = tf.nn.l2_normalize(cs_code,axis=-1)

        with tf.variable_scope('ff'):            
            cs_code_shape = cs_code.shape
            cs_code_residual = tf.reshape(cs_code, [-1, cs_code_shape[-1]])
            for id, h_dim in enumerate(self._latent_dims_ff):
                _act_func = None if id == len(self._latent_dims_ff) - 1 else self._hidden_act_func                    
                cs_code_residual = tf.layers.dense(inputs=cs_code_residual, units=h_dim, activation=_act_func, kernel_initializer=tf.contrib.layers.xavier_initializer())
            cs_code_residual = tf.reshape(cs_code_residual, [-1,cs_code_shape[1],cs_code_shape[-1]])
            cs_code = cs_code + cs_code_residual
        
        return cs_code






class ConditionedCodebook(base.AbstractModule):
    def __init__(self, embedding_dim, num_embeddings, func_f=None, name='position_encoding'):
        super(ConditionedCodebook, self).__init__(name=name)
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._func_f = func_f 

        with self._enter_variable_scope():
            initializer = tf.random_normal_initializer()
            self._w = tf.get_variable('embedding', [self._embedding_dim,self._num_embeddings], initializer=initializer, use_resource=True)

    def assign_w(self, sess, rotation_pe):
        rotation_pe_tensor = tf.placeholder(tf.float32, shape=[self._embedding_dim, self._num_embeddings])
        rotation_pe_assign_op = tf.assign(self._w, rotation_pe_tensor)
        sess.run(rotation_pe_assign_op, feed_dict={rotation_pe_tensor: rotation_pe})

    def conditioned_embedding_codebook(self, conditioned_code):
        ori_w = tf.stop_gradient(self._w.read_value())
        w = tf.transpose(self._func_f(conditioned_code, tf.transpose(ori_w, [1, 0])), [0, 2, 1])
        return w

    def _build(self, inputs, conditioned_code):
        with tf.control_dependencies([inputs, conditioned_code]):
            w = self.conditioned_embedding_codebook(conditioned_code)

        input_shape = tf.shape(inputs)
        # Flat inputs
        with tf.control_dependencies([tf.Assert(tf.equal(input_shape[-1], self._embedding_dim), [input_shape])]):
            flat_inputs = tf.reshape(inputs, [-1, 1, self._embedding_dim])

        normalized_inputs = tf.nn.l2_normalize(flat_inputs, axis=2)
        normalized_w = tf.nn.l2_normalize(w, axis=1)
        
        distances = -tf.matmul(normalized_inputs, normalized_w)
        distances = tf.reshape(distances, [-1, self._num_embeddings])
        encoding_1nn_indices = tf.argmax(-distances, 1)


        return_dict = {'encoding_indices': encoding_1nn_indices,} 
        
        return return_dict 


    @property
    def embeddings(self):
        return self._w



def generate_pose_position_encoding(quaternion_samples, max_n, beta_bias_deg=0., verbose=False):
    num_embeddings = quaternion_samples.shape[0]
    embedding_dim = 0
    for n in range(0, max_n + 1):
        embedding_dim += (n + 1) * (n + 2)

    pe = np.zeros((num_embeddings, embedding_dim))

    for sid in range(0, num_embeddings):
        cur_q = quaternion_samples[sid]
        euler = transform.euler_from_quaternion(cur_q, axes='szxz')
        phi = euler[0] if euler[0] > 0 else euler[0] + np.pi * 2  # [0,2pi]
        theta = euler[1] if euler[1] > 0 else euler[1] + np.pi * 2  # [0,pi]

        omega = euler[2] + (beta_bias_deg / 180.) * np.pi * 2
        if omega < 0:
            omega = omega + np.pi * 2
        if omega >= np.pi * 2:
            omega = omega - np.pi * 2
        beta = omega / 2

        peid = 0
        for cn in range(0, max_n + 1):
            for cl in range(0, cn + 1):
                term_2_exp = np.power(2, cl + 0.5)
                term_sqrt = (cn + 1) * gamma(cn - cl + 1) / (np.pi * gamma(cn + cl + 2))
                gegen_coeff = eval_gegenbauer(cn - cl, cl + 1, math.cos(beta))  # n,alpha,x
                term_sin = np.power(math.sin(beta), cl)
                p_nl = term_2_exp * (term_sqrt ** 0.5) * gamma(cl + 1) * term_sin * gegen_coeff

                for cm in range(0, cl + 1):
                    Y_ml = sph_harm(cm, cl, phi, theta)  # .real
                    pe[sid, peid] = p_nl * Y_ml.real
                    pe[sid, peid + 1] = p_nl * Y_ml.imag
                    peid += 2

    return pe


def compute_reconstruction_losses(dec_out, dec_gt, weight_branches):
    with tf.variable_scope('recons_loss'):
        if 'bgr' in weight_branches.keys():
            with tf.variable_scope('bgr'):
                bootstrap_ratio = 4

                gt_flat_bgr = tf.contrib.layers.flatten(dec_gt['bgr'])
                out_flat_bgr = tf.contrib.layers.flatten(dec_out['x_bgr'])

                l2_bgr = tf.losses.mean_squared_error(
                    out_flat_bgr,
                    gt_flat_bgr,
                    reduction=tf.losses.Reduction.NONE
                )
                l2_val_bgr, _ = tf.nn.top_k(l2_bgr, k=l2_bgr.shape[1] // bootstrap_ratio)
                loss_bgr = tf.reduce_mean(l2_val_bgr) * weight_branches['bgr']
        else:
            loss_bgr = tf.constant(0.)

        if 'depth' in weight_branches.keys():
            with tf.variable_scope('depth'):
                bootstrap_ratio = 4
                gt_flat_depth = tf.contrib.layers.flatten(dec_gt['depth'])
                out_flat_depth = tf.contrib.layers.flatten(dec_out['x_depth'])

                l2_depth = tf.losses.mean_squared_error(
                    out_flat_depth,
                    gt_flat_depth,
                    reduction=tf.losses.Reduction.NONE
                )
                l2_val_depth, _ = tf.nn.top_k(l2_depth, k=l2_depth.shape[1] // bootstrap_ratio)
                loss_depth = tf.reduce_mean(l2_val_depth) * weight_branches['depth']
        else:
            loss_depth = tf.constant(0.)

        loss=loss_bgr+loss_depth
        return {'reconst_loss': loss,
                'reconst_loss_bgr': loss_bgr,
                'reconst_loss_depth': loss_depth,}



