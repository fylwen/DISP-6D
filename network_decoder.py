
"""Network architectures used in the StyleGAN paper."""

import numpy as np
import sonnet as snt
import tensorflow as tf

# NOTE: Do not import any application-specific modules here!
# Specify all network parameters as kwargs.

#----------------------------------------------------------------------------
# Primitive ops for manipulating 4D activation tensors.
# The gradients of these are not necessary efficient or even meaningful.

#----------------------------------------------------------------------------
# Get/create weight tensor for a convolutional or fully-connected layer.

def get_weight(shape, gain=np.sqrt(2), use_wscale=False, lrmul=1):
    fan_in = np.prod(shape[:-1]) # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    he_std = gain / np.sqrt(fan_in) # He init

    # Equalized learning rate and custom learning rate multiplier.
    if use_wscale:
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
    else:
        init_std = he_std / lrmul
        runtime_coef = lrmul

    # Create variable.
    init = tf.initializers.random_normal(0, init_std)
    return tf.get_variable('weight', shape=shape, initializer=init) * runtime_coef
#----------------------------------------------------------------------------
# Fully-connected layer.
def dense(x, fmaps, **kwargs):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], **kwargs)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)
#----------------------------------------------------------------------------
# Convolutional layer.
def conv2d(x, fmaps, kernel, **kwargs):
    assert kernel >= 1 and kernel % 2 == 1
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], **kwargs)
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME', data_format='NCHW')


def apply_bias(x, lrmul=1): #NCHW
    b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros()) * lrmul
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + b
    return x + tf.reshape(b, [1, -1, 1, 1])

#----------------------------------------------------------------------------
# Leaky ReLU activation. More efficient than tf.nn.leaky_relu() and supports FP16.
def leaky_relu(x, alpha=0.2):
    with tf.variable_scope('LeakyReLU'):
        alpha = tf.constant(alpha, dtype=x.dtype, name='alpha')
        @tf.custom_gradient
        def func(x):
            y = tf.maximum(x, x * alpha)
            @tf.custom_gradient
            def grad(dy):
                dx = tf.where(y >= 0, dy, dy * alpha)
                return dx, lambda ddx: tf.where(y >= 0, ddx, ddx * alpha)
            return y, grad
        return func(x)
#----------------------------------------------------------------------------
def instance_norm(x, epsilon=1e-8):
    assert len(x.shape) == 4 # NCHW
    with tf.variable_scope('InstanceNorm'):
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float32)
        x -= tf.reduce_mean(x, axis=[2,3], keepdims=True)
        epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
        x *= tf.rsqrt(tf.reduce_mean(tf.square(x), axis=[2,3], keepdims=True) + epsilon)
        x = tf.cast(x, orig_dtype)
        return x

def layer_norm(x, epsilon=1e-8):
    assert len(x.shape) == 2 # NC
    with tf.variable_scope('LayerNorm'):
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float32)

        x -= tf.reduce_mean(x, axis=1, keepdims=True)
        print('Layer norm', tf.reduce_mean(x, axis=1, keepdims=True).shape,'//',tf.reduce_mean(tf.square(x), axis=1, keepdims=True).shape)
        epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
        x *= tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)
        x = tf.cast(x, orig_dtype)
        return x



#----------------------------------------------------------------------------
# Style modulation.
def style_mod(x, dlatent, **kwargs):
    with tf.variable_scope('StyleMod'):
        style = apply_bias(dense(dlatent, fmaps=x.shape[1]*2, gain=1, **kwargs))
        style = tf.reshape(style, [-1, 2, x.shape[1]] + [1] * (len(x.shape) - 2))
        return x * (style[:,0] + 1) + style[:,1]

# Things to do at the end of each layer.
def layer_epilogue(x, dlatent, nonlinearty, use_wscale=True):
    x = apply_bias(x)
    act, gain = {'relu': (tf.nn.relu, np.sqrt(2)), 'lrelu': (leaky_relu, np.sqrt(2))}[nonlinearty]#['relu']
    x = act(x)
    #if use_pixel_norm:
    #    x = pixel_norm(x)
    #if use_instance_norm:
    x = instance_norm(x)
    #if use_styles:
    x = style_mod(x, dlatent, use_wscale=use_wscale)
    return x


#----------------------------------------------------------------------------
class AdaptiveDecoder(snt.AbstractModule):
    def __init__(self, reconstruction_shape, num_filters, kernel_size, strides,
                hidden_act, final_act, branches, name='decoder'):
        super(AdaptiveDecoder, self).__init__(name=name)
        self._reconstruction_shape=reconstruction_shape
       
        self._num_filters = num_filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._hidden_act=hidden_act


        if final_act=='sigmoid':
            self._final_act_func=tf.nn.sigmoid
        elif final_act=='none':
            self._final_act_func=None

        self._branches=branches
        
    def branch(self, pose_code, obj_code, branch_type, final_act_func, num_out_channels):
        z = pose_code
        h, w, _ = self._reconstruction_shape
        layer_dimensions = [ [h//np.prod(self._strides[i:]), w//np.prod(self._strides[i:])]  for i in range(len(self._strides))]

        block_id=0
        with tf.variable_scope('decoder_{0}'.format(branch_type)):
            #FC->reshape->NCHW->AdaIN
            x = tf.layers.dense(
                inputs=z,
                units= layer_dimensions[0][0]*layer_dimensions[0][1]*self._num_filters[0],
                activation=None, 
                use_bias=False,
                kernel_initializer=tf.contrib.layers.xavier_initializer()
            )
            
            x = tf.reshape(x, [-1, layer_dimensions[0][0], layer_dimensions[0][1], self._num_filters[0]])
            x = layer_epilogue(tf.transpose(x, [0,3,1,2]),obj_code,self._hidden_act) # NHWC->NCHW

            for filters, layer_size in zip(self._num_filters[1:],layer_dimensions[1:]):
                with tf.variable_scope('block_{:d}'.format(block_id)):
                    #NHWC->Size x2->NCHW->Conv->AdaIN
                    x=tf.transpose(x,[0,2,3,1])#NCHW->NHCW
                    x = tf.image.resize_nearest_neighbor(x, layer_size)

                    x = tf.transpose(x,[0,3,1,2])#NHWC->NCHW
                    x = layer_epilogue(conv2d(x=x,fmaps=filters,kernel=self._kernel_size),obj_code,self._hidden_act, use_wscale=True)

                    block_id+=1

            #NHWC->Size x2->Conv
            x = tf.transpose(x, [0, 2, 3, 1])  # NCHW->NHCW
            x = tf.image.resize_nearest_neighbor(x,[h, w])
            outx = tf.layers.conv2d(
                    inputs=x,
                    filters=num_out_channels,
                    kernel_size=self._kernel_size,
                    padding='same',
                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                    activation=final_act_func)

            return outx



    def _build(self, pose_code, obj_code,is_training):
        return_dict={}
        if 'bgr' in self._branches:
            return_dict['x_bgr']=self.branch(pose_code, obj_code, 'bgr', self._final_act_func, 3)

        if 'depth' in self._branches:
            return_dict['x_depth']=self.branch(pose_code,obj_code,'depth', self._final_act_func, 1)

        return return_dict