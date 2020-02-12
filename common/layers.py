import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils import tf_utils

from common import utils
from common.ops import ops as custom_ops
from common.ops import transformation
from common.ops.em_routing import em_routing
from common.ops.routing import dynamic_routing

eps = 1e-10


class Activation(Layer):
    def __init__(self,
                 activation='squash',
                 with_prob=False,
                 **kwargs):
        super(Activation, self).__init__(**kwargs)
        self.activation_fn = custom_ops.get_activation(activation)
        self.with_prob = with_prob

    def call(self, inputs, **kwargs):
        if self.activation_fn:
            pose, prob = self.activation_fn(inputs, axis=-1)
        else:
            pose, prob = inputs, None
        if self.with_prob:
            return pose, prob
        else:
            return pose


class PrimaryCapsule(Layer):
    def __init__(self,
                 kernel_size,
                 strides,
                 use_bias=False,
                 conv_caps=False,
                 padding='valid',
                 groups=32,
                 atoms=8,
                 activation='squash',
                 kernel_initializer=keras.initializers.glorot_normal(),
                 kernel_regularizer=None,
                 **kwargs):
        super(PrimaryCapsule, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.groups = groups
        self.atoms = atoms
        self.conv_caps = conv_caps
        self.activation_fn = custom_ops.get_activation(activation)
        self.conv = keras.layers.Conv2D(filters=self.groups * self.atoms,
                                        kernel_size=self.kernel_size,
                                        strides=self.strides,
                                        padding=self.padding,
                                        use_bias=use_bias,
                                        activation=None,
                                        kernel_initializer=kernel_initializer,
                                        kernel_regularizer=kernel_regularizer)

    def call(self, inputs, **kwargs):
        pose = self.conv(inputs)
        pose_shape = pose.get_shape().as_list()
        if self.conv_caps:
            pose = tf.reshape(pose, shape=[-1, pose_shape[1], pose_shape[2], self.groups, self.atoms])
        else:
            pose = tf.reshape(pose, shape=[-1, pose_shape[1]*pose_shape[2]*self.groups, self.atoms])
        if self.activation_fn:
            pose, prob = self.activation_fn(pose, axis=-1)
            return pose, prob
        else:
            return pose


class CapsuleTransformDense(Layer):
    def __init__(self,
                 num_out,
                 out_atom,
                 share_weights=False,
                 matrix=False,
                 initializer=keras.initializers.glorot_normal(),
                 regularizer=None,
                 **kwargs):
        super(CapsuleTransformDense, self).__init__(**kwargs)
        self.num_out = num_out
        self.out_atom = out_atom
        self.share_weights = share_weights
        self.matrix = matrix
        self.wide = None
        self.initializer = initializer
        self.regularizer = regularizer

    def build(self, input_shape):
        in_atom = input_shape[-1]
        in_num = input_shape[-2]
        if self.matrix:
            self.wide = int(np.sqrt(in_atom))
        if self.share_weights:
            if self.wide:
                self.kernel = self.add_weight(
                    name='capsule_kernel',
                    shape=(1, self.num_out, self.wide, self.wide),
                    initializer=self.initializer,
                    regularizer=self.regularizer,
                    trainable=True)
                self.kernel = tf.tile(self.kernel, [in_num, 1, 1, 1])
            else:
                self.kernel = self.add_weight(
                    name='capsule_kernel',
                    shape=(1, in_atom,
                           self.num_out * self.out_atom),
                    initializer=self.initializer,
                    regularizer=self.regularizer,
                    trainable=True)
                self.kernel = tf.tile(self.kernel, [in_num, 1, 1])
        else:
            if self.wide:
                self.kernel = self.add_weight(
                    name='capsule_kernel',
                    shape=(in_num, self.num_out, self.wide, self.wide),
                    initializer=self.initializer,
                    regularizer=self.regularizer,
                    trainable=True)
            else:
                self.kernel = self.add_weight(
                    name='capsule_kernel',
                    shape=(in_num, in_atom,
                           self.num_out * self.out_atom),
                    initializer=self.initializer,
                    regularizer=self.regularizer,
                    trainable=True)

    def call(self, inputs, **kwargs):
        in_shape = inputs.get_shape().as_list()
        in_shape[0] = -1
        if self.wide:
            # [bs, in_num, in_atom] -> [bs, in_num, wide, wide]
            inputs = tf.reshape(inputs, in_shape[:-1]+[self.wide, self.wide])
            # [bs, in_num, a, b]  X  [in_num, out_num, b, c]
            # -> [bs, in_num, out_num, a, c]
            outputs = transformation.matrix_capsule_element_wise(inputs, self.kernel, self.num_out)
            outputs = tf.reshape(outputs, in_shape[:-1] + [self.num_out] + [in_shape[-1]])
        else:
            # [bs, in_num, in_atom] X [in_num, in_atom, out_num*out_atom]
            #  -> [bs, in_num, out_num, out_atom]
            outputs = transformation.matmul_element_wise(inputs, self.kernel, self.num_out, self.out_atom)
        return outputs


class CapsuleTransformConv(Layer):
    def __init__(self,
                 kernel_size,
                 stride,
                 filter,
                 atom,
                 initializer=keras.initializers.glorot_normal(),
                 regularizer=None,
                 **kwargs):
        super(CapsuleTransformConv, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.filter = filter
        self.atom = atom
        self.initializer = initializer
        self.regularizer = regularizer

    def build(self, input_shape):
        # inputs [bs, height, width, channel, in_atom]
        in_channel = input_shape[-2]
        in_atom = input_shape[-1]

        self.matrix = self.add_weight(
            name='capsule_kernel',
            shape=(self.kernel_size*self.kernel_size*in_channel, in_atom,
                   self.filter*self.atom),
            initializer=self.initializer,
            regularizer=self.regularizer,
            trainable=True)

    def call(self, inputs, **kwargs):
        # inputs [bs, height, width, channel, in_atom]
        inputs_tile, _ = utils.kernel_tile(inputs, self.kernel_size, self.stride)
        # tile [bs, out_height, out_width, kernel*kernel*channel, in_atom]
        outputs = transformation.matmul_element_wise(inputs_tile, self.matrix, self.filter, self.atom)
        # [bs, out_height, out_width, kernel*kernel*channel, out_num, out_atom]
        return outputs


class CapsuleGroups(Layer):
    def __init__(self,
                 height,
                 width,
                 channel,
                 atoms,
                 activation=None,
                 **kwargs):
        super(CapsuleGroups, self).__init__(**kwargs)
        self.height = height
        self.width = width
        self.channel = channel
        self.atoms = atoms
        self.activation_fn = custom_ops.get_activation(activation)

    def call(self, inputs, **kwargs):
        group_num = self.channel // self.atoms
        vote = tf.reshape(inputs, shape=[-1, self.height * self.width, group_num, self.atoms])
        if self.activation_fn:
            vote, _ = self.activation_fn(vote)
        return vote


class RoutingPooling(Layer):
    def __init__(self,
                 kernel_size,
                 strides,
                 atoms,
                 in_norm=True,
                 num_routing=2,
                 temper=1.0,
                 **kwargs):
        super(RoutingPooling, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.strides = strides
        self.atoms = atoms
        self.in_norm = in_norm
        self.num_routing = num_routing
        self.temper = temper

    def call(self, inputs, **kwargs):
        patched = batch_2d(inputs, self.kernel_size, self.strides)
        patched_shape = patched.get_shape().as_list()
        patched = tf.reshape(patched,
                             [-1] + patched_shape[1:4] + [patched_shape[4] // self.atoms, self.atoms])
        patched = tf.transpose(patched, perm=[0, 1, 2, 4, 3, 5])
        patched = tf.expand_dims(patched, axis=-2)

        pose, _ = dynamic_routing(patched,
                                  num_routing=self.num_routing,
                                  softmax_in=True,
                                  temper=self.temper,
                                  activation='norm')
        pose = tf.reshape(pose, [-1] + patched_shape[1:3] + [patched_shape[4]])
        return pose


class FMPooling(Layer):
    def __init__(self,
                 kernel_size,
                 strides,
                 fm_norm,
                 atoms,
                 out_norm=False,
                 **kwargs):
        super(FMPooling, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.strides = strides
        self.fm_norm = fm_norm
        self.atoms = atoms
        self.out_norm = out_norm

    def call(self, inputs, **kwargs):
        patched = batch_2d(inputs, self.kernel_size, self.strides)
        patched_shape = patched.get_shape().as_list()
        patched = tf.reshape(patched,
                             [-1] + patched_shape[1:4] + [patched_shape[4] // self.atoms, self.atoms])
        patched = tf.transpose(patched, perm=[0, 1, 2, 4, 3, 5])

        pool = get_factorization_machines(patched, axis=-2)
        pool = tf.reshape(pool, [-1] + patched_shape[1:3] + [patched_shape[4]])
        if self.out_norm:
            pool, _ = custom_ops.vector_norm(pool)
        return pool


class DynamicRouting(Layer):
    def __init__(self,
                 num_routing=3,
                 softmax_in=False,
                 temper=1.0,
                 activation='squash',
                 pooling=False,
                 log=None,
                 **kwargs):
        super(DynamicRouting, self).__init__(**kwargs)
        self.num_routing = num_routing
        self.softmax_in = softmax_in
        self.temper = temper
        self.activation = activation
        self.pooling = pooling
        self.log = log

    def call(self, inputs, **kwargs):
        if self.pooling:
            inputs = tf.expand_dims(inputs, -2)
        pose, prob = dynamic_routing(inputs,
                                     num_routing=self.num_routing,
                                     softmax_in=self.softmax_in,
                                     temper=self.temper,
                                     activation=self.activation)
        pose = tf.squeeze(pose, axis=-3)
        prob = tf.squeeze(prob, axis=[-3, -1])
        return pose, prob


class EMRouting(Layer):
    def __init__(self,
                 num_routing=3,
                 log=None,
                 **kwargs):
        super(EMRouting, self).__init__(**kwargs)
        self.num_routing = num_routing
        self.log = log

    def build(self, input_shape):
        # ----- Betas -----#

        """
        # Initialization from Jonathan Hui [1]:
        beta_v_hui = tf.get_variable(
          name='beta_v',
          shape=[1, 1, 1, o],
          dtype=tf.float32,
          initializer=tf.contrib.layers.xavier_initializer())
        beta_a_hui = tf.get_variable(
          name='beta_a',
          shape=[1, 1, 1, o],
          dtype=tf.float32,
          initializer=tf.contrib.layers.xavier_initializer())

        # AG 21/11/2018:
        # Tried to find std according to Hinton's comments on OpenReview
        # https://openreview.net/forum?id=HJWLfGWRb&noteId=r1lQjCAChm
        # Hinton: "We used truncated_normal_initializer and set the std so that at the
        # start of training half of the capsules in each layer are active and half
        # inactive (for the Primary Capsule layer where the activation is not computed
        # through routing we use different std for activation convolution weights &
        # for pose parameter convolution weights)."
        #
        # std beta_v seems to control the spread of activations
        # To try and achieve what Hinton said about half active and half not active,
        # I change the std values and check the histogram/distributions in
        # Tensorboard
        # to try and get a good spread across all values. I couldn't get this working
        # nicely.
        beta_v_hui = slim.model_variable(
          name='beta_v',
          shape=[1, 1, 1, 1, o, 1],
          dtype=tf.float32,
          initializer=tf.truncated_normal_initializer(mean=0.0, stddev=10.0))
        """
        o = input_shape[0].as_list()[-2]  # out caps
        self.beta_a = self.add_weight(name='beta_a',
                                      shape=[1, 1, o, 1],
                                      dtype=tf.float32,
                                      initializer=tf.keras.initializers.TruncatedNormal(mean=-1000.0, stddev=500.0))

        # AG 04/10/2018: using slim.variable to create instead of tf.get_variable so
        # that they get correctly placed on the CPU instead of GPU in the multi-gpu
        # version.
        # One beta per output capsule type
        # (N, i, o, atom)
        self.beta_v = self.add_weight(name='beta_v',
                                      shape=[1, 1, o, 1],
                                      dtype=tf.float32,
                                      initializer=tf.keras.initializers.GlorotNormal(),
                                      regularizer=None)

    def call(self, inputs, **kwargs):
        # votes (bs, in, out, atom)
        # activations (bs, in, 1)
        votes_flat, activation_flat = inputs
        pose, prob = em_routing(votes_flat,
                                activation_flat,
                                self.beta_a,
                                self.beta_v,
                                self.num_routing,
                                final_lambda=0.01,
                                epsilon=1e-9,
                                spatial_routing_matrix=[[1]])
        prob = tf.squeeze(prob, axis=[-1])
        return pose, prob


class Decoder(Layer):
    def __init__(self,
                 height,
                 width,
                 channel,
                 balance_factor,
                 layers=[512, 1024],
                 **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.height = height
        self.width = width
        self.channel = channel
        self.balance_factor = balance_factor
        self.layers = []
        for layer in layers:
            self.layers.append(keras.layers.Dense(layer, tf.nn.relu))
        self.layers.append(keras.layers.Dense(self.height*self.width*self.channel, tf.sigmoid))

    def call(self, inputs, **kwargs):
        active_caps, images = inputs
        for layer in self.layers:
            active_caps = layer(active_caps)
        recons = active_caps
        recons_img = tf.reshape(recons, [-1, self.height, self.width, self.channel])
        distance = tf.pow(recons_img - images, 2)
        loss = tf.reduce_sum(distance, [-1, -2, -3])
        recons_loss = self.balance_factor * tf.reduce_mean(loss)
        self.add_loss(recons_loss)
        return recons_loss, recons_img


class Mask(Layer):
    def __init__(self,
                 order,
                 share=True,
                 out_num=10,
                 **kwargs):
        super(Mask, self).__init__(**kwargs)
        self.order = order
        self.share = share
        self.out_num = out_num

    def call(self, inputs, **kwargs):
        poses, probs, labels = inputs
        if len(labels.get_shape()) != len(probs.get_shape()):
            labels = tf.one_hot(labels, probs.get_shape().as_list()[-1])

        def inference():
            if self.order > 0:
                _, top_k = tf.nn.top_k(probs, self.order + 1)
                split = tf.split(top_k, self.order + 1, -1)
                split = split[-1]
                predictions = tf.expand_dims(tf.one_hot(tf.squeeze(split, -1), self.out_num), -1)
            else:
                predictions = tf.expand_dims(tf.one_hot(tf.argmax(probs, -1), self.out_num), -1)
            return predictions

        training = keras.backend.learning_phase()
        mask = tf_utils.smart_cond(training, lambda: tf.expand_dims(labels, -1), inference)
        masked_caps = tf.multiply(poses, mask)
        if self.share:
            active_caps = tf.reduce_sum(masked_caps, axis=-2)
        else:
            active_caps = keras.layers.Flatten()(masked_caps)
        return active_caps


class DecoderConv(Layer):
    def __init__(self,
                 height,
                 width,
                 channel,
                 balance_factor,
                 base=10,
                 filter=64,
                 kernel_initializer=keras.initializers.he_normal(),
                 kernel_regularizer=keras.regularizers.l2(1e-4),
                 **kwargs):
        super(DecoderConv, self).__init__(**kwargs)
        self.height = height
        self.width = width
        self.channel = channel
        self.balance_factor = balance_factor
        self.layers = [keras.layers.Dense(base*base*filter),
                       keras.layers.BatchNormalization(),
                       keras.layers.LeakyReLU(),
                       keras.layers.Reshape((base, base, filter)),
                       keras.layers.Conv2DTranspose(filter//2, (5, 5), strides=(1, 1), padding='same',
                                                    kernel_initializer=kernel_initializer,
                                                    kernel_regularizer=kernel_regularizer),
                       keras.layers.BatchNormalization(),
                       keras.layers.LeakyReLU(),
                       keras.layers.Conv2DTranspose(filter//4, (5, 5), strides=(2, 2), padding='same',
                                                    kernel_initializer=kernel_initializer,
                                                    kernel_regularizer=kernel_regularizer),
                       keras.layers.BatchNormalization(),
                       keras.layers.LeakyReLU(),
                       keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation=tf.sigmoid,
                                                    kernel_initializer=kernel_initializer,
                                                    kernel_regularizer=kernel_regularizer),
                       ]

    def call(self, inputs, **kwargs):
        active_caps, images = inputs
        for layer in self.layers:
            active_caps = layer(active_caps)
        recons_img = active_caps
        distance = tf.pow(recons_img - images, 2)
        loss = tf.reduce_sum(distance, [-1, -2, -3])
        recons_loss = self.balance_factor * tf.reduce_mean(loss)
        self.add_loss(recons_loss)
        return recons_loss, recons_img


class VectorNorm(Layer):
    def __init__(self, **kwargs):
        super(VectorNorm, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        x = custom_ops.vector_norm(inputs, -1)
        return x


class LastFMPool(Layer):
    def __init__(self,
                 axis=-2,
                 activation='accumulate',
                 shrink=None,
                 stable=False,
                 norm_pose=True,
                 log=None,
                 regularize=True,
                 **kwargs):
        super(LastFMPool, self).__init__(**kwargs)
        self.axis = axis
        self.activation = activation
        self.shrink = shrink
        self.stable = stable
        self.norm_pose = norm_pose
        self.log = log
        self.regularize = regularize

    def call(self, inputs, **kwargs):
        # [bs, caps_in, caps_out, atom]
        outputs, importance = get_factorization_machines(inputs,
                                                         self.axis,
                                                         regularize=self.regularize)
        if self.log:
            if importance:
                self.log.add_hist('importance', importance)
            self.log.add_hist('fm_out', outputs)
            self.log.add_hist('fm_similarity', tf.reduce_sum(outputs, axis=-1))
        if self.activation == 'accumulate':
            outputs, norm = custom_ops.accumulate(outputs, shrink=self.shrink, stable=self.stable, norm_pose=self.norm_pose)
            norm = tf.squeeze(norm, -1)
            return outputs, norm
        elif self.activation == 'squash':
            outputs, norm = custom_ops.squash(outputs)
            norm = tf.squeeze(norm, -1)
            return outputs, norm
        elif self.activation == 'norm':
            outputs, _ = custom_ops.vector_norm(outputs)
            return outputs
        else:
            return outputs


class LastAveragePooling(Layer):
    def __init__(self,
                 axis=-2,
                 **kwargs):
        super(LastAveragePooling, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs, **kwargs):
        outputs = tf.reduce_mean(inputs, axis=self.axis)
        return outputs


class LastMaxPooling(Layer):
    def __init__(self,
                 axis=-2,
                 **kwargs):
        super(LastMaxPooling, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs, **kwargs):
        outputs = tf.reduce_max(inputs, axis=self.axis)
        return outputs


def get_original_capsule_layer(inputs, num_out, out_atom, num_routing=3, temper=1.0):
    transformed = CapsuleTransformDense(num_out=num_out, out_atom=out_atom, share_weights=False)(inputs)
    routed = DynamicRouting(num_routing=num_routing, temper=temper)(transformed)
    return routed


def get_factorization_machines(inputs, axis=-2, regularize=True):
    # [bs, caps_in, caps_out, atom]
    if regularize:
        cap_in = inputs.get_shape()[1]
        inputs /= np.sqrt(cap_in)
    x1 = tf.reduce_sum(inputs, axis, keepdims=True)  # [bs, 1, caps_out, atom]
    x1 = tf.square(x1)
    x2_square = tf.square(inputs)
    x2 = tf.reduce_sum(x2_square, axis, keepdims=True)  # [bs, 1, caps_out, atom]
    outputs = x1 - x2
    outputs = tf.squeeze(outputs, axis)
    weight = None
    return outputs, weight


def get_average_pooling(inputs):
    x = tf.reduce_mean(inputs, axis=-2)
    return


def get_cross_mul(inputs):
    n = inputs.get_shape().as_list()[-2]
    outputs = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                outputs += inputs[i] * inputs[j]
    outputs /= (2 * n)
    return outputs


def batch_2d(inputs, kernel_size, strides, name=None):
    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, kernel_size)
    if not isinstance(strides, tuple):
        strides = (strides, strides)
    name = "batch_to_pool" if name is None else name
    with tf.name_scope(name):
        height, width = inputs.get_shape().as_list()[1:3]
        h_offsets = [[(h + k) for k in range(0, kernel_size[0])] for h in range(0, height + 1 - kernel_size[0], strides[0])]
        w_offsets = [[(w + k) for k in range(0, kernel_size[1])] for w in range(0, width + 1 - kernel_size[1], strides[1])]
        patched = tf.gather(inputs, h_offsets, axis=1)
        patched = tf.gather(patched, w_offsets, axis=3)
        perm = [0, 1, 3, 2, 4, 5]
        patched = tf.transpose(patched, perm=perm)
        shape = patched.get_shape().as_list()
        shape = [-1] + shape[1:3] + [np.prod(shape[3:-1]), shape[-1]]
        patched = tf.reshape(patched, shape=shape)
    return patched


def test_routing_pool():
    x = tf.random.normal([64, 16, 16, 64])
    x = RoutingPooling(3, 2, 8)(x)
    print(x.shape)


def test_batch_2d():
    x = tf.random.normal([128, 32, 32, 3])
    x = batch_2d(x, 2, 2)
    print(x.shape)


def verify_factorization_machines():
    x = tf.random.normal([1000, 16])
    t1 = time.time()
    out1 = get_cross_mul(x)
    t2 = time.time()
    print('cross mul cost:', t2 - t1)
    print('cross mul result:', out1)
    out2 = get_factorization_machines(x)
    t3 = time.time()
    print('FM cost:', t3 - t2)
    print('FM result:', out2)


def test_ablation_fm():
    num = 1000
    atom = 16
    a = tf.random.normal([10000, num, atom], 2, 20)
    a = a / tf.norm(a, axis=-1, keepdims=True)
    a = a / tf.sqrt(tf.cast(num, tf.float32))
    b1 = tf.square(tf.reduce_sum(a, 1))
    # b1_mean, b1_var = tf.nn.moments(b1, 0)
    # print('b1_mean:', b1_mean.numpy())
    # print('b1_var:', b1_var.numpy())
    b2 = tf.reduce_sum(tf.square(a), 1)
    # b2_mean, b2_var = tf.nn.moments(b2, 0)
    # print('b2_mean:', b2_mean.numpy())
    # print('b2_var:', b2_var.numpy())
    fm = b1 - b2
    b3_mean, b3_var = tf.nn.moments(fm, 0)
    print('b3_mean:', b3_mean.numpy())
    print('b3_var:', b3_var.numpy())
    act = tf.reduce_sum(fm, 1)
    act_mean, act_var = tf.nn.moments(act, 0)
    print('act_mean:', act_mean.numpy())
    print('act_var:', act_var.numpy())


def verify_vec_norm():
    vec_norm = VectorNorm()
    x = tf.random.normal([64, 16])
    x = vec_norm(x)
    print(tf.norm(x, axis=-1))


def verify_average_pooling():
    x = tf.random.normal([10, 8, 8, 64])
    x1 = keras.layers.AveragePooling2D([8, 8])(x)
    x1 = tf.squeeze(x1)
    x = tf.reshape(x, [10, 64, 64])
    x2 = LastAveragePooling()(x)
    print(tf.reduce_sum(x1-x2))


def verify_pri_capsule():
    x = tf.random.normal([10, 8, 8, 64])
    x = CapsuleGroups(height=8,
                      width=8,
                      channel=64,
                      atoms=32,
                      activation='norm')(x)
    print(tf.norm(x, axis=-1))


def verify_dynamic_routing():
    x = tf.random.normal([10, 2, 64, 1, 32])
    y = DynamicRouting()(x)
    print(y)


def verify_fm_pool():
    x = tf.random.normal([128, 32, 32, 64])
    y = FMPooling(3, 2, 8)(x)
    print(y)


def verify_last_fm_pool():
    x = tf.random.normal([128, 1152, 16])
    x_mean, x_var = tf.nn.moments(x, [0,1,2])
    y = LastFMPool(activation=None)(x)
    y_mean, y_var = tf.nn.moments(y, [0,1])
    print(y)
    print(y_var)


def verify_transform():
    x = tf.random.normal((128, 30, 8))
    trans = CapsuleTransformDense(10, 16)(x)
    print(trans)


def var_experiments():
    x = tf.random.normal([10240, 64])
    x1_mean, x1_var = tf.nn.moments(x, [0, 1])
    x2_mean, x2_var = tf.nn.moments(x*x, [0, 1])
    x4_mean = tf.reduce_mean(x*x*x*x)
    print('x1_var', x1_var)
    print('x2_var', x2_var)
    print('x4_mean', x4_mean)
    y = get_factorization_machines(x, 1)
    y_mean, y_var = tf.nn.moments(y, [0])
    print('y_mean', y_mean)
    print('y_var', y_var)


def var_vec_norm_scale():
    x = tf.random.normal([12800, 160])
    x_norm, _ = custom_ops.vector_norm_scale(x, -1)
    x_norm_verify = tf.norm(x_norm, axis=-1)
    y_mean, y_var = tf.nn.moments(x_norm, [0, 1])
    print('E:',y_mean.numpy(), 'D:', y_var.numpy())


if __name__ == "__main__":
    # verify_factorization_machines()
    # verify_last_fm_pool()
    # var_experiments()
    # verify_dynamic_routing()
    # verify_transform()
    # test_batch_2d()
    # test_routing_pool()
    # verify_fm_pool()
    var_vec_norm_scale()
