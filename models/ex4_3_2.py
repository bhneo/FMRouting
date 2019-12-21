import os
import sys

sys.path.append(os.getcwd())

import tensorflow as tf
from tensorflow import keras

from common import res_blocks, layers, utils
from config import params as cfg

WEIGHT_DECAY = 1e-4
BATCH_NORM_EPSILON = 1e-3
BATCH_NORM_DECAY = 0.99

kernel_regularizer = keras.regularizers.l2(WEIGHT_DECAY)
kernel_initializer = keras.initializers.he_normal()


def build_model_name(params):
    model_name = '_'.join(['Ex4_3_2', params.model.pool])

    model_name += '_atom{}'.format(params.caps.atoms)

    model_name += '_factor{}'.format(params.recons.balance_factor)

    if params.recons.conv:
        model_name += '_conv'

    if params.recons.share:
        model_name += '_shareCaps'

    model_name += '_bs{}'.format(str(params.training.batch_size))
    model_name += '_trial{}'.format(str(params.training.idx))
    return model_name


def build_model(shape, num_out, params):
    # optimizer = keras.optimizers.SGD(learning_rate=keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001,
    #                                                                                            decay_steps=5000,
    #                                                                                            decay_rate=0.5), momentum=0.9)  # 3e-3 20000 0.96
    optimizer = keras.optimizers.Adam(0.0001)
    inputs = keras.Input(shape=shape)
    model_name = build_model_name(params)
    model_log = utils.TensorLog()
    pose, prob = build_encoder(inputs, num_out, params.caps.atoms, model_log)
    encoder = keras.Model(inputs=inputs, outputs=(pose, prob), name='encoder')
    encoder.compile(optimizer=optimizer, metrics=[])
    encoder.summary()

    labels = keras.Input(shape=(), dtype=tf.int32)
    in_pose = keras.Input(shape=(num_out, cfg.caps.atoms))
    in_prob = keras.Input(shape=(num_out,))
    inputs_shape = inputs.get_shape().as_list()
    active_cap = layers.Mask(order=0, share=cfg.recons.share, out_num=num_out)((in_pose, in_prob, labels))
    if cfg.recons.conv:
        decoder_layer = layers.DecoderConv(height=inputs_shape[1], width=inputs_shape[2], channel=inputs_shape[3],
                                           balance_factor=params.recons.balance_factor,
                                           base=8, filter=128)
    else:
        decoder_layer = layers.Decoder(height=inputs_shape[1], width=inputs_shape[2], channel=inputs_shape[3],
                                       balance_factor=params.recons.balance_factor,
                                       layers=[512, 1024])

    recons_loss, recons_img = decoder_layer((active_cap, inputs))
    decoder = keras.Model(inputs=(in_pose, in_prob, inputs, labels), outputs=recons_img, name='decoder')
    decoder.compile(optimizer=optimizer, metrics=[])
    decoder.summary()

    active_cap = layers.Mask(order=0, share=cfg.recons.share, out_num=num_out)((pose, prob, labels))
    recons_loss, recons_img = decoder_layer((active_cap, inputs))
    model_log.add_scalar('reconstruction_loss', recons_loss)
    image_out = tf.concat([inputs, recons_img], 1)
    model_log.add_image('recons_img', image_out)

    model = keras.Model(inputs=(inputs, labels), outputs=(prob, recons_img), name=model_name)
    model.compile(optimizer=optimizer,
                  # loss=losses.MarginLoss(True, 0.9, 0.1, 0.5),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[])
    model.summary()
    # lr_scheduler = keras.callbacks.LearningRateScheduler(schedule=lr_scheduler)
    # lr_scheduler.set_model(model)
    # callbacks = [lr_scheduler]
    model.callbacks = []

    log_model = keras.Model(inputs=(inputs, labels), outputs=model_log.get_outputs(), name='model_log')
    model_log.set_model(log_model)

    return model, model_log, encoder, decoder


def build_encoder(inputs, num_out, atoms, log):
    backbone = res_blocks.build_resnet_backbone(inputs=inputs,
                                                layer_num=0, repetitions=[3, 3, 3],
                                                start_filters=16, arch='cifar',
                                                use_bias=False,
                                                kernel_initializer=kernel_initializer,
                                                kernel_regularizer=kernel_regularizer,
                                                bn_axis=-1, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON,
                                                version='v2')
    log.add_hist('backbone', backbone)
    pri_caps = layers.PrimaryCapsule(kernel_size=5, strides=2, padding='same',
                                     groups=4, atoms=atoms,
                                     activation=None,
                                     kernel_initializer=kernel_initializer,
                                     kernel_regularizer=kernel_regularizer)(backbone)

    pri_caps = keras.layers.BatchNormalization(axis=[1, 2])(pri_caps)

    poses, probs = multi_caps_layer(pri_caps, [num_out], log)

    return poses, probs


def multi_caps_layer(inputs, out_caps, log):
    # inputs [bs, caps_in, atoms]
    poses, probs = layers.Activation('squash', with_prob=True)(inputs)
    for i, out_num in enumerate(out_caps):
        prediction_caps = layers.CapsuleTransformDense(num_out=out_num, matrix=True, out_atom=0,
                                                       share_weights=False,
                                                       regularizer=kernel_regularizer)(poses)
        prediction_caps = keras.layers.BatchNormalization(axis=[1, 2, 3])(prediction_caps)
        log.add_hist('prediction_caps{}'.format(i+1), prediction_caps)

        prediction_caps = layers.Activation('norm')(prediction_caps)
        poses, probs = layers.LastFMPool(axis=-3, activation='accumulate',
                                         shrink=False, stable=False, regularize=True,
                                         norm_pose=True if i==len(out_caps)-1 else False,
                                         log=None)(prediction_caps)

        log.add_hist('prob{}'.format(i+1), probs)
    return poses, probs


def test_build():
    tf.keras.backend.set_learning_phase(1)
    inputs = tf.random.normal([128, 32, 32, 1])
    labels = tf.random.uniform([128, ], 0, 5, tf.int32)
    labels = tf.one_hot(labels, 5)
    outputs = build_encoder(inputs, 5, 16, 3, 'FM', utils.TensorLog())

