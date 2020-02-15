import os
import sys

sys.path.append(os.getcwd())

from tensorflow import keras
from common import res_blocks, layers, utils, train
from common.inputs import data_input, small_norb

import tensorflow as tf
import config

WEIGHT_DECAY = 1e-4
BATCH_NORM_EPSILON = 1e-3
BATCH_NORM_DECAY = 0.99

kernel_regularizer = keras.regularizers.l2(WEIGHT_DECAY)
kernel_initializer = keras.initializers.he_normal()
bn_before_caps = True


def build_model_name(params):
    model_name = '_'.join(['Ex4_4', params.model.pool])
    if params.model.pool == 'FM':
        model_name += '_atom{}'.format(params.caps.atoms)

    elif params.model.pool == 'dynamic' or params.model.pool == 'EM':
        model_name += '_iter{}'.format(params.routing.iter_num)
        model_name += '_atom{}'.format(params.caps.atoms)

    if bn_before_caps:
        model_name += '_bn'

    model_name += '_bs{}'.format(str(params.training.batch_size))
    model_name += '_trial{}'.format(str(params.training.idx))
    return model_name


def build_model(shape, num_out, params):
    inputs = keras.Input(shape=shape)
    model_name = build_model_name(params)
    pose, prob, tensor_log = build(inputs, num_out, params.caps.atoms, params.routing.iter_num, params.model.pool)
    model = keras.Model(inputs=inputs, outputs=prob, name=model_name)
    log_model = keras.Model(inputs=inputs, outputs=tensor_log.get_outputs(), name=model_name + '_log')
    tensor_log.set_model(log_model)
    optimizer = keras.optimizers.SGD(
        learning_rate=keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.01,
                                                                  decay_steps=20000,
                                                                  decay_rate=0.5), momentum=0.9)  # 3e-3 20000 0.96
    model.compile(optimizer=optimizer,
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[])
    model.summary()
    model.callbacks = []
    return model, tensor_log


def build(inputs, num_out, atoms, iter_num, pool):
    log = utils.TensorLog()
    backbone = res_blocks.build_resnet_backbone(inputs=inputs,
                                                layer_num=0, repetitions=[1, 1, 1],
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

    if bn_before_caps:
        pri_caps = keras.layers.BatchNormalization(axis=[1, 2])(pri_caps)

    poses, probs = multi_caps_layer(pri_caps, [num_out], pool, iter_num, log)

    return poses, probs, log


def multi_caps_layer(inputs, out_caps, pool, iter_num, log):
    # inputs [bs, caps_in, atoms]
    poses, probs = layers.Activation('squash', with_prob=True)(inputs)
    for i, out_num in enumerate(out_caps):
        prediction_caps = layers.CapsuleTransformDense(num_out=out_num, matrix=True, out_atom=0,
                                                       share_weights=False,
                                                       regularizer=kernel_regularizer)(poses)
        prediction_caps = keras.layers.BatchNormalization(axis=[1, 2, 3])(prediction_caps)
        log.add_hist('prediction_caps{}'.format(i+1), prediction_caps)
        if pool == 'dynamic':
            poses, probs = layers.DynamicRouting(num_routing=iter_num,
                                                 softmax_in=False,
                                                 temper=1,
                                                 activation='squash',
                                                 pooling=False)(prediction_caps)
        elif pool == 'EM':
            poses, probs = layers.EMRouting(num_routing=iter_num)((prediction_caps, probs))
        elif pool == 'FM':
            prediction_caps = layers.Activation('norm')(prediction_caps)
            poses, probs = layers.LastFMPool(axis=-3, activation='accumulate',
                                             shrink=False, stable=False, regularize=True,
                                             norm_pose=True if i==len(out_caps)-1 else False,
                                             log=None)(prediction_caps)

        log.add_hist('prob{}'.format(i+1), probs)
    return poses, probs


def main():
    args, params = config.parse_args()
    params.dataset.name = 'smallNORB'
    train_set, test_set, info = small_norb.build_dataset(batch_size=params.training.batch_size)
    model, tensor_log = build_model(shape=info.features['image'].shape,
                                    num_out=info.features['label'].num_classes,
                                    params=params)

    trainer = train.Trainer(model, params, info, tensor_log)
    if args.train:
        trainer.fit(train_set, test_set)
    else:
        trainer.evaluate(test_set)


if __name__ == "__main__":
    main()


def test_build():
    tf.keras.backend.set_learning_phase(1)
    inputs = tf.random.normal([128, 32, 32, 1])
    outputs = build(inputs, 5, 16, 3, 'EM')
