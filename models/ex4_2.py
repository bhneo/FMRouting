import os
import sys

sys.path.append(os.getcwd())

from tensorflow import keras
from common import res_blocks, layers, utils, train
from common.inputs import data_input

import config

WEIGHT_DECAY = 1e-4
BATCH_NORM_EPSILON = 1e-3
BATCH_NORM_DECAY = 0.99

kernel_regularizer = keras.regularizers.l2(WEIGHT_DECAY)
kernel_initializer = keras.initializers.he_normal()


def build_model_name(params):
    model_name = '_'.join(['Ex4_2_2', params.model.pool])
    model_name += '_atom{}'.format(params.caps.atoms)

    if params.dataset.flip:
        model_name += '_flip'
    if params.dataset.crop:
        model_name += '_crop'

    model_name += '_bs{}'.format(str(params.training.batch_size))
    model_name += '_trial{}'.format(str(params.training.idx))
    return model_name


def build_model(shape, num_out, params):
    inputs = keras.Input(shape=shape)
    model_name = build_model_name(params)
    pose, prob, tensor_log = build(inputs, num_out, params.caps.atoms)
    model = keras.Model(inputs=inputs, outputs=prob, name=model_name)
    log_model = keras.Model(inputs=inputs, outputs=tensor_log.get_outputs(), name=model_name + '_log')
    tensor_log.set_model(log_model)
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1, momentum=params.training.momentum),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[])
    model.summary()
    lr_scheduler = keras.callbacks.LearningRateScheduler(schedule=lr_schedule, verbose=1)
    lr_scheduler.set_model(model)
    callbacks = [lr_scheduler]
    model.callbacks = callbacks
    return model, tensor_log


def build(inputs, num_out, atoms):
    log = utils.TensorLog()
    backbone = res_blocks.build_resnet_backbone(inputs=inputs,
                                                layer_num=0, repetitions=[8, 8, 8],
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

    pri_caps = keras.layers.BatchNormalization()(pri_caps)

    poses, probs = multi_caps_layer(pri_caps, [32, 16, num_out], log)
    # poses, probs = multi_caps_layer(pri_caps, [num_out], log)

    return poses, probs, log


def multi_caps_layer(inputs, out_caps, log):
    # inputs [bs, caps_in, atoms]
    poses, probs = layers.Activation('squash', with_prob=True)(inputs)
    for i, out_num in enumerate(out_caps):
        prediction_caps = layers.CapsuleTransformDense(num_out=out_num, matrix=True, out_atom=0,
                                                       share_weights=False,
                                                       regularizer=kernel_regularizer)(poses)
        prediction_caps = keras.layers.BatchNormalization()(prediction_caps)
        log.add_hist('prediction_caps{}'.format(i+1), prediction_caps)
        prediction_caps = layers.Activation('norm')(prediction_caps)
        poses, probs = layers.LastFMPool(axis=-3, activation='accumulate',
                                         shrink=False, stable=False, regularize=True,
                                         norm_pose=True if i==len(out_caps)-1 else False,
                                         log=None)(prediction_caps)

        log.add_hist('prob{}'.format(i+1), probs)
    return poses, probs


def lr_schedule(epoch, lr):
    if epoch in [81, 122]:
        lr /= 10
    return lr


def main():
    args, params = config.parse_args()
    train_set, test_set, info = data_input.build_dataset(params.dataset.name,
                                                         flip=params.dataset.flip,
                                                         crop=params.dataset.crop,
                                                         batch_size=params.training.batch_size)
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
