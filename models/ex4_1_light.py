import sys
import os

sys.path.append(os.getcwd())

from tensorflow import keras
from common.inputs import data_input
from common import layers, losses, utils, train

import tensorflow as tf
import time
import config


def build_model_name(params):
    model_name = '_'.join(['Ex4_1', params.model.pool])
    if params.model.pool == 'dynamic':
        model_name += '_iter{}'.format(params.routing.iter_num)

    if params.model.in_norm_fn != '':
        model_name += '_{}'.format(params.model.in_norm_fn)

    model_name += '_trial{}'.format(str(params.training.idx))

    if params.dataset.flip:
        model_name += '_flip'
    if params.dataset.crop:
        model_name += '_crop'
    return model_name


def build_model(shape, num_out, params):
    inputs = keras.Input(shape=shape)
    model_name = build_model_name(params)
    pose, prob, tensor_log = build(inputs, num_out, params.routing.iter_num, params.model.pool, params.model.in_norm_fn)
    model = keras.Model(inputs=inputs, outputs=prob, name=model_name)
    log_model = keras.Model(inputs=inputs, outputs=tensor_log.get_outputs(), name=model_name + '_log')
    tensor_log.set_model(log_model)
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=losses.MarginLoss(upper_margin=0.9, bottom_margin=0.1, down_weight=0.5),
                  metrics=[])
    model.summary()

    model.callbacks = []
    return model, tensor_log


def build(inputs, num_out, iter_num, pool, in_norm_fn):
    log = utils.TensorLog()
    conv1 = keras.layers.Conv2D(filters=64,
                                kernel_size=9,
                                strides=1,
                                padding='valid',
                                activation='relu')(inputs)
    pose, prob = layers.PrimaryCapsule(kernel_size=9,
                                       strides=2,
                                       padding='valid',
                                       groups=8,
                                       use_bias=True,
                                       atoms=8,
                                       activation=in_norm_fn,
                                       kernel_initializer=keras.initializers.he_normal())(conv1)
    transformed_caps = layers.CapsuleTransformDense(num_out=num_out,
                                                    out_atom=16,
                                                    share_weights=False,
                                                    initializer=keras.initializers.glorot_normal())(pose)
    if pool == 'dynamic':
        pose, prob = layers.DynamicRouting(num_routing=iter_num,
                                           softmax_in=False,
                                           temper=1,
                                           activation='squash',
                                           pooling=False)(transformed_caps)
    elif pool == 'EM':
        pose, prob = layers.EMRouting(num_routing=iter_num)((transformed_caps, prob))
    elif pool == 'FM':
        pose, prob = layers.LastFMPool(axis=-3, activation='accumulate', shrink=True,
                                       stable=False,
                                       log=log)(transformed_caps)

    log.add_hist('prob', prob)
    return pose, prob, log


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


def print_results(model, model_dir, data):
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=losses.MarginLoss(0.9, 0.1, 0.5),
                  metrics=[])
    ckpt = tf.train.Checkpoint(optimizer=model.optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))

    train_set, test_set, info = data_input.build_dataset(data, path='../../data', batch_size=128,
                                                         flip=False, crop=False)
    metric = keras.metrics.SparseCategoricalAccuracy()
    with tf.device('/GPU:0'):
        gpu_results = []
        for image, label in test_set:
            t1 = time.time() * 1000
            result = model(image)
            t2 = time.time() * 1000
            gpu_results.append(t2 - t1)
            metric.update_state(label, result)
    # mean1, var1 = tf.nn.moments(tf.constant(gpu_results), 0)
    # print('GPU:', mean1, var1)
    mean2, var2 = tf.nn.moments(tf.constant(gpu_results[1:]), 0)
    print('GPU:', mean2.numpy(), var2.numpy())
    print('acc:', metric.result().numpy())

    metric.reset_states()
    with tf.device('/CPU:0'):
        cpu_results = []
        for image, label in test_set:
            t1 = time.time() * 1000
            result = model(image)
            t2 = time.time() * 1000
            cpu_results.append(t2 - t1)
            metric.update_state(label, result)
    mean, var = tf.nn.moments(tf.constant(cpu_results), 0)
    print('CPU:', mean.numpy(), var.numpy())
    print('acc:', metric.result().numpy())


def test_inference_cifar10():
    model_dir = '../../log/cifar10_ori/ori.vectorcaps_fm_routing_iter1_trial1_flip_crop'
    inputs = keras.Input((32, 32, 3))
    pose, prob, _ = build(inputs, 10, 1, 'routing', 'squash')
    model = keras.Model(inputs=inputs, outputs=prob)

    print_results(model, model_dir, 'cifar10')


def test_inference_cifar10_iter3():
    model_dir = '../../log/cifar10_ori/ori.vectorcaps_fm_routing_iter3_trial1_flip_crop'
    inputs = keras.Input((32, 32, 3))
    pose, prob, _ = build(inputs, 10, 3, 'routing', 'squash')
    model = keras.Model(inputs=inputs, outputs=prob)

    print_results(model, model_dir, 'cifar10')


def test_inference_cifar10_FM():
    model_dir = '../../log/cifar10_ori/ori.vectorcaps_fm_FM_trial4_flip_crop'
    inputs = keras.Input((32, 32, 3))
    pose, prob, _ = build(inputs, 10, 3, 'FM', 'squash')
    model = keras.Model(inputs=inputs, outputs=prob)

    print_results(model, model_dir, 'cifar10')


def test_inference_fashionmnist():
    model_dir = '../../log/fashion_mnist/ori.vectorcaps_fm_routing_iter1_trial1'
    inputs = keras.Input((28, 28, 1))
    pose, prob, _ = build(inputs, 10, 1, 'routing', 'squash')
    model = keras.Model(inputs=inputs, outputs=prob)

    print_results(model, model_dir, 'fashion_mnist')


def test_inference_fashionmnist_iter3():
    model_dir = '../../log/fashion_mnist/ori.vectorcaps_fm_routing_iter3_trial4_crop'
    inputs = keras.Input((28, 28, 1))
    pose, prob, _ = build(inputs, 10, 3, 'routing', 'squash')
    model = keras.Model(inputs=inputs, outputs=prob)

    print_results(model, model_dir, 'fashion_mnist')


def test_inference_fashionmnist_FM():
    model_dir = '../../log/fashion_mnist/ori.vectorcaps_fm_FM_trial5_crop'
    inputs = keras.Input((28, 28, 1))
    pose, prob, _ = build(inputs, 10, 3, 'FM', 'squash')
    model = keras.Model(inputs=inputs, outputs=prob)

    print_results(model, model_dir, 'fashion_mnist')


def test_inference_svhn():
    model_dir = '../../log/svhn_cropped/ori.vectorcaps_fm_routing_iter1_trial1'
    inputs = keras.Input((32, 32, 3))
    pose, prob, _ = build(inputs, 10, 1, 'routing', 'squash')
    model = keras.Model(inputs=inputs, outputs=prob)

    print_results(model, model_dir, 'svhn_cropped')


def test_inference_svhn_iter3():
    model_dir = '../../log/svhn_cropped/ori.vectorcaps_fm_routing_iter3_trial4_crop'
    inputs = keras.Input((32, 32, 3))
    pose, prob, _ = build(inputs, 10, 3, 'routing', 'squash')
    model = keras.Model(inputs=inputs, outputs=prob)

    print_results(model, model_dir, 'svhn_cropped')


def test_inference_svhn_FM():
    model_dir = '../../log/svhn_cropped/ori.vectorcaps_fm_FM_trial5_crop'
    inputs = keras.Input((32, 32, 3))
    pose, prob, _ = build(inputs, 10, 3, 'FM', 'squash')
    model = keras.Model(inputs=inputs, outputs=prob)

    print_results(model, model_dir, 'svhn_cropped')

