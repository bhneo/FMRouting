import sys
import os

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
    model_name = '_'.join(['Ex4_2_1', str(params.model.layer_num)])
    model_name = '_'.join([model_name, params.model.pool])

    model_name += '_trial{}'.format(str(params.training.idx))
    return model_name


def build_model(shape, num_out, params):
    inputs = keras.Input(shape=shape)
    model_name = build_model_name(params)
    pose, prob, tensor_log = build(inputs, num_out, params.model.layer_num, params.model.pool)
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


def build(inputs, num_out, layer_num, pool):
    log = utils.TensorLog()
    conv1 = keras.layers.Conv2D(filters=64, kernel_size=5,
                                strides=1, padding='same', use_bias=True,
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer,
                                name='conv1')(inputs)

    conv2 = keras.layers.Conv2D(filters=64, kernel_size=5,
                                strides=1, padding='same', use_bias=True,
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer,
                                name='conv1')(conv1)

    conv3 = keras.layers.Conv2D(filters=64, kernel_size=5,
                                strides=2, padding='same', use_bias=True,
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer,
                                name='conv1')(conv2)

    conv4 = keras.layers.Conv2D(filters=64, kernel_size=5,
                                strides=2, padding='same', use_bias=True,
                                kernel_initializer=kernel_initializer,
                                kernel_regularizer=kernel_regularizer,
                                name='conv1')(conv3)

    backbone = res_blocks.build_resnet_backbone(inputs=inputs, layer_num=layer_num,
                                                start_filters=16, arch='cifar',
                                                use_bias=False,
                                                kernel_initializer=kernel_initializer,
                                                kernel_regularizer=kernel_regularizer,
                                                bn_axis=-1, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON,
                                                version='v2')
    log.add_hist('backbone', backbone)
    backbone_shape = backbone.get_shape().as_list()

    capsules = layers.CapsuleGroups(height=backbone_shape[1],
                                    width=backbone_shape[2],
                                    channel=backbone_shape[3],
                                    atoms=64,
                                    activation='norm' if pool == 'FM' else None)(backbone)
    log.add_hist('capsules_group', capsules)

    if pool == 'max':
        pose = layers.LastMaxPooling(axis=-3)(capsules)
    elif pool == 'average':
        pose = layers.LastAveragePooling(axis=-3)(capsules)
    else:
        pose = layers.LastFMPool(axis=-3, activation='None', log=log)(capsules)

    pose = keras.layers.Reshape([backbone_shape[3]])(pose)
    flatten = keras.layers.Flatten()(pose)
    prediction = keras.layers.Dense(units=num_out,
                                    kernel_initializer=keras.initializers.glorot_normal(),
                                    kernel_regularizer=kernel_regularizer)(flatten)
    log.add_hist('prediction', prediction)
    return pose, prediction, log


def lr_schedule(epoch, lr):
    if epoch in [81, 122]:
        lr /= 10
    return lr


def main():
    args, params = config.parse_args()
    train_set, test_set, info = data_input.build_dataset(params.dataset.name,
                                                         batch_size=params.training.batch_size,
                                                         flip=params.dataset.flip, crop=params.dataset.crop)
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
    inputs = keras.layers.Input((32, 32, 3))
    build(inputs, 10, 20, 'norm', 'max')
