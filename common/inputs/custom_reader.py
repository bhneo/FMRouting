import os

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

from common.inputs.data_input import DataInfo


bxs_m2 = [[1, 1], [1, -1], [-1, 1], [-1, -1]]


def parse_multi_mnist1(serialized_example):
    """ Data parsing function.
    """
    features = tf.io.parse_single_example(serialized_example,
                                          features={
                                              'height': tf.io.FixedLenFeature([], tf.int64),
                                              'width': tf.io.FixedLenFeature([], tf.int64),
                                              'depth': tf.io.FixedLenFeature([], tf.int64),
                                              'label_1': tf.io.FixedLenFeature([], tf.int64),
                                              'label_2': tf.io.FixedLenFeature([], tf.int64),
                                              'image_raw_1': tf.io.FixedLenFeature([], tf.string),
                                              'image_raw_2': tf.io.FixedLenFeature([], tf.string),
                                              'merged_raw': tf.io.FixedLenFeature([], tf.string),
                                          })
    # Decode 3 images
    image_raw_1 = tf.io.decode_raw(features['image_raw_1'], tf.uint8)
    image_raw_1 = tf.reshape(image_raw_1, shape=[36, 36, 1])
    image_raw_2 = tf.io.decode_raw(features['image_raw_2'], tf.uint8)
    image_raw_2 = tf.reshape(image_raw_2, shape=[36, 36, 1])
    merged_raw = tf.io.decode_raw(features['merged_raw'], tf.uint8)
    merged_raw = tf.reshape(merged_raw, shape=[36, 36, 1])

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image_raw_1 = tf.cast(image_raw_1, tf.float32) * (1. / 255)
    image_raw_2 = tf.cast(image_raw_2, tf.float32) * (1. / 255)
    merged_raw = tf.cast(merged_raw, tf.float32) * (1. / 255)

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label_1 = tf.one_hot(tf.cast(features['label_1'], tf.int32), 10)
    label_2 = tf.one_hot(tf.cast(features['label_2'], tf.int32), 10)

    label = label_1 + label_2
    features = {'images': merged_raw,
                'labels': label,
                'recons_label': label_1,
                'recons_image': image_raw_1,
                'spare_label': label_2,
                'spare_image': image_raw_2}
    return features


def parse_multi_mnist(serialized_example):
    """ Data parsing function.
    """
    features = tf.io.parse_single_example(serialized_example,
                                          features={
                                              'height': tf.io.FixedLenFeature([], tf.int64),
                                              'width': tf.io.FixedLenFeature([], tf.int64),
                                              'depth': tf.io.FixedLenFeature([], tf.int64),
                                              'label_1': tf.io.FixedLenFeature([], tf.int64),
                                              'label_2': tf.io.FixedLenFeature([], tf.int64),
                                              'image_raw_1': tf.io.FixedLenFeature([], tf.string),
                                              'image_raw_2': tf.io.FixedLenFeature([], tf.string),
                                              # 'merged_raw': tf.io.FixedLenFeature([], tf.string),
                                          })
    # Decode 3 images
    image_raw_1 = tf.io.decode_raw(features['image_raw_1'], tf.uint8)
    image_raw_1 = tf.reshape(image_raw_1, shape=[36, 36, 1])
    image_raw_2 = tf.io.decode_raw(features['image_raw_2'], tf.uint8)
    image_raw_2 = tf.reshape(image_raw_2, shape=[36, 36, 1])
    merged_raw = tf.add(tf.cast(image_raw_1, tf.int32), tf.cast(image_raw_2, tf.int32))
    merged_raw = tf.minimum(merged_raw, 255)

    # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image_raw_1 = tf.cast(image_raw_1, tf.float32) * (1. / 255)
    image_raw_2 = tf.cast(image_raw_2, tf.float32) * (1. / 255)
    merged_raw = tf.cast(merged_raw, tf.float32) * (1. / 255)

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label_1 = tf.one_hot(tf.cast(features['label_1'], tf.int32), 10)
    label_2 = tf.one_hot(tf.cast(features['label_2'], tf.int32), 10)

    label = label_1 + label_2
    features = {'images': merged_raw,
                'labels': label,
                'recons_label': label_1,
                'recons_image': image_raw_1,
                'spare_label': label_2,
                'spare_image': image_raw_2}
    return features


def parse_aff_mnist(serialized_example):
    """ Data parsing function.
    """
    features = tf.io.parse_single_example(serialized_example,
                                          features={'image': tf.io.FixedLenFeature([], tf.string),
                                                    'label': tf.io.FixedLenFeature([], tf.int64),
                                                    'height': tf.io.FixedLenFeature([], tf.int64),
                                                    'width': tf.io.FixedLenFeature([], tf.int64),
                                                    'depth': tf.io.FixedLenFeature([], tf.int64)})
    image = tf.io.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image, shape=[40, 40, 1])
    image = tf.cast(image, tf.float32) * (1. / 255)
    label = tf.cast(features['label'], tf.int32)
    label = tf.one_hot(label, 10)
    return image, label


def build_parse(dataset):
    if dataset == 'aff_mnist':
        return parse_aff_mnist
    elif dataset == 'shift_mnist':
        return parse_aff_mnist
    elif dataset == 'multi_mnist':
        return parse_multi_mnist
    elif dataset == 'multi_mnist1':
        return parse_multi_mnist1


def get_dataset(name, data_path):
    if name == 'aff_mnist':# 1920000/320000
        train_files = os.path.join(data_path, "train_affnist.tfrecord")
        test_files = os.path.join(data_path, "test_affnist.tfrecord")
        train_parse_fun = build_parse('aff_mnist')
        test_parse_fun = build_parse('aff_mnist')
        info = DataInfo(tfds.features.FeaturesDict({'image': tfds.features.Image(shape=(40, 40, 1)),
                                                    'label': tfds.features.ClassLabel(num_classes=10)}),
                        {'train_examples': 1920000,
                         'test_examples': 320000})
    elif name == 'shift_mnist':# 10140000/1690000
        train_files = os.path.join(data_path, "train_6shifted_mnist.tfrecord")
        test_files = os.path.join(data_path, "test_6shifted_mnist.tfrecord")
        train_parse_fun = build_parse('shift_mnist')
        test_parse_fun = build_parse('shift_mnist')
        info = DataInfo(tfds.features.FeaturesDict({'image': tfds.features.Image(shape=(40, 40, 1)),
                                                    'label': tfds.features.ClassLabel(num_classes=10)}),
                        {'train_examples': 10140000,
                         'test_examples': 1690000})
    # elif name == 'multi_mnist':# 6000000/1000000
    #     train_files = [os.path.join(data_path, 'train', "multitrain_6shifted_mnist.tfrecords-0000{}-of-00060".format(i)) for i
    #                    in range(10)] + [os.path.join(data_path, 'train', "multitrain_6shifted_mnist.tfrecords-000{}-of-00060".format(i)) for i
    #                    in range(10, 60)]
    #     test_files = [os.path.join(data_path, 'test', "multitest_6shifted_mnist.tfrecords-0000{}-of-00010".format(i)) for i in range(10)]
    #     train_parse_fun = build_parse('multi_mnist')
    #     test_parse_fun = build_parse('multi_mnist')
    #     info = DataInfo(tfds.features.FeaturesDict({'image': tfds.features.Image(shape=(36, 36, 1)),
    #                                                 'label': tfds.features.ClassLabel(num_classes=10)}),
    #                     {'train_examples': 6000000,
    #                      'test_examples': 1000000})
    elif name == 'multi_mnist':# 599999/100000
        train_files = [os.path.join(data_path, "multitrain_6shifted_mnist.tfrecords-0000{}-of-00006".format(i)) for i
                       in range(6)]
        test_files = [os.path.join(data_path, "multitest_6shifted_mnist.tfrecords-00000-of-00001")]
        train_parse_fun = build_parse('multi_mnist1')
        test_parse_fun = build_parse('multi_mnist1')
        info = DataInfo(tfds.features.FeaturesDict({'image': tfds.features.Image(shape=(36, 36, 1)),
                                                    'label': tfds.features.ClassLabel(num_classes=10)}),
                        {'train_examples': 599999,
                         'test_examples': 100000})
    else:
        raise Exception('dataset note support!')

    train = tf.data.TFRecordDataset(train_files)
    test = tf.data.TFRecordDataset(test_files)

    return train, test, train_parse_fun, test_parse_fun, info


def build_dataset(name, data_dir='data', batch_size=128, buffer_size=50000):
    data_path = os.path.join(data_dir, name)
    train, test, train_parse_fun, test_parse_fun, info = get_dataset(name, data_path)

    if buffer_size > 0:
        train = train.shuffle(buffer_size=buffer_size)

    train = train.map(train_parse_fun,
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .batch(batch_size)\
        .prefetch(tf.data.experimental.AUTOTUNE)
    test = test.map(test_parse_fun,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .batch(batch_size)\
        .prefetch(tf.data.experimental.AUTOTUNE)

    return train, test, info


def count_data(name):
    train, test, _ = build_dataset(name, '')
    train_num = 0
    for image, label in train:
        train_num += image.shape[0]

    test_num = 0
    for image, label in test:
        test_num += image.shape[0]

    print('train num:', train_num)
    print('test num:', test_num)


def count_multi_mnist():
    train, test, _ = build_dataset('multi_mnist', '')
    train_num = 0
    for feature in train:
        train_num += feature['images'].shape[0]

    test_num = 0
    for feature in test:
        test_num += feature['images'].shape[0]

    print('train num:', train_num)
    print('test num:', test_num)


def view_data(name, img_stand=False):
    train, test, _ = build_dataset(name, '')
    for image, label in train:
        if not img_stand:
            image /= 255.
        out_image(image, label)
        break

    for image, label in test:
        if not img_stand:
            image /= 255.
        out_image(image, label)
        break


def view_multi_mnist(img_stand=False):
    train, test, _ = build_dataset('multi_mnist', '')
    for features in train:
        image = features['images']
        label = features['labels']
        recons_label = features['recons_label']
        recons_image = features['recons_image']
        spare_label = features['spare_label']
        spare_image = features['spare_image']

        image_final = tf.concat([recons_image, spare_image, image], axis=2)
        if not img_stand:
            image_final /= 255.
        out_image(image_final, label)
        break


def out_image(images, labels):
    plt.figure()
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.title(tf.argmax(labels[i]).numpy())
        image = images[i, :, :, :]
        if image.shape[-1] == 1:
            image = np.squeeze(image, -1)
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(image)
    plt.subplots_adjust(hspace=0.5)
    plt.show()


if __name__ == "__main__":
    # view_data('aff_mnist')
    # count_data('shift_mnist')
    # view_multi_mnist()
    count_multi_mnist()
