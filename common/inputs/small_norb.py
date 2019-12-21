"""
License: Apache 2.0
Author: Ashley Gritzman
E-mail: ashley.gritzman@za.ibm.com
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os

from common.inputs import data_input


def _parser(serialized_example):
    """Parse smallNORB example from tfrecord.

    Author:
      Ashley Gritzman 15/11/2018
    Args:
      serialized_example: serialized example from tfrecord
    Returns:
      img: image
      lab: label
      cat:
        category
        the instance in the category (0 to 9)
      elv:
        elevation
        the elevation (0 to 8, which mean cameras are 30,
        35,40,45,50,55,60,65,70 degrees from the horizontal respectively)
      azi:
        azimuth
        the azimuth (0,2,4,...,34, multiply by 10 to get the azimuth in
        degrees)
      lit:
        lighting
        the lighting condition (0 to 5)
    """

    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'img_raw': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'category': tf.io.FixedLenFeature([], tf.int64),
            'elevation': tf.io.FixedLenFeature([], tf.int64),
            'azimuth': tf.io.FixedLenFeature([], tf.int64),
            'lighting': tf.io.FixedLenFeature([], tf.int64),
        })

    img = tf.io.decode_raw(features['img_raw'], tf.float64)
    img = tf.reshape(img, [96, 96, 1])
    img = tf.cast(img, tf.float32)  # * (1. / 255) # left unnormalized

    lab = tf.cast(features['label'], tf.int32)
    cat = tf.cast(features['category'], tf.int32)
    elv = tf.cast(features['elevation'], tf.int32)
    azi = tf.cast(features['azimuth'], tf.int32)
    lit = tf.cast(features['lighting'], tf.int32)

    return img, lab, cat, elv, azi, lit


def _train_preprocess(img, lab, cat, elv, azi, lit):
    """Preprocessing for training.

    Preprocessing from Hinton et al. (2018) "Matrix capsules with EM routing."
    Hinton2018: "We downsample smallNORB to 48 × 48 pixels and normalize each
    image to have zero mean and unit variance. During training, we randomly crop
    32 × 32 patches and add random brightness and contrast to the cropped images.
    During test, we crop a 32 × 32 patch from the center of the image and
    achieve..."

    Author:
      Ashley Gritzman 15/11/2018
    Args:
      img: this fn only works on the image
      lab, cat, elv, azi, lit: allow these to pass through
    Returns:
      img: image processed
      lab, cat, elv, azi, lit: allow these to pass through
    """

    img = img / 255.
    img = tf.image.resize(img, [48, 48])
    # img = tf.image.per_image_standardization(img)
    img = tf.image.random_crop(img, [32, 32, 1])
    # img = tf.image.random_brightness(img, max_delta=2.0)
    # original 0.5, 1.5
    img = tf.image.random_contrast(img, lower=0.5, upper=1.5)

    # Original
    # image = tf.image.random_brightness(image, max_delta=32. / 255.)
    # image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    # image = tf.image.resize_images(image, [48, 48])
    # image = tf.random_crop(image, [32, 32, 1])

    return img, lab, cat, elv, azi, lit


def _val_preprocess(img, lab, cat, elv, azi, lit):
    """Preprocessing for validation/testing.

    Preprocessing from Hinton et al. (2018) "Matrix capsules with EM routing."
    Hinton2018: "We downsample smallNORB to 48 × 48 pixels and normalize each
    image to have zero mean and unit variance. During training, we randomly crop
    32 × 32 patches and add random brightness and contrast to the cropped
    images. During test, we crop a 32 × 32 patch from the center of the image
    and achieve..."

    Author:
      Ashley Gritzman 15/11/2018
    Args:
      img: this fn only works on the image
      lab, cat, elv, azi, lit: allow these to pass through
    Returns:
      img: image processed
      lab, cat, elv, azi, lit: allow these to pass through
    """

    img = img / 255.
    img = tf.image.resize(img, [48, 48])
    # img = tf.image.per_image_standardization(img)
    img = tf.slice(img, [8, 8, 0], [32, 32, 1])

    # Original
    # image = tf.image.resize_images(image, [48, 48])
    # image = tf.slice(image, [8, 8, 0], [32, 32, 1])

    return img, lab, cat, elv, azi, lit


def _simply(img, lab, cat, elv, azi, lit):
    return img, lab


def input_fn(path, is_train: bool, batch_size=128, simplify=False):
    """Input pipeline for smallNORB using tf.data.

    Author:
      Ashley Gritzman 15/11/2018
    Args:
      is_train:
    Returns:
      dataset: image tf.data.Dataset
    """

    import re
    if is_train:
        CHUNK_RE = re.compile(r"train.*\.tfrecords")
    else:
        CHUNK_RE = re.compile(r"test.*\.tfrecords")

    chunk_files = [os.path.join(path, fname)
                   for fname in os.listdir(path)
                   if CHUNK_RE.match(fname)]

    # 1. create the dataset
    dataset = tf.data.TFRecordDataset(chunk_files)

    # 2. map with the actual work (preprocessing, augmentation…) using multiple
    # parallel calls
    dataset = dataset.map(_parser, num_parallel_calls=4)
    if is_train:
        dataset = dataset.map(_train_preprocess,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.map(_val_preprocess,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if simplify:
        dataset = dataset.map(_simply, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # 3. shuffle (with a big enough buffer size)
    # In response to a question on OpenReview, Hinton et al. wrote the
    # following:
    # https://openreview.net/forum?id=HJWLfGWRb&noteId=rJgxonoNnm
    # "We did not have any special ordering of training batches and we random
    # shuffle. In terms of TF batch:
    # capacity=2000 + 3 * batch_size, ensures a minimum amount of shuffling of
    # examples. min_after_dequeue=2000."
    capacity = 2000 + 3 * batch_size
    dataset = dataset.shuffle(buffer_size=capacity)

    # 4. batch
    dataset = dataset.batch(batch_size, drop_remainder=False)

    # 5. repeat
    # dataset = dataset.repeat(count=FLAGS.epoch)

    # 6. prefetch
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def build_dataset(data_dir='data', batch_size=128, simplify=True):
    data_path = os.path.join(data_dir, 'smallNORB')
    train = input_fn(data_path, True, batch_size=batch_size, simplify=simplify)
    test = input_fn(data_path, False, batch_size=batch_size, simplify=simplify)
    info = data_input.DataInfo(tfds.features.FeaturesDict({'image': tfds.features.Image(shape=(32, 32, 1)),
                                                'label': tfds.features.ClassLabel(num_classes=5)}),
                    {'train_examples': 48600,
                     'test_examples': 48600})
    return train, test, info


def count_data(path):
    train, test, _ = build_dataset(path)
    train_num = 0
    for image, label in train:
        train_num += image.shape[0]

    test_num = 0
    for image, label in test:
        test_num += image.shape[0]

    print('train num:', train_num)
    print('test num:', test_num)


def plot_smallnorb(path, is_train=False, samples_per_class=5):
    """Plot examples from the smallNORB dataset.

    Execute this command in a Jupyter Notebook.

    Author:
      Ashley Gritzman 18/04/2019
    Args:
      is_train: True for the training dataset, False for the test dataset
      samples_per_class: number of samples images per class
    Returns:
      None
    """

    # To plot pretty figures
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    CLASSES = ['animal', 'human', 'airplane', 'truck', 'car']

    # Get batch from data queue. Batch size is FLAGS.batch_size, which is then
    # divided across multiple GPUs
    path = os.path.join(path, 'smallNORB')
    dataset = input_fn(path, is_train)
    img_bch, lab_bch, cat_bch, elv_bch, azi_bch, lit_bch = next(iter(dataset))
    img_bch = img_bch.numpy()
    lab_bch = lab_bch.numpy()
    cat_bch = cat_bch.numpy()
    elv_bch = elv_bch.numpy()
    azi_bch = azi_bch.numpy()
    lit_bch = lit_bch.numpy()

    num_classes = len(CLASSES)

    fig = plt.figure(figsize=(num_classes * 2, samples_per_class * 2))
    fig.suptitle("category, elevation, azimuth, lighting")
    for y, cls in enumerate(CLASSES):
        idxs = np.flatnonzero(lab_bch == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            # plt.imshow(img_bch[idx].astype('uint8').squeeze())
            plt.imshow(np.squeeze(img_bch[idx]))
            plt.xticks([], [])
            plt.yticks([], [])
            plt.xlabel("{}, {}, {},{}".format(cat_bch[idx], elv_bch[idx],
                                              azi_bch[idx], lit_bch[idx]))
            # plt.axis('off')

            if i == 0:
                plt.title(cls)
    plt.show()


if __name__ == '__main__':
    path = '../../data'
    plot_smallnorb(path, is_train=False)
    # count_data(path)
