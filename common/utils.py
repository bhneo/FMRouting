import tensorflow as tf
import numpy as np


class TensorLog(object):
    def __init__(self):
        self.hist = {}
        self.scalar = {}
        self.image = {}
        self.model = None

    def add_hist(self, name, tensor):
        self.hist[name] = tensor

    def add_scalar(self, name, tensor):
        self.scalar[name] = tensor

    def add_image(self, name, image):
        self.image[name] = image

    def get_outputs(self):
        outputs = []
        for key in self.hist:
            outputs.append(self.hist[key])
        for key in self.scalar:
            outputs.append(self.scalar[key])
        for key in self.image:
            outputs.append(self.image[key])
        return outputs

    def set_model(self, model):
        self.model = model

    def summary(self, outputs, epoch):
        i = 0
        for key in self.hist:
            tf.summary.histogram(key, outputs[i], step=epoch + 1)
            i += 1
        for key in self.scalar:
            tf.summary.scalar(key, outputs[i], step=epoch + 1)
            i += 1
        for key in self.image:
            tf.summary.image(key, outputs[i], step=epoch + 1)
            i += 1


def str2bool(obj):
    if isinstance(obj, str):
        if obj == 'True':
            return True
        elif obj == 'False':
            return False
        else:
            raise TypeError('Type not support:{}'.format(obj))
    if isinstance(obj, bool):
        return obj
    else:
        raise TypeError('{} is not str'.format(obj))


def kernel_tile(input, kernel, stride):
    """Tile the children poses/activations so that the children for each parent occur in one axis.

    Author:
      Ashley Gritzman 19/10/2018
    Args:
      input:
        tensor of child poses or activations
        poses (N, child_space, child_space, i, atom) -> (64, 7, 7, 8, 16)
        activations (N, child_space, child_space, i, 1) -> (64, 7, 7, 8, 1)
      kernel:
      stride:
    Returns:
      tiled:
        (N, parent_space, parent_space, kh*kw, i, 16 or 1)
        (64, 5, 5, 9, 8, 16 or 1)
      child_parent_matrix:
        A 2D numpy matrix containing mapping between children capsules along the
        rows, and parent capsules along the columns.
        (child_space^2, parent_space^2)
        (7*7, 5*5)
    """

    input_shape = input.get_shape()
    batch_size = tf.shape(input)[0]
    spatial_size = int(input_shape[1])
    n_capsules = int(input_shape[3])
    parent_spatial_size = int((spatial_size - kernel) / stride + 1)

    # Check that dim 1 and 2 correspond to the spatial size
    assert input_shape[1] == input_shape[2]

    # Matrix showing which children map to which parent. Children are rows,
    # parents are columns.
    child_parent_matrix = create_routing_map(spatial_size, kernel, stride)

    # Convert from np to tf
    # child_parent_matrix = tf.constant(child_parent_matrix)

    # Each row contains the children belonging to one parent
    child_to_parent_idx = group_children_by_parent(child_parent_matrix)

    # Spread out spatial dimension of children
    input = tf.reshape(input, [batch_size, spatial_size * spatial_size, -1])

    # Select which children go to each parent capsule
    tiled = tf.gather(input, child_to_parent_idx, axis=1)

    tiled = tf.squeeze(tiled)
    tiled = tf.reshape(tiled, [batch_size, parent_spatial_size, parent_spatial_size, kernel * kernel * n_capsules, -1])

    return tiled, child_parent_matrix


def group_children_by_parent(bin_routing_map):
    """Groups children capsules by parent capsule.

    Rearrange the bin_routing_map so that each row represents one parent capsule,   and the entries in the row are indexes of the children capsules that route to   that parent capsule. This mapping is only along the spatial dimension, each
    child capsule along in spatial dimension will actually contain many capsules,   e.g. 32. The grouping that we are doing here tell us about the spatial
    routing, e.g. if the lower layer is 7x7 in spatial dimension, with a kernel of
    3 and stride of 1, then the higher layer will be 5x5 in the spatial dimension.
    So this function will tell us which children from the 7x7=49 lower capsules
    map to each of the 5x5=25 higher capsules. One child capsule can be in several
    different parent capsules, children in the corners will only belong to one
    parent, but children towards the center will belong to several with a maximum   of kernel*kernel (e.g. 9), but also depending on the stride.

    Author:
      Ashley Gritzman 19/10/2018
    Args:
      bin_routing_map:
        binary routing map with children as rows and parents as columns
    Returns:
      children_per_parents:
        parents are rows, and the indexes in the row are which children belong to       that parent
    """

    tmp = np.where(np.transpose(bin_routing_map))
    children_per_parent = np.reshape(tmp[1], [bin_routing_map.shape[1], -1])

    return children_per_parent


def create_routing_map(child_space, k, s):
    """Generate TFRecord for train and test datasets from .mat files.

    Create a binary map where the rows are capsules in the lower layer (children)
    and the columns are capsules in the higher layer (parents). The binary map
    shows which children capsules are connected to which parent capsules along the   spatial dimension.

    Author:
      Ashley Gritzman 19/10/2018
    Args:
      child_space: spatial dimension of lower capsule layer
      k: kernel size
      s: stride
    Returns:
      binmap:
        A 2D numpy matrix containing mapping between children capsules along the
        rows, and parent capsules along the columns.
        (child_space^2, parent_space^2)
        (7*7, 5*5)
    """

    parent_space = int((child_space - k) / s + 1)
    binmap = np.zeros((child_space ** 2, parent_space ** 2))
    for r in range(parent_space):
        for c in range(parent_space):
            p_idx = r * parent_space + c
            for i in range(k):
                # c_idx stand for child_index; p_idx is parent_index
                c_idx = r * s * child_space + c * s + child_space * i
                binmap[(c_idx):(c_idx + k), p_idx] = 1
    return binmap