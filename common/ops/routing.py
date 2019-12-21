# Copyright 2018 The CapsLayer Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==========================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from common.ops import ops


def dynamic_routing(votes,
                    num_routing=3,
                    softmax_in=False,
                    temper=1.0,
                    activation='squash'):
    """ Dynamic routing algorithm.
    Args:
        votes: a tensor with shape [batch_size, ..., num_in, num_out, out_dims]
        num_routing: integer, number of routing iterations.
        softmax_in: do softmax on input capsules, default as False, which is the same as original routing
        temper: a param to make result sparser
        activation: activation of vector

    Returns:
        pose: a tensor with shape [batch_size, ..., num_out, out_dims]
        prob: a tensor with shape [batch_size, ..., num_out]
    """
    b = tf.zeros_like(votes, name='b')
    b = tf.reduce_sum(b, -1, keepdims=True)
    activation_fn = ops.get_activation(activation)

    for i in range(num_routing):
        if softmax_in:
            c = tf.nn.softmax(temper*b, axis=-3)
        else:
            c = tf.nn.softmax(temper*b, axis=-2)

        pose = tf.reduce_sum(c * votes, axis=-3, keepdims=True)
        pose, prob = activation_fn(pose, axis=-1)  # get [batch_size, ..., 1, num_out, out_dim]
        distances = votes * pose
        distances = tf.reduce_sum(distances, axis=-1, keepdims=True)  # [batch_size, ..., num_in, num_out, 1]
        b += distances

    return pose, prob

