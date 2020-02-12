import tensorflow as tf
from tensorflow import keras


class MarginLoss(keras.losses.Loss):
    def __init__(self,
                 sparse=True,
                 upper_margin=0.9,
                 bottom_margin=0.1,
                 down_weight=0.5,
                 reduction=keras.losses.Reduction.AUTO,
                 name=None):
        super(MarginLoss, self).__init__(reduction=reduction, name=name)
        self.sparse = sparse
        self.upper_margin = upper_margin
        self.bottom_margin = bottom_margin
        self.down_weight = down_weight

    def call(self, y_true, y_pred):
        num_out = y_pred.get_shape().as_list()[-1]
        if self.sparse:
            y_true = tf.reshape(y_true, [-1])
            y_true = tf.one_hot(tf.cast(y_true, tf.int32), num_out)
        return margin_loss(y_true, y_pred, self.upper_margin, self.bottom_margin, self.down_weight)

    def get_config(self):
        config = {'upper_margin': self.upper_margin, 'bottom_margin': self.bottom_margin,
                  'down_weight': self.down_weight}
        base_config = super(MarginLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def spread_loss(labels, predictions, margin):
    a_target = tf.reduce_sum(labels * predictions, axis=1, keepdims=True)
    dist = (1 - labels) * margin - (a_target - predictions)
    loss = tf.pow(tf.maximum(0., dist), 2)
    return loss


def margin_loss(labels,
                predictions,
                upper_margin=0.9,
                bottom_margin=0.1,
                down_weight=0.5):
    positive_selector = tf.cast(tf.less(predictions, upper_margin), tf.float32)
    positive_cost = positive_selector * labels * tf.pow(predictions - upper_margin, 2)

    negative_selector = tf.cast(tf.greater(predictions, bottom_margin), tf.float32)
    negative_cost = negative_selector * (1 - labels) * tf.pow(predictions - bottom_margin, 2)
    loss = positive_cost + down_weight * negative_cost
    return loss


