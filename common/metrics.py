from tensorflow.python.keras.metrics import MeanMetricWrapper
from tensorflow.python.keras.utils import metrics_utils


class Mean(Reduce):
  """Computes the (weighted) mean of the given values.

  For example, if values is [1, 3, 5, 7] then the mean is 4.
  If the weights were specified as [1, 1, 0, 0] then the mean would be 2.

  This metric creates two variables, `total` and `count` that are used to
  compute the average of `values`. This average is ultimately returned as `mean`
  which is an idempotent operation that simply divides `total` by `count`.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  Usage:

  ```python
  m = tf.keras.metrics.Mean()
  m.update_state([1, 3, 5, 7])
  print('Final result: ', m.result().numpy())  # Final result: 4.0
  ```

  Usage with tf.keras API:

  ```python
  model = tf.keras.Model(inputs, outputs)
  model.add_metric(tf.keras.metrics.Mean(name='mean_1')(outputs))
  model.compile('sgd', loss='mse')
  ```
  """

  def __init__(self, name='mean', dtype=None):
    """Creates a `Mean` instance.

    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    """
    super(Mean, self).__init__(
        reduction=metrics_utils.Reduction.WEIGHTED_MEAN, name=name, dtype=dtype)


class MeanMetricWrapper(Mean):
  """Wraps a stateless metric function with the Mean metric."""

  def __init__(self, fn, name=None, dtype=None, **kwargs):
    """Creates a `MeanMetricWrapper` instance.

    Args:
      fn: The metric function to wrap, with signature
        `fn(y_true, y_pred, **kwargs)`.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
      **kwargs: The keyword arguments that are passed on to `fn`.
    """
    super(MeanMetricWrapper, self).__init__(name=name, dtype=dtype)
    self._fn = fn
    self._fn_kwargs = kwargs

  def update_state(self, y_true, y_pred, sample_weight=None):
    """Accumulates metric statistics.

    `y_true` and `y_pred` should have the same shape.

    Args:
      y_true: The ground truth values.
      y_pred: The predicted values.
      sample_weight: Optional weighting of each example. Defaults to 1. Can be
        a `Tensor` whose rank is either 0, or the same rank as `y_true`,
        and must be broadcastable to `y_true`.

    Returns:
      Update op.
    """
    y_true = math_ops.cast(y_true, self._dtype)
    y_pred = math_ops.cast(y_pred, self._dtype)
    [y_true, y_pred], sample_weight = \
        metrics_utils.ragged_assert_compatible_and_get_flat_values(
            [y_true, y_pred], sample_weight)
    y_pred, y_true = tf_losses_utils.squeeze_or_expand_dimensions(
        y_pred, y_true)

    matches = self._fn(y_true, y_pred, **self._fn_kwargs)
    return super(MeanMetricWrapper, self).update_state(
        matches, sample_weight=sample_weight)

  def get_config(self):
    config = {}
    for k, v in six.iteritems(self._fn_kwargs):
      config[k] = K.eval(v) if is_tensor_or_variable(v) else v
    base_config = super(MeanMetricWrapper, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))



class ThresholdCategoricalAccuracy(MeanMetricWrapper):

    def __init__(self, threshold, name='threshold_categorical_accuracy', dtype=None):
        super(ThresholdCategoricalAccuracy, self).__init__(
          self.threshold_categorical_accuracy,
          name,
          dtype=dtype)
        self.thre


    def threshold_categorical_accuracy(y_true, y_pred):

        y_pred_rank = ops.convert_to_tensor(y_pred).shape.ndims
        y_true_rank = ops.convert_to_tensor(y_true).shape.ndims
        # If the shape of y_true is (num_samples, 1), squeeze to (num_samples,)
        if (y_true_rank is not None) and (y_pred_rank is not None) and (len(
            K.int_shape(y_true)) == len(K.int_shape(y_pred))):
          y_true = array_ops.squeeze(y_true, [-1])
        y_pred = math_ops.argmax(y_pred, axis=-1)

        # If the predicted output and actual output types don't match, force cast them
        # to match.
        if K.dtype(y_pred) != K.dtype(y_true):
          y_pred = math_ops.cast(y_pred, K.dtype(y_true))

        return math_ops.cast(math_ops.equal(y_true, y_pred), K.floatx())