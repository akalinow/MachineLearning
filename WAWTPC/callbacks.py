from tensorflow.python.platform import tf_logging as logging
import numpy as np
from keras.utils import io_utils
from tensorflow import keras

class BatchEarlyStopping(keras.callbacks.Callback):
    """Stop training when a monitored metric has stopped improving.
    Assuming the goal of a training is to minimize the loss. With this, the
    metric to be monitored would be `'loss'`, and mode would be `'min'`. A
    `model.fit()` training loop will check at end of every batch whether
    the loss is no longer decreasing, considering the `min_delta` and
    `patience` if applicable. Once it's found no longer decreasing,
    `model.stop_training` is marked True and the training terminates.
    The quantity to be monitored needs to be available in `logs` dict.
    To make it so, pass the loss or metrics at `model.compile()`.
    Args:
      monitor: Quantity to be monitored.
      min_delta: Minimum change in the monitored quantity
          to qualify as an improvement, i.e. an absolute
          change of less than min_delta, will count as no
          improvement.
      patience: Number of batches with no improvement
          after which training will be stopped.
      verbose: Verbosity mode, 0 or 1. Mode 0 is silent, and mode 1
          displays messages when the callback takes an action.
      mode: One of `{"auto", "min", "max"}`. In `min` mode,
          training will stop when the quantity
          monitored has stopped decreasing; in `"max"`
          mode it will stop when the quantity
          monitored has stopped increasing; in `"auto"`
          mode, the direction is automatically inferred
          from the name of the monitored quantity.
      baseline: Baseline value for the monitored quantity.
          Training will stop if the model doesn't show improvement over the
          baseline.
      restore_best_weights: Whether to restore model weights from
          the batch with the best value of the monitored quantity.
          If False, the model weights obtained at the last step of
          training are used. An batch will be restored regardless
          of the performance relative to the `baseline`. If no batch
          improves on `baseline`, training will run for `patience`
          batches and restore weights from the best batch in that set.
      start_from_batch: Number of batches to wait before starting
          to monitor improvement. This allows for a warm-up period in which
          no improvement is expected and thus training will not be stopped.
    Example:
    >>> callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    >>> # This callback will stop the training when there is no improvement in
    >>> # the loss for three consecutive batches.
    >>> model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
    >>> model.compile(tf.keras.optimizers.SGD(), loss='mse')
    >>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
    ...                     batches=10, batch_size=1, callbacks=[callback],
    ...                     verbose=0)
    >>> len(history.history['loss'])  # Only 4 batches are run.
    4
    """

    def __init__(
        self,
        monitor="val_loss",
        min_delta=0,
        patience=0,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
        start_from_batch=0,
    ):
        super().__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_batch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.start_from_batch = start_from_batch

        if mode not in ["auto", "min", "max"]:
            logging.warning(
                "EarlyStopping mode %s is unknown, fallback to auto mode.",
                mode,
            )
            mode = "auto"

        if mode == "min":
            self.monitor_op = np.less
        elif mode == "max":
            self.monitor_op = np.greater
        else:
            if (
                self.monitor.endswith("acc")
                or self.monitor.endswith("accuracy")
                or self.monitor.endswith("auc")
            ):
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_batch = 0
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None
        self.best_batch = 0

    def on_batch_end(self, batch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None or batch < self.start_from_batch:
            # If no monitor value exists or still in initial warm-up stage.
            return
        if self.restore_best_weights and self.best_weights is None:
            # Restore the weights after first batch if no progress is ever made.
            self.best_weights = self.model.get_weights()

        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best = current
            self.best_batch = batch
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
            # Only restart wait if we beat both the baseline and our previous
            # best.
            if self.baseline is None or self._is_improvement(
                current, self.baseline
            ):
                self.wait = 0

        # Only check after the first batch.
        if self.wait >= self.patience and batch > 0:
            self.stopped_batch = batch
            self.model.stop_training = True
            if self.restore_best_weights and self.best_weights is not None:
                if self.verbose > 0:
                    io_utils.print_msg(
                        "Restoring model weights from "
                        "the end of the best batch: "
                        f"{self.best_batch + 1}."
                    )
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_batch > 0 and self.verbose > 0:
            io_utils.print_msg(
                f"Batch {self.stopped_batch + 1}: early stopping"
            )

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logging.warning(
                "Early stopping conditioned on metric `%s` "
                "which is not available. Available metrics are: %s",
                self.monitor,
                ",".join(list(logs.keys())),
            )
        return monitor_value

    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value - self.min_delta, reference_value)