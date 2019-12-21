import os
from importlib import import_module
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.keras import backend as K
import tensorflow as tf
from common.inputs import data_input

from config import params, parse_args


is_tracing = False


def get_extra_losses(model):
    loss = 0
    if len(model.losses) > 0:
        loss = tf.math.add_n(model.losses)
    return loss


def get_train_step(model, loss, accuracy):
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images)
            extra_loss = get_extra_losses(model)
            pred_loss = model.loss(labels, predictions)
            total_loss = pred_loss + extra_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        # Update the metrics
        loss.update_state(total_loss)
        accuracy.update_state(labels, predictions)
    return train_step


def get_test_step(model, loss, accuracy):
    @tf.function
    def test_step(images, labels):
        predictions = model(images)
        extra_loss = get_extra_losses(model)
        pred_loss = model.loss(labels, predictions)
        total_loss = pred_loss + extra_loss
        # Update the metrics
        loss.update_state(total_loss)
        accuracy.update_state(labels, predictions)
    return test_step


def get_log_step(tensor_log, writer):

    @tf.function
    def log_step(images, labels, epoch):
        if tensor_log:
            logs = tensor_log.model((images, labels))
            with writer.as_default():
                tensor_log.summary(logs, epoch)
    return log_step


def do_callbacks(state, callbacks, epoch=0, batch=0):
    if state == 'on_train_begin':
        for callback in callbacks:
            callback.on_train_begin()
    if state == 'on_epoch_begin':
        for callback in callbacks:
            callback.on_epoch_begin(epoch)
    if state == 'on_epoch_end':
        for callback in callbacks:
            callback.on_epoch_end(epoch)
    if state == 'on_batch_begin':
        for callback in callbacks:
            callback.on_batch_begin(batch)
    if state == 'on_batch_end':
        for callback in callbacks:
            callback.on_batch_end(batch)


def log_trace(step, writer, logdir):
    global is_tracing
    if step == 1 and not is_tracing:
        summary_ops_v2.trace_on(graph=True, profiler=True)
        is_tracing = True
        print('start tracing...')
    elif is_tracing:
        with writer.as_default():
            summary_ops_v2.trace_export(
                name='Default',
                step=step,
                profiler_outdir=os.path.join(logdir, 'train'))
        is_tracing = False
        print('export trace!')


def train(model, tensor_log, manager, init_epoch, train_set, test_set):
    logdir = os.path.join(params.logdir, model.name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    train_writer = tf.summary.create_file_writer(os.path.join(logdir, 'train'))
    test_writer = tf.summary.create_file_writer(os.path.join(logdir, 'test'))

    with train_writer.as_default():
        summary_ops_v2.graph(K.get_graph(), step=0)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    train_step = get_train_step(model, train_loss, train_accuracy)
    test_step = get_test_step(model, test_loss, test_accuracy)
    log_step = get_log_step(tensor_log, train_writer)

    do_callbacks('on_train_begin', model.callbacks)
    for epoch in range(init_epoch, params.training.epochs):
        do_callbacks('on_epoch_begin', model.callbacks, epoch=epoch)
        # Reset the metrics
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        tf.keras.backend.set_learning_phase(1)
        for batch, (images, labels) in enumerate(train_set):
            do_callbacks('on_batch_begin', model.callbacks, batch=batch)
            train_step(images, labels)
            if batch == 0 and params.training.log:
                log_step(images, labels, epoch)
            do_callbacks('on_batch_end', model.callbacks, batch=batch)
        do_callbacks('on_epoch_end', model.callbacks, epoch=epoch)
        # Get the metric results
        train_loss_result = float(train_loss.result())
        train_accuracy_result = float(train_accuracy.result())
        with train_writer.as_default():
            tf.summary.scalar('loss', train_loss_result, step=epoch+1)
            tf.summary.scalar('accuracy', train_accuracy_result, step=epoch+1)

        # Run a test loop at the end of each epoch.
        tf.keras.backend.set_learning_phase(0)
        for images, labels in test_set:
            test_step(images, labels)
        # Get the metric results
        test_loss_result = float(test_loss.result())
        test_accuracy_result = float(test_accuracy.result())
        with test_writer.as_default():
            tf.summary.scalar('loss', test_loss_result, step=epoch+1)
            tf.summary.scalar('accuracy', test_accuracy_result, step=epoch+1)

        print('Epoch:{}, train acc:{:f}, test acc:{:f}'.format(epoch + 1, train_accuracy_result, test_accuracy_result))
        if (params.training.save_frequency != 0 and epoch % params.training.save_frequency == 0) or epoch == params.training.epochs-1:
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(model.optimizer.iterations.numpy(), save_path))


def evaluate(model, test_set):
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    test_step = get_test_step(model, test_loss, test_accuracy)
    # Run a test loop at the end of each epoch.
    print('learning phase:', tf.keras.backend.learning_phase())
    for images, labels in test_set:
        test_step(images, labels)
    # Get the metric results
    test_loss_result = test_loss.result()
    test_accuracy_result = test_accuracy.result()

    print('test loss:{:f}, test acc:{:f}'.format(test_loss_result, test_accuracy_result))


def main(arguments):
    print(os.getcwd())
    train_set, test_set, info = data_input.build_dataset(params.dataset.name, batch_size=params.training.batch_size, flip=params.dataset.flip, crop=params.dataset.crop)
    model, tensor_log = import_module('models.' + params.model.name).build_model(shape=info.features['image'].shape,
                                                                                 num_out=info.features['label'].num_classes)
    progress = tf.keras.callbacks.ProgbarLogger('steps')
    progress.set_params({'verbose': True,
                         'epochs': int(arguments.epochs),
                         'metrics': '',
                         'steps': 1 + info.splits['train_examples'] // params.training.batch_size})
    model.callbacks.append(progress)

    params.logdir = os.path.join(params.logdir, params.dataset.name)
    print('config:', params)
    model_dir = os.path.join(params.logdir, model.name)

    ckpt = tf.train.Checkpoint(optimizer=model.optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
        init_epoch = params.training.batch_size * model.optimizer.iterations.numpy() // info.splits['train_examples']
    else:
        print("Initializing from scratch.")
        init_epoch = 0

    if arguments.train:
        train(model, tensor_log, manager, init_epoch, train_set, test_set)
    else:
        evaluate(model, test_set)


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(params.gpu)
    main(args)
