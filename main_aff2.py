import os
from importlib import import_module
from tensorflow_core.python.ops import summary_ops_v2
from tensorflow_core.python.keras import backend as K
import tensorflow as tf
import numpy as np
from common.inputs import custom_reader

from config import params, parse_args


is_tracing = False


def get_extra_losses(model):
    loss = 0
    if len(model.losses) > 0:
        loss = tf.math.add_n(model.losses)
    return loss


def get_recons_loss(recons_img, images):
    distance = tf.pow(recons_img - images, 2)
    loss = tf.reduce_sum(distance, [-1, -2, -3])
    recons_loss = params.recons.balance_factor * tf.reduce_mean(loss)
    return recons_loss


def get_difference(labels, predictions):
    total = predictions.get_shape().as_list()[0]

    predictions_traditional = tf.one_hot(tf.argmax(predictions, axis=-1), 10)
    results = tf.reduce_sum(tf.abs(labels - predictions_traditional), -1)
    accuracy = tf.reduce_sum(tf.cast(tf.equal(results, 0), tf.float32)) / total

    return accuracy


def get_train_step(model):
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions, recons_img = model((images, labels))
            extra_loss = get_extra_losses(model)
            pred_loss = model.loss(labels, predictions)
            total_loss = pred_loss + extra_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return pred_loss, predictions

    return train_step


def get_test_step(model):
    @tf.function
    def test_step(images, labels):
        predictions, recons_img = model((images, labels))
        extra_loss = get_extra_losses(model)
        pred_loss = model.loss(labels, predictions)

        return pred_loss, predictions
    return test_step


def get_log_step(model_log, writer):

    # @tf.function
    def log_step(images, labels, step):
        logs = model_log.model((images, labels))
        with writer.as_default():
            model_log.summary(logs, step)
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


def train(model, model_log, manager, init_epoch, shift_train, shift_test, aff_test):
    logdir = os.path.join(params.logdir, model.name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    train_writer = tf.summary.create_file_writer(os.path.join(logdir, 'train'))
    test1_writer = tf.summary.create_file_writer(os.path.join(logdir, 'test1'))
    test2_writer = tf.summary.create_file_writer(os.path.join(logdir, 'test2'))

    with train_writer.as_default():
        summary_ops_v2.graph(K.get_graph(), step=0)

    loss = tf.keras.metrics.Mean(name='loss')
    acc = tf.keras.metrics.Mean(name='acc')

    train_step = get_train_step(model)
    test1_step = get_test_step(model)
    test2_step = get_test_step(model)
    train_log_step = get_log_step(model_log, train_writer)
    test1_log_step = get_log_step(model_log, test1_writer)
    test2_log_step = get_log_step(model_log, test2_writer)

    do_callbacks('on_train_begin', model.callbacks)
    for epoch in range(init_epoch, params.training.epochs):
        do_callbacks('on_epoch_begin', model.callbacks, epoch=epoch)
        # Reset the metrics
        loss.reset_states()
        acc.reset_states()

        step = 0
        tf.keras.backend.set_learning_phase(1)
        for batch, (images, labels) in enumerate(shift_train):
            do_callbacks('on_batch_begin', model.callbacks, batch=batch)
            pred_loss, prediction = train_step(images, labels)
            loss.update_state(pred_loss)
            acc.update_state(get_difference(labels, prediction))

            step = model.optimizer.iterations.numpy()
            if step > params.training.steps:
                break
            if step % params.training.log_steps == 0 and params.training.log:
                tf.keras.backend.set_learning_phase(0)
                train_log_step(images, labels, step)
                # Get the metric results
                train_loss_result = float(loss.result())
                train_acc_result = float(acc.result())
                with train_writer.as_default():
                    tf.summary.scalar('loss', train_loss_result, step=step)
                    tf.summary.scalar('accuracy', train_acc_result, step=step)
                    loss.reset_states()
                    acc.reset_states()

                if step % (10*params.training.log_steps) == 0:
                    # shift mnist
                    log_batch = np.random.randint(0, 500)
                    for batch, (images, labels) in enumerate(shift_test):
                        pred_loss, prediction = test1_step(images, labels)
                        loss.update_state(pred_loss)
                        acc.update_state(get_difference(labels, prediction))
                        if batch == log_batch:
                            test1_log_step(images, labels, step)
                    # Get the metric results
                    test1_loss_result = float(loss.result())
                    test1_acc_result = float(acc.result())
                    with test1_writer.as_default():
                        tf.summary.scalar('loss', test1_loss_result, step=step)
                        tf.summary.scalar('accuracy', test1_acc_result, step=step)
                        loss.reset_states()
                        acc.reset_states()

                # aff mnist
                log_batch = np.random.randint(0, 500)
                for batch, (images, labels) in enumerate(aff_test):
                    pred_loss, prediction = test2_step(images, labels)
                    loss.update_state(pred_loss)
                    acc.update_state(get_difference(labels, prediction))
                    if batch == log_batch:
                        test2_log_step(images, labels, step)
                # Get the metric results
                test2_loss_result = float(loss.result())
                test2_acc_result = float(acc.result())
                with test2_writer.as_default():
                    tf.summary.scalar('loss', test2_loss_result, step=step)
                    tf.summary.scalar('accuracy', test2_acc_result, step=step)
                    loss.reset_states()
                    acc.reset_states()

                tf.keras.backend.set_learning_phase(1)

            do_callbacks('on_batch_end', model.callbacks, batch=batch)
        do_callbacks('on_epoch_end', model.callbacks, epoch=epoch)

        if (params.training.save_frequency != 0 and epoch % params.training.save_frequency == 0) or epoch == params.training.epochs-1:
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(model.optimizer.iterations.numpy(), save_path))

        if step > params.training.steps:
            break


def evaluate(model, model_log, shift_test, aff_test):
    test1_loss = tf.keras.metrics.Mean(name='test1_loss')
    test1_acc = tf.keras.metrics.Mean(name='test1_acc')
    test2_loss = tf.keras.metrics.Mean(name='test2_loss')
    test2_acc = tf.keras.metrics.Mean(name='test2_acc')
    test1_step = get_test_step(model, test1_loss, test1_acc)
    test2_step = get_test_step(model, test2_loss, test2_acc)
    # Run a test loop at the end of each epoch.
    print('learning phase:', tf.keras.backend.learning_phase())
    for images, labels in shift_test:
        test1_step(images, labels)
    # Get the metric results
    test1_loss_result = test1_loss.result()
    print('test loss:{:f}'.format(test1_loss_result))

    for images, labels in aff_test:
        test2_step(images, labels)
    # Get the metric results
    test2_loss_result = test2_loss.result()
    print('test loss:{:f}'.format(test2_loss_result))


def main(args, params):
    print(os.getcwd())
    shift_train, shift_test, info = custom_reader.build_dataset('shift_mnist', batch_size=params.training.batch_size)
    _, aff_test, _ = custom_reader.build_dataset('aff_mnist', batch_size=params.training.batch_size)
    model, model_log, encoder, decoder = import_module('models.' + params.model.name).build_model(shape=info.features['image'].shape,
                                                                                                  num_out=info.features['label'].num_classes,
                                                                                                  params=params)
    progress = tf.keras.callbacks.ProgbarLogger('steps')
    progress.set_params({'verbose': True,
                         'epochs': int(params.training.epochs),
                         'metrics': '',
                         'steps': 1 + info.splits['train_examples'] // params.training.batch_size})
    model.callbacks.append(progress)

    params.logdir = os.path.join(params.logdir, 'aff_mnist')
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

    if args.train:
        train(model, model_log, manager, init_epoch, shift_train, shift_test, aff_test)
    else:
        evaluate(model, model_log, shift_test, aff_test)


if __name__ == "__main__":
    args, params = parse_args()
    main(args, params)
