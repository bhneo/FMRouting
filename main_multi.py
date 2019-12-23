import os
from importlib import import_module
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.keras import backend as K
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


def get_difference(labels, predictions, threshold):
    total = predictions.get_shape().as_list()[0]
    _, indices = tf.nn.top_k(predictions, 2)
    predictions_traditional = tf.one_hot(indices[:, 0], 10) + tf.one_hot(indices[:, 1], 10)
    results = tf.reduce_sum(tf.abs(labels - predictions_traditional), -1)
    both_correct = tf.reduce_sum(tf.cast(tf.equal(results, 0), tf.float32))
    partly_correct = tf.reduce_sum(tf.cast(tf.equal(results, 2), tf.float32))

    accuracy_both = both_correct / total
    accuracy_part = (both_correct*2+partly_correct) / (2*total)

    # predictions_threshold = tf.cast(tf.greater(predictions, threshold), tf.float32)
    # results = tf.reduce_sum(tf.abs(labels - predictions_threshold), -1)
    # accuracy_threshold = tf.reduce_sum(tf.cast(tf.equal(results, 0), tf.float32)) / total
    return accuracy_both, accuracy_part


def get_train_step(model):
    @tf.function
    def train_step(images, labels, image1, image2, label1, label2):
        with tf.GradientTape() as tape:
            predictions, recons_img1, recons_img2 = model((images, image1, image2, label1, label2))
            extra_loss = get_extra_losses(model)
            pred_loss = model.loss(labels, predictions)
            total_loss = pred_loss + extra_loss

        gradients = tape.gradient(total_loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return pred_loss, predictions

    return train_step


def get_test_step(model):
    @tf.function
    def test_step(images, labels, image1, image2, label1, label2):
        predictions, recons_img1, recons_img2 = model((images, image1, image2, label1, label2))
        extra_loss = get_extra_losses(model)
        pred_loss = model.loss(labels, predictions)

        return pred_loss, predictions
    return test_step


def get_log_step(model_log, writer):

    # @tf.function
    def log_step(images, labels, image1, image2, label1, label2, step):
        logs = model_log.model((images, image1, image2, label1, label2))
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


def parse_feature(features):
    images = features['images']
    labels = features['labels']
    recons_labels = features['recons_label']
    recons_images = features['recons_image']
    spare_labels = features['spare_label']
    spare_images = features['spare_image']
    return images, labels, recons_images, recons_labels, spare_images, spare_labels


def train(model, model_log, manager, init_epoch, train_set, test_set):
    logdir = os.path.join(params.logdir, model.name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    train_writer = tf.summary.create_file_writer(os.path.join(logdir, 'train'))
    test_writer = tf.summary.create_file_writer(os.path.join(logdir, 'test'))

    with train_writer.as_default():
        summary_ops_v2.graph(K.get_graph(), step=0)

    loss = tf.keras.metrics.Mean(name='loss')
    acc_both = tf.keras.metrics.Mean(name='acc_both')
    acc_part = tf.keras.metrics.Mean(name='acc_part')

    train_step = get_train_step(model)
    test_step = get_test_step(model)
    train_log_step = get_log_step(model_log, train_writer)
    test_log_step = get_log_step(model_log, test_writer)

    do_callbacks('on_train_begin', model.callbacks)
    for epoch in range(init_epoch, params.training.epochs):
        do_callbacks('on_epoch_begin', model.callbacks, epoch=epoch)
        # Reset the metrics
        loss.reset_states()
        acc_both.reset_states()
        acc_part.reset_states()

        tf.keras.backend.set_learning_phase(1)
        for batch, features in enumerate(train_set):
            images, labels, image1, label1, image2, label2 = parse_feature(features)
            do_callbacks('on_batch_begin', model.callbacks, batch=batch)
            pred_loss, predictions = train_step(images, labels, image1, image2, label1, label2)
            # Update the metrics
            loss.update_state(pred_loss)
            acc1, acc2 = get_difference(labels, predictions, params.recons.threshold)
            acc_both.update_state(acc1)
            acc_part.update_state(acc2)
            step = model.optimizer.iterations.numpy()
            if step > params.training.steps:
                break
            if step % params.training.log_steps == 0 and params.training.log:
                train_log_step(images, labels, image1, image2, label1, label2, step)
                # Get the metric results
                train_loss_result = float(loss.result())
                train_acc_both_result = float(acc_both.result())
                train_acc_part_result = float(acc_part.result())
                with train_writer.as_default():
                    tf.summary.scalar('loss', train_loss_result, step=step)
                    tf.summary.scalar('accuracy_both', train_acc_both_result, step=step)
                    tf.summary.scalar('accuracy_part', train_acc_part_result, step=step)
                    loss.reset_states()
                    acc_both.reset_states()
                    acc_part.reset_states()

                tf.keras.backend.set_learning_phase(0)
                log_batch = np.random.randint(0, 500)
                for batch, features in enumerate(test_set):
                    images, labels, image1, label1, image2, label2 = parse_feature(features)
                    pred_loss, predictions = test_step(images, labels, image1, image2, label1, label2)
                    # Update the metrics
                    loss.update_state(pred_loss)
                    acc1, acc2 = get_difference(labels, predictions, params.recons.threshold)
                    acc_both.update_state(acc1)
                    acc_part.update_state(acc2)
                    if batch == log_batch:
                        test_log_step(images, labels, image1, image2, label1, label2, step)
                # Get the metric results
                test_loss_result = float(loss.result())
                test_acc_both = float(acc_both.result())
                test_acc_part = float(acc_part.result())
                with test_writer.as_default():
                    tf.summary.scalar('loss', test_loss_result, step=step)
                    tf.summary.scalar('accuracy_both', test_acc_both, step=step)
                    tf.summary.scalar('accuracy_part', test_acc_part, step=step)
                    loss.reset_states()
                    acc_both.reset_states()
                    acc_part.reset_states()

                tf.keras.backend.set_learning_phase(1)

            do_callbacks('on_batch_end', model.callbacks, batch=batch)
        do_callbacks('on_epoch_end', model.callbacks, epoch=epoch)

        if (params.training.save_frequency != 0 and epoch % params.training.save_frequency == 0) or epoch == params.training.epochs-1:
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(model.optimizer.iterations.numpy(), save_path))

        if step > params.training.steps:
            break

def evaluate(model, model_log, test_set):
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_acc1 = tf.keras.metrics.Mean(name='test_acc1')
    test_acc2 = tf.keras.metrics.Mean(name='test_acc2')
    test_step = get_test_step(model, test_loss, test_acc1, test_acc2)
    # Run a test loop at the end of each epoch.
    print('learning phase:', tf.keras.backend.learning_phase())
    for features in test_set:
        images, labels, image1, label1, image2, label2 = parse_feature(features)
        test_step(images, labels, image1, image2, label1, label2)
    # Get the metric results
    test_loss_result = test_loss.result()

    print('test loss:{:f}'.format(test_loss_result))


def main(args, params):
    print(os.getcwd())
    train_set, test_set, info = custom_reader.build_dataset('multi_mnist', batch_size=params.training.batch_size)
    model, model_log, encoder, decoder = import_module('models.' + params.model.name).build_model(shape=info.features['image'].shape,
                                                                                                  num_out=info.features['label'].num_classes,
                                                                                                  params=params)
    progress = tf.keras.callbacks.ProgbarLogger('steps')
    progress.set_params({'verbose': True,
                         'epochs': int(params.training.epochs),
                         'metrics': '',
                         'steps': 1 + info.splits['train_examples'] // params.training.batch_size})
    model.callbacks.append(progress)

    params.logdir = os.path.join(params.logdir, 'multi_mnist')
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
        train(model, model_log, manager, init_epoch, train_set, test_set)
    else:
        evaluate(model, model_log, test_set)


if __name__ == "__main__":
    args, params = parse_args()
    main(args, params)
