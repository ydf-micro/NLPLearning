import tensorflow as tf
import os
import time
import datetime
from textCNN import TextCNN
from PreProcess import PreProcess
from config import *

tf.flags.DEFINE_float('dev_sample_percentage', 0.1, 'Percentage of the training data to use for validation')

# Model Hyperparameters
tf.flags.DEFINE_integer('embedding_dim', 128, 'Dimensionality of character embedding(default: 128)')
tf.flags.DEFINE_string('filter_sizes', '3,4,5', 'Comma-separated filter sizes (default: "3,4,5")')
tf.flags.DEFINE_integer('num_filters', 128, 'Number of filters per filter size (default: 128)')
tf.flags.DEFINE_float('dropout_keep_prob', 0.5, 'Dropout keep probability (default: 0.5)')
tf.flags.DEFINE_float('l2_reg_lambda', 0.0, 'L2 regularization lambda (default: 0.0)')

# Training parameters
tf.flags.DEFINE_integer('batch_size', 64, 'Batch Size(Default: 64)')
tf.flags.DEFINE_integer('num_epochs', 200, 'Number of training epochs (default: 200)')
tf.flags.DEFINE_integer('evaluate_every', 100, 'Evaluate model on dev set after this many steps (default: 100)')
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer('num_checkpoints', 5, 'Number of checkpoints to store (default:5)')

# Misc Parameters
tf.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft device placement')
tf.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices')

FLAGS = tf.flags.FLAGS
FLAGS.flag_values_dict()
print('\nParameters:')
for attr, value in sorted(FLAGS.__flags.items()):
    print('{}={}'.format(attr.upper(), value))
print('')

def data_preprocess():
    print('Loading data...')
    preprocess = PreProcess()
    train_X_w2v, train_Y = preprocess.data_preprocess(train_data_path)  # 训练数据
    dev_X_w2v, dev_Y = preprocess.data_preprocess(dev_data_path)  #交叉验证数据

    return train_X_w2v, train_Y, dev_X_w2v, dev_Y

def train():
    print('start training...')
    train_X, train_Y, dev_X, dev_Y = data_preprocess()

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=train_X.shape[1],
                num_classes=train_Y.shape[1],
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(','))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # define Training procedure
            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)    # Adam optimization algorithm
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram('{}/grad/hist'.format(v.name), g)
                    sparsity_summary = tf.summary.scalar('{}/grad/spasity'.format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, 'logs', timestamp))
            print('Writting to {}\n'.format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar('loss', cnn.loss)
            acc_summary = tf.summary.scalar('accuracy', cnn.accuracy)

            # train summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, 'summaries', 'dev')
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # checkpoint directory. tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoint'))
            checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # initialize all variables
            sess.run(tf.global_variables_initializer())


            def train_step(x_batch, y_batch):
                # a single training step
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }

                _, step, summaries, loss, accuracy, result = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.result],
                    feed_dict)

                # print result
                time_str = datetime.datetime.now().isoformat()
                print('{}: 训练集 第{}次, loss:{:g}, accuracy:{:g}'.format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                # evaluates model on a dev set
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print('{}: 验证集 第{}次, loss: {:g}, accuracy: {:g}'.format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)
            # generate batches
            batches = PreProcess().batch_iter(
                list(zip(train_X, train_Y)), FLAGS.batch_size, FLAGS.num_epochs)

            # training loop. For each batch
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print('\nEvaluation:')
                    dev_step(dev_X, dev_Y, writer=dev_summary_writer)
                    print('')
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print('Saved model checkpoint to {}\n'.format(path))

if __name__ == '__main__':
    train()