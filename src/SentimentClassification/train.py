import tensorflow as tf
import numpy as np
import os
from textCNN import TextCNN
from PreProcess import PreProcess
from config import *


def train(train, val, batch_size, max_steps, log_step, val_step, snapshot, out_dir):

    max_sentence_length = 10
    embedding_dim = 5
    filter_sizes = [3, 4, 5]
    num_filters = 200
    base_lr = 0.001 #学习率
    dropout_keep_prob = 0.5
    l2_reg_lambda = 0.0
    label_nums = 2

    allow_soft_placement = True  #如果指定的设备不存在，允许tf自动分配设备
    log_device_placement = False #是否打印设备分配日志

    print('Lodaing data...')
    preprocess = PreProcess()
    train_X_w2v, train_Y = preprocess.data_preprocess(train) #训练数据
    dev_X_w2v, dev_Y = preprocess.data_preprocess(val)

    #training
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=allow_soft_placement, log_device_placement=log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(sequence_length=max_sentence_length,
                          num_classes=label_nums,
                          embedding_size=embedding_dim,
                          filter_sizes=filter_sizes,
                          num_filters=num_filters,
                          l2_reg_lambda=l2_reg_lambda)

            global_step = tf.Variable(0, name='gloabl_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(learning_rate=base_lr)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_sumary = tf.summary.histogram('{}/grad/hist'.format(v.name), g)
                    sparsity_summary = tf.summary.scalar('{}/grad/sparsity'.format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_sumary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            print('Writing to {}\n'.format(out_dir))
            loss_summary = tf.summary.scalar('loss', cnn.loss)
            acc_summary = tf.summary.scalar('accuracy', cnn.accuracy)

            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, 'summaries', 'dev')
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
            checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

            sess.run(tf.global_variables_initializer())


            def train_step(x_batch, y_batch):
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                if step % log_step == 0:
                    print('training:step {}, loss {:g}'.format(step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict
                )
                if writer:
                    writer.add_summary(summaries, step)

                return loss, accuracy

            for i in range(max_steps):
                train_step(train_X_w2v, train_Y)
                current_step = tf.train.global_step(sess, global_step)

                if current_step % val_step == 0:
                    val_losses = []
                    val_accs = []

                    for k in range(100):
                        val_loss, val_acc = dev_step(dev_X_w2v, dev_Y)
                        val_losses.append(val_loss)
                        val_accs.append(val_acc)
                    mean_loss = np.array(val_losses, dtype=np.float32).mean()
                    mean_acc = np.array(val_accs, dtype=np.float32).mean()
                    print('------Evaluation:step{}, loss:{:g}, acc{:g}'.format(current_step, mean_loss, mean_acc))

                if current_step % snapshot == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print('Saved model checkpoint to {}\n'.format(path))

def main():

    max_steps = 10000
    batch_size = 5

    out_dir = './models'
    train_dir = train_data_path
    val_dir = dev_data_path

    train(train_dir,
          val_dir,
          batch_size,
          max_steps,
          log_step=50,
          val_step=500,
          snapshot=1000,
          out_dir=out_dir)


if __name__ == '__main__':
    main()