import tensorflow as tf
import numpy as np
from PreProcess import PreProcess
from config import *

# Eval Parameters
tf.flags.DEFINE_integer('batch_size', 64, 'Batch Size (default: 64)')
tf.flags.DEFINE_string('checkpoint_dir', 'logs/1547270686/checkpoint/', 'Checkpoint directory forom training run')

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
    test_X_w2v, test_Y = preprocess.data_preprocess(test_data_path)  # 测试数据

    return test_X_w2v, test_Y

def prediction():
    test_X, test_Y = data_preprocess()
    test_Y = np.argmax(test_Y, axis=1)


    print('\nPrediction\n')

    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # get the placeholders from the graph by name
            input_x = graph.get_operation_by_name('input_x').outputs[0]
            dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]

            # tensor we want to evaluate
            predictions = graph.get_operation_by_name('output/predictions').outputs[0]

            #generate batches for one epoch
            batches = PreProcess().batch_iter(list(test_X), FLAGS.batch_size, 1, shuffle=False)

            #collect the predictions here
            all_predictions = []

            for x_batch in batches:
                batch_predictions = sess.run(predictions, feed_dict={input_x: x_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

    correct_predictions = float(sum(all_predictions == test_Y))
    print('测试样例一共有: {}'.format(len(test_Y)))
    print('准确率为: {:.3%}'.format(correct_predictions/float(len(test_Y))))

if __name__ == '__main__':
    prediction()