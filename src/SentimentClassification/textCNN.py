from PreProcess import PreProcess
from config import *
import tensorflow as tf

class TextCNN(object):
    def __init__(self,
                 sequence_length, num_classes, embedding_size,
                 filter_sizes, num_filters, l2_reg_lambda=0.0):
        '''

        :param sequence_length = 10: 句子最大长度
        :param num_classes = 2: 输出层的类别数，这里只有1和0两种
        :param embedding_size = 128: 词向量长度
        :param filter_sizes = [3, 4, 5]: 卷积核大小
        :param num_filters = 200: 卷积核个数
        :param L2_reg_lambda: 正则化lambda
        '''

        #input, output, dropout, result 占位符
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.result = tf.placeholder(tf.float32, [None, num_filters * len(filter_sizes)], name='result') #获取池化后的向量

        l2_loss = tf.constant(0.0) #l2正则后的损失值

        #embedding layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.embedded_chars = self.input_x
            self.embedded_chars_expended = tf.expand_dims(self.embedded_chars, -1) #增加一个维度

        #create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes): #3, 4, 5
            with tf.name_scope('conv-maxpool-%s' % filter_size):
                #Convolution layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]  #200
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                #convolution layer
                conv = tf.nn.conv2d(self.embedded_chars_expended,
                                    W,
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name='conv')
                #nonlinearity activation funciton
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                #maxpooling layer
                pooled = tf.nn.max_pool(
                    h,  #feature map
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='pool')
                pooled_outputs.append(pooled)

        #combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        self.result = self.h_pool_flat

        #dropout
        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        with tf.name_scope('output'):
            W = tf.get_variable(
                'W',
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.softmax(tf.nn.xw_plus_b(self.h_drop, W, b, name='scores'))
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

        #calculate mean cross-entropy loss
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
        #accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')

if __name__ == '__main__':
    # preprocess = PreProcess()
    # data_X_w2v, data_Y = preprocess.data_preprocess(train_data_path)
    # print(data_X_w2v)
    # print(data_Y)