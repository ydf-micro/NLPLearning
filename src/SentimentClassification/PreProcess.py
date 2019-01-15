
import os
import re
import jieba
import logging
from config import *
import numpy as np
import gensim.models.word2vec as w2v
'''
    对数据读取并进行预处理---------提数据集中的带有情感的句子，进行去标点、去停用词、分词、词向量化（这里用的是word2Vec）
'''
class PreProcess(object):
    def __init__(self):
        pass

    #读取数据
    def load_data(self, data_path, stopwords=stopwords_path):
        #读取停用词
        stopwords_list = []
        with open(stopwords, 'r') as f:
            for word in f.readlines():
                stopwords_list.append(word.strip())
        data = []
        #找出带有情感的句子及其分类
        with open(data_path, 'r') as f:
            for line in f:
                line = line.strip()
                sent = re.findall(r'[，。！？、；]*([,.+a-zA-Z0-9\u4e00-\u9fa5]*)<e[0-9]+-([0-1])>(-|[，。《》！？、；‘’：“”【】｛｝（）…~,.+a-zA-Z0-9\u4e00-\u9fa5]*)</e[0-9]+>([+a-zA-Z0-9\u4e00-\u9fa5]*)[，。！？、；]+', line)
                data.extend([w for w in sent])

        return data, stopwords_list

    #分词并去掉停用词、标点
    def del_valid_word(self, data, stopwords):
        data_X = []
        data_Y = []
        for x1, y, x2, x3 in data:
            # 去掉标点符号
            x = x1 + x2 + x3
            x = re.sub(r'[,，。《》！？、；‘’：“”【】｛｝（）…~a-zA-Z0-9+]+', ' ', x)
            x = ' '.join(jieba.cut(x))
            x = re.sub(r'[ ]+', ' ', x)
            x = x.split(' ')
            # 去掉停用词
            for word in x:
                if word in stopwords:
                    x.remove(word)
            # 去掉大于10个词的句子多余10个部分，不够10个的补<PAD>
            if len(x) < 10:
                x.extend((10 - len(x)) * ['<PAD>'])
            elif len(x) > 10:
                x = x[:10]
            data_X.append(x)
            if int(y) == 0:
                data_Y.append([1, 0])
            elif int(y) == 1:
                data_Y.append([0, 1])

        return data_X, np.array(data_Y)

    #以word2Vec的方式将词向量化
    def toword2Vec(self, data_X, embedding_size = 128):
        if os.path.exists(w2vmodel_path) == False:
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
            w2vmodel = w2v.Word2Vec(data_X, sg=1, size=embedding_size,  window=3,  min_count=5,  negative=3, sample=0.001, hs=1, workers=4)
            w2vmodel.save(w2vmodel_path)
        w2vmodel = w2v.Word2Vec.load(w2vmodel_path)
        data_X_ndarray = np.array(data_X)
        data_X_w2v = np.zeros((data_X_ndarray.shape[0], data_X_ndarray.shape[1], embedding_size), dtype=float)
        for i in range(data_X_ndarray.shape[0]):
            for j in range(data_X_ndarray.shape[1]):
                if data_X_ndarray[i, j] in w2vmodel:
                    data_X_w2v[i, j, :] = w2vmodel[data_X_ndarray[i, j]]
        return data_X_w2v

    def data_preprocess(self, path):
        data, stopwords_list = self.load_data(path)
        data_X, data_Y = self.del_valid_word(data, stopwords_list)
        data_X_w2v = self.toword2Vec(data_X, )

        return data_X_w2v, data_Y

    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        # generates a batch iterator for a dataset.
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
        for epoch in range(num_epochs):
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

if __name__ == '__main__':
    preprocess = PreProcess()
    data_X_w2v, data_Y = preprocess.data_preprocess(train_data_path)
    print(data_X_w2v)
    print(data_Y)
    print(data_X_w2v.shape)
    print(data_Y.shape)
    print(len(data_Y))
    print(len(list(zip(data_X_w2v, data_Y))))
    print(data_Y)
    print(np.argmax(data_Y, axis=1))