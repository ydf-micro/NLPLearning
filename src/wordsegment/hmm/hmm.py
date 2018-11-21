'''
Author: Ydf
Created: November 17, 2018
Version 1.0
Update:
'''
import re
import math
import time
from config import *
from Evaluate import Evaluate

class HMM(object):
    def __init__(self):
        self.__initStatus = {} #初始概率
        self.__transProbMatrix = {} #状态转移矩阵
        self.__emitProbMatrix = {} #发射矩阵
        self.__status_count = {} #状态数量
        self.__status = ['B', 'M', 'E', 'S']
        self.__num = set()

    def initMatrix(self):
        for status in self.__status:
            self.__initStatus[status] = 0.0
            self.__transProbMatrix[status] = {}
            for label in self.__status:
                self.__transProbMatrix[status][label] = 0.0
            self.__emitProbMatrix[status] = {}
            self.__status_count[status] = 0.0

    def Training(self):
        print('start training...')
        start = time.clock()
        training_file = open(train_data_path)
        self.initMatrix() #初始化
        '''对测试数据预处理'''
        # for line in training_file:
        #     line = re.split(r'、|”|“|。|（|）|：|《|》|；|！|，|、|…|‘|’|？|-|－|\.', line.strip())
        #     for sent in line:
        #         if sent != '' and sent != '  ':
        #             test.write(sent.strip() + '\n')
        for line in training_file:
            sentStatus = ''
            sent = ''
            for words in line.strip().split('  '):
                sent += words
                if words.__len__() == 1:
                    sentStatus += 'S'
                    self.__status_count['S'] += 1
                else:
                    sentStatus += 'B' + 'M' * (len(words) - 2) + 'E'
                    self.__status_count['B'] += 1
                    self.__status_count['M'] += len(words) - 2
                    self.__status_count['E'] += 1
            for pos, word in enumerate(sent):
                self.__num.add(word)
                if not self.__emitProbMatrix[sentStatus[pos]].__contains__(word):
                    self.__emitProbMatrix[sentStatus[pos]][word] = 1
                else:
                    self.__emitProbMatrix[sentStatus[pos]][word] += 1
            for pos, status in enumerate(sentStatus):
                if pos != 0:
                    self.__transProbMatrix[sentStatus[pos-1]][status] += 1

        for pre in self.__status:
            if pre == 'M' or pre == 'E':
                self.__initStatus[pre] = -3.14e+100
            else:
                self.__initStatus[pre] = math.log(self.__status_count[pre] / (self.__status_count['B'] + self.__status_count['S']))
            for next in self.__status:
                self.__transProbMatrix[pre][next] /= self.__status_count[pre]
                if self.__transProbMatrix[pre][next] == 0.0:
                    self.__transProbMatrix[pre][next] = -3.14e+100
                else:
                    self.__transProbMatrix[pre][next] = math.log(self.__transProbMatrix[pre][next])
        for status in self.__emitProbMatrix:
            for word in self.__emitProbMatrix[status]:
                self.__emitProbMatrix[status][word] /= self.__status_count[status]
                self.__emitProbMatrix[status][word] = math.log(self.__emitProbMatrix[status][word])

        training_file.close()
        print('training done...')
        end = time.clock()
        print('训练用时：{}s'.format(end - start))
        print('初始矩阵', self.__initStatus)
        print('状态转移矩阵', self.__transProbMatrix)
        print('发射矩阵', self.__emitProbMatrix)

    def PreProcessint(self, line):
        '''
        预处理
        :return:
        '''
        sentList = []
        tmp_words = ''

        #记录非标点字段
        flag = 0
        for word in line:
            if word in punctuation:
                if flag == 1:
                    sentList.append(tmp_words)
                    tmp_words = ''
                    flag = 0
                sentList.append(word)
            else:
                flag = 1
                tmp_words += word
        if tmp_words != '':
            sentList.append(tmp_words)

        return sentList

    def SegmentWords(self):
        print('start Segment Words')
        start = time.clock()
        test_file = open(test_data_path)
        test_result_file = open(test_result_path, 'w')
        paths = []

        for line in test_file:
            line = line.strip()
            tmp_words = ''

            sentList = self.PreProcessint(line)
            for sent in sentList:
                if sent not in punctuation:
                    maxprob = 0
                    for pos, ch in enumerate(sent):
                        maximum = -3.14e+100
                        path = ''
                        for status in self.__status:
                            if not self.__emitProbMatrix[status].__contains__(ch):
                                # 当发射矩阵中没有该词的时候，自定义一个
                                self.__emitProbMatrix[status][ch] = math.log((len(self.__num) - 1)/self.__status_count[status])
                            if pos == 0:
                                prob = self.__initStatus[status] + self.__emitProbMatrix[status][ch]
                            else:
                                prob = maxprob + self.__transProbMatrix[paths[-1]][status] + self.__emitProbMatrix[status][ch]
                            if prob > maximum:
                                maximum = prob
                                path = status
                        maxprob = maximum
                        paths.append(path)
                        tmp_words += ch
                        if paths[-1] == 'S' or paths[-1] == 'E':
                            tmp_words += ' '

                else:
                    tmp_words += sent + ' '

            test_result_file.write(tmp_words + '\n')

        test_result_file.close()
        test_file.close()
        print('segment words done...')
        end = time.clock()
        print('分词用时：{}s'.format(end - start))

if __name__ == '__main__':
    hmm = HMM()
    hmm.Training()
    hmm.SegmentWords()
    e = Evaluate()
    e.evaluate()
    e.result()