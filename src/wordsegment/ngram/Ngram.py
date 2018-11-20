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

class Ngram():
    '''
        2gram
    '''
    def __init__(self):
        self.__WordDict = {}
        self.__NextDict = {}
        self.__NextSize = 0
        self.__WordSize = 0

    def Training(self):
        '''
            获得每个词出现的个数 self.__WordDict
            获得每个词和紧接着那个词出现的个数self.__NextDict
        :return:
        '''
        print('start training...')
        start = time.clock()
        self.__NextDict['<BEG>'] = {}
        training_file = open(train_data_path)
        training_cnt = 0
        for line in training_file:
            line = line.strip().split(' ')
            line_list = []
            #每个单词出现的个数
            for pos, words in enumerate(line):
                if words != '' and words not in punctuation:
                    line_list.append(words)
            training_cnt += len(line_list)
            for pos, words in enumerate(line_list):
                if not self.__WordDict.__contains__(words):
                    self.__WordDict[words] = 1
                else:
                    self.__WordDict[words] += 1

                #每个单词后面那个单词出现的次数
                if pos == 0:
                    wordpre, wordnext = '<BEG>', words
                elif pos == len(line_list)-1:
                    wordpre, wordnext = words, '<END>'
                else:
                    wordpre, wordnext = words, line_list[pos+1]
                if not self.__NextDict.__contains__(wordpre):
                    self.__NextDict[wordpre] = {}
                if not self.__NextDict[wordpre].__contains__(wordnext):
                    self.__NextDict[wordpre][wordnext] = 1
                else:
                    self.__NextDict[wordpre][wordnext] += 1

        training_file.close()
        self.__NextSize = training_cnt
        print('总训练文本长度：', training_cnt)
        print('training done...')
        end = time.clock()
        print('训练用时：{}s'.format(end-start))
        self.__WordSize = len(self.__WordDict)
        print('训练集词表：', len(self.__WordDict))
        print('2-gram词表：', len(self.__NextDict))

    def PreProcessint(self, line, sentListcnt, SpecialDict):
        '''
        预处理
        :return:
        '''
        sentList = []
        tmp_words = ''

        # 记录是否有英文或者数字的flag
        flag = 0
        for word in line:
            if re.match(r'[ａ-ｚＡ-Ｚ０-９％%．.]', word):
                if tmp_words != '' and flag == 0:
                    sentList.append(tmp_words)
                    tmp_words = ''
                flag = 1
                tmp_words += word
            elif word in punctuation:
                if tmp_words != '':
                    sentList.append(tmp_words)
                    sentListcnt += 1
                    if flag == 1:
                        SpecialDict[tmp_words] = 1
                        flag = 0
                sentList.append(word)
                tmp_words = ''
            else:
                if flag == 1:
                    sentList.append(tmp_words)
                    sentListcnt += 1
                    SpecialDict[tmp_words] = 1
                    flag = 0
                    tmp_words = word
                else:
                    tmp_words += word
        if tmp_words != '':
            sentList.append(tmp_words)
            sentListcnt += 1
            if flag == 1:
                SpecialDict[tmp_words] = 1

        return sentList, sentListcnt, SpecialDict

    def SegmentWords(self, mode):
        '''
        分词
        :param mode:
        :return:
        '''
        print('start Segment Words')
        start = time.clock()
        test_file = open(test_data_path)
        test_result_file = open(test_result_path, 'w')

        sentListcnt = 0
        SpecialDict = {}
        for line in test_file:
            line = line.strip()
            tmp_words = ''

            sentList, sentListcnt, SpecialDict = self.PreProcessint(line, sentListcnt, SpecialDict)

            for sent in sentList:
                if sent not in punctuation and sent not in SpecialDict:
                    if mode == 'FMM':
                        ParseList = self.FMM(sent)
                    elif mode == 'RMM':
                        ParseList = self.RMM(sent)
                    else:
                        ParseListF = self.FMM(sent)
                        ParseListR = self.RMM(sent)
                        ParseListF.insert(0, '<BEG>')
                        ParseListF.append('<END>')
                        ParseListR.insert(0, '<BEG>')
                        ParseListR.append('<END>')
                        #根据钱箱最大匹配和后向最大匹配得到句子的两个词序列（添加BEG和END作为句子的开始和结束）

                        #记录最终选择的句子
                        ParseList = []

                        #两种分词结果不同的部分
                        DiffListF = []
                        DiffListR = []

                        #indexF, indexR记录两个句子的第几个词
                        indexF, indexR = 0, 0
                        #posF, posR记录两个句子当前词的位置
                        posF, posR = 0, 0
                        while(indexF < len(ParseListF) and indexR < len(ParseListR)):
                            if ParseListF[indexF] == ParseListR[indexR] and posF == posR:
                                #如果DiffListF不为空，就表明有不同的序列待处理
                                if DiffListF:
                                    #将第一个不同的词的前一个相同的词插入,计算出概率之后将该位去掉
                                    DiffListF.insert(0, ParseList[-1])
                                    DiffListR.insert(0, ParseList[-1])

                                    pF = self.CalProbability(DiffListF)
                                    pR = self.CalProbability(DiffListR)

                                    if pF > pR:
                                        ParseList.extend(DiffListF[1:])
                                    else:
                                        ParseList.extend(DiffListR[1:])

                                    DiffListF = []
                                    DiffListR = []

                                posF += len(ParseListF[indexF])
                                posR += len(ParseListR[indexR])
                                ParseList.append(ParseListF[indexF])
                                indexF += 1
                                indexR += 1
                            else:
                                if posF + len(ParseListF[indexF]) == posR + len(ParseListR[indexR]):
                                    DiffListF.append(ParseListF[indexF])
                                    DiffListR.append(ParseListR[indexR])
                                    posF += len(ParseListF[indexF])
                                    posR += len(ParseListR[indexR])
                                    indexF += 1
                                    indexR += 1
                                elif posF + len(ParseListF[indexF]) > posR + len(ParseListR[indexR]):
                                    DiffListR.append(ParseListR[indexR])
                                    posR += len(ParseListR[indexR])
                                    indexR += 1
                                else:
                                    DiffListF.append(ParseListF[indexF])
                                    posF += len(ParseListF[indexF])
                                    indexF += 1


                        ParseList.remove('<BEG>')
                        ParseList.remove('<END>')

                    for pos, words in enumerate(ParseList):
                        tmp_words += ' ' + words

                else:
                    tmp_words += ' ' + sent

            test_result_file.write(tmp_words.strip() + '\n')

        test_file.close()
        test_result_file.close()

        print('Segment Words done...')
        end = time.clock()
        print('分词用时：{}s'.format(end-start))
        print('句子列表长度：', sentListcnt)

    def CalProbability(self, ParseList):
        '''
        Calculate Probability
        :param ParseList:
        :return:
        '''
        p = 0
        #取对数，将连乘变成加,并进行拉普拉斯修正
        for pos, words in enumerate(ParseList):
            if pos < len(ParseList)-1:
                wordpre, wordnext = words, ParseList[pos+1]
                if not self.__NextDict.__contains__(wordpre):
                    p += math.log(1.0 / self.__NextSize)
                else:
                    mole, deno = 1.0, self.__NextSize
                    for key in self.__NextDict[wordpre]:
                        if key == wordnext:
                            mole += self.__NextDict[wordpre][wordnext]
                        deno += self.__NextDict[wordpre][key]
                    p += math.log(mole / deno)

            #计算条件词的概率
            if (pos == 0 and words != '<BEG>') or (pos == 1 and ParseList[0] == '<BEG>'):
                if self.__WordDict.__contains__(words):
                    p += math.log(float(self.__WordDict[words]) + 1 / (self.__WordSize + self.__NextSize))  #self.__NextSize为拉普拉斯平滑的分母处理
                else:
                    p += math.log(1 / (self.__WordSize + self.__NextSize))

        return p

    def FMM(self, sent):
        '''
        Forward Maximum Match Method
        :param sent:
        :return:
        '''
        cur, tail = 0, span
        ParseList  =[]

        while(cur < tail and cur <= len(sent)):
            if len(sent) < tail:
                tail = len(sent)
            if tail == cur+1:
                ParseList.append(sent[cur:tail])
                cur += 1
                tail = cur + span
            elif self.__WordDict.__contains__(sent[cur:tail]):
                ParseList.append(sent[cur:tail])
                cur = tail
                tail = cur + span
            else:
                tail -= 1

        return ParseList

    def RMM(self, sent):
        '''
        Reverse Maximum Match Method
        :param sent:
        :return:
        '''
        cur, tail = len(sent)-span, len(sent)
        ParseList = []
        if cur < 0:
            cur = 0

        while(cur < tail and tail > 0):
            if tail == cur+1:
                ParseList.append(sent[cur:tail])
                tail -= 1
                cur = tail -span
                if cur < 0:
                    cur = 0
            elif self.__WordDict.__contains__(sent[cur:tail]):
                ParseList.append(sent[cur:tail])
                tail = cur
                cur = tail - span
                if cur < 0:
                    cur = 0
            else:
                cur += 1
        ParseList.reverse()

        return ParseList

if __name__ == '__main__':
    ngram = Ngram()
    ngram.Training()

    '''FMM'''
    print('前向最大匹配')
    ngram.SegmentWords('FMM')
    e = Evaluate()
    e.evaluate()
    e.result()

    '''RMM'''
    print('逆向最大匹配')
    ngram.SegmentWords('RMM')
    e = Evaluate()
    e.evaluate()
    e.result()

    '''Ngram'''
    print('2gram模型匹配')
    ngram.SegmentWords('Ngram')
    e = Evaluate()
    e.evaluate()
    e.result()