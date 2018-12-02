import CRFPP
from config import *
from Evaluate import Evaluate

class CRF(object):
    def __init__(self):
        pass

    def tagging(self):
        training_file = open(train_data_path)
        tagging_file = open(tagging_data_path, 'w')

        for sent in training_file:
            word_list = sent.strip().split()
            for word in word_list:
                if word.__len__() == 1:
                    tagging_file.write(word + '\tS\n')
                else:
                    tagging_file.write(word[0] + '\tB\n')
                    for w in word[1:len(word)-1]:
                        tagging_file.write(w + '\tM\n')
                    tagging_file.write(word[-1] + '\tE\n')
            tagging_file.write('\n')

        training_file.close()
        tagging_file.close()

    def segment(self, tagger):
        test_file = open(test_data_path)
        test_result_file = open(test_result_path, 'w')

        for line in test_file:
            tagger.clear()
            for word in line.strip():
                word = word.strip()
                if word:
                    tagger.add((word + '\to\tB'))
            tagger.parse()
            size = tagger.size()
            xsize = tagger.xsize()
            for i in range(size):
                for j in range(xsize):
                    char = tagger.x(i, j)
                    tag = tagger.y2(i)
                    if tag == 'B':
                        test_result_file.write(' ' + char)
                    elif tag == 'M':
                        test_result_file.write(char)
                    elif tag == 'E':
                        test_result_file.write(char + ' ')
                    else:
                        test_result_file.write(' ' + char + ' ')
            test_result_file.write('\n')

        test_file.close()
        test_result_file.close()

if __name__ == '__main__':
    crf = CRF()
    # crf.tagging()
    tagger = CRFPP.Tagger('-m' + crf_model)
    crf.segment(tagger)
    e = Evaluate()
    e.evaluate()
    e.result()