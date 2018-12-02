'''
Author: Ydf
Created: November 20, 2018
Version 1.0
Update:
'''

from config import test_gold_path
from config import test_result_path

class Evaluate():
    def __init__(self):
        self.test_result_file = open(test_result_path)
        self.test_gold_file = open(test_gold_path)

        self.result_cnt = 0.0
        self.gold_cnt = 0.0
        self.right_cnt = 0.0

    def evaluate(self):
        for result, gold in zip(self.test_result_file, self.test_gold_file):
            result_list = result.strip().split(' ')
            gold_list = gold.strip().split(' ')
            for words in gold_list:
                if words == '':
                    gold_list.remove(words)
            for words in result_list:
                if words == '':
                    result_list.remove(words)

            self.result_cnt += len(result_list)
            self.gold_cnt += len(gold_list)
            for words in result_list:
                if words in gold_list:
                    self.right_cnt += 1.0
                    gold_list.remove(words)

        self.test_result_file.close()
        self.test_gold_file.close()

    def result(self):
        P = self.right_cnt / self.result_cnt  # precision
        R = self.right_cnt / self.gold_cnt  # recall
        F1 = 2 * P * R / (P + R)

        print('\n命中{}个'.format(self.right_cnt))
        print('分词结果{}个'.format(self.result_cnt))
        print('标准结果{}个'.format(self.gold_cnt))
        print('查准率为：{:%}'.format(P))
        print('查全率为：{:%}'.format(R))
        print('F1为：{:%}'.format(F1))
        print('\n')

if __name__ == '__main__':
    e = Evaluate()
    e.evaluate()
    e.result()