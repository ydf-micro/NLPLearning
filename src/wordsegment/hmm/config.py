'''
Author: Ydf
Created: November 20, 2018
Version 1.0
Update:
'''

train_data_path = '../../../data/msr_train_preprocess.utf8'
test_data_path = '../../../data/msr_test.utf8'
test_result_path = '../../../data/msr_test_result.utf8'
test_gold_path = '../../../data/msr_test_gold.utf8'

punctuation = ['、', '”', '“', '。', '（', '）', '：', '《', '》', '；', '！', '，', '、', '…', '？', '?']

span = 12  #max word length