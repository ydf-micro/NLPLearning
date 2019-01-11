import re
from config import *

class Pretreatment(object):
    def __init__(self):
        self.__name_indicator = {}  #人名指示词库
        self.__loc_indicator = {}   #地名指示词库
        self.__org_indicator = {}   #组织名指示词库
        self.__time_indicator = {}  #时间指示词库
        self.__pro_indicator = {}   #专有名词指示词库

    def get_indicator(self):
        return self.__name_indicator, self.__loc_indicator, self.__org_indicator, self.__time_indicator, self.__pro_indicator


    def pretreate(self, words_list): #对文本进行预处理
        new_words_list = []
        temp_words = ''
        for word in words_list:
            if temp_words == '':
                bracket_pos = word.find('[')
                w, h = word.split('/')
                if bracket_pos == -1: #没找到
                    new_words_list.append(word)
                else:
                    bracket_pos1 = word.find(']')
                    if bracket_pos1 != -1:
                        new_words_list.append(word[1:bracket_pos1])
                    else:
                        w = w[bracket_pos+1:]
                        temp_words += w
            else:
                bracket_pos = word.find(']')
                w, h = word.split('/')
                if bracket_pos == -1:
                    temp_words += w
                else:
                    h = word[bracket_pos+1:]
                    temp_words += w + '/' + h
                    new_words_list.append(temp_words)
                    temp_words = ''

        assert temp_words == ''
        return new_words_list

    def add_indicator(self, new_line, pos, loc):
        if pos == 'ns':  # 地名
            if loc != 0:
                if not self.__loc_indicator.__contains__(new_line[loc-1]):
                    self.__loc_indicator[new_line[loc - 1]] = 1
                else:
                    self.__loc_indicator[new_line[loc - 1]] += 1
            if loc != len(new_line)-1:
                if not self.__loc_indicator.__contains__(new_line[loc+1]):
                    self.__loc_indicator[new_line[loc + 1]] = 1
                else:
                    self.__loc_indicator[new_line[loc + 1]] += 1
        elif pos == 'nt':  # 组织机构名
            if loc != 0:
                if not self.__org_indicator.__contains__(new_line[loc-1]):
                    self.__org_indicator[new_line[loc - 1]] = 1
                else:
                    self.__org_indicator[new_line[loc - 1]] += 1
            if loc != len(new_line)-1:
                if not self.__org_indicator.__contains__(new_line[loc+1]):
                    self.__org_indicator[new_line[loc + 1]] = 1
                else:
                    self.__org_indicator[new_line[loc + 1]] += 1
        elif pos == 'nz':  # 专有名词
            if loc != 0:
                if not self.__pro_indicator.__contains__(new_line[loc-1]):
                    self.__pro_indicator[new_line[loc - 1]] = 1
                else:
                    self.__pro_indicator[new_line[loc - 1]] += 1
            if loc != len(new_line)-1:
                if not self.__pro_indicator.__contains__(new_line[loc+1]):
                    self.__pro_indicator[new_line[loc + 1]] = 1
                else:
                    self.__pro_indicator[new_line[loc + 1]] += 1
        elif pos == 'nr':  # 人名
            if loc != 0:
                if not self.__name_indicator.__contains__(new_line[loc-1]):
                    self.__name_indicator[new_line[loc - 1]] = 1
                else:
                    self.__name_indicator[new_line[loc - 1]] += 1
            if loc != len(new_line)-1:
                if not self.__name_indicator.__contains__(new_line[loc+1]):
                    self.__name_indicator[new_line[loc + 1]] = 1
                else:
                    self.__name_indicator[new_line[loc + 1]] += 1
        elif pos == 't':  # 时间
            if loc != 0:
                if not self.__time_indicator.__contains__(new_line[loc-1]):
                    self.__time_indicator[new_line[loc - 1]] = 1
                else:
                    self.__time_indicator[new_line[loc - 1]] += 1
            if loc != len(new_line)-1:
                if not self.__time_indicator.__contains__(new_line[loc+1]):
                    self.__time_indicator[new_line[loc + 1]] = 1
                else:
                    self.__time_indicator[new_line[loc + 1]] += 1

    def load_copus(self):
        train_pretreat_file = open(train_pretreat_path, 'w')
        train_file = open(train_data_path)
        for line in train_file:
            line = re.sub(r'[0-9]{8}-[0-9]{2}-[0-9]{3}-[0-9]{3}/m', '', line).strip('\t\n\r ')
            line = re.sub(r'([\u4e00-\u9fa5]{1,3})/nr  ([\u4e00-\u9fa5]{1,3})/nr', '\\1\\2/nr', line)  # 将名字合并
            line = re.sub(r'(/[a-z]+)/[a-z%]+', '\\1', line).strip('\t\n\r ')
            if line == '': continue
            words_list = line.split()
            new_line = self.pretreate(words_list)
            for i, words in enumerate(new_line):
                w, pos = words.split('/')
                self.add_indicator(new_line, pos, i)
                train_pretreat_file.write(words + ' ')
            train_pretreat_file.write('\n')
        train_file.close()

        train_pretreat_file.close()


if __name__ == '__main__':
    pre = Pretreatment()
    pre.load_copus()