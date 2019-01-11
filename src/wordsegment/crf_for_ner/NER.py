import re
from config import *
import CRFPP
from Pretreatment import *

class NER(object):
    def __init__(self):
        pass

    def tagging(self, word, pos):  #打标签
        tags = []
        if pos == 'ns':  # 地名
            tags += ['B_LOC'] + ['M_LOC'] * (len(word) - 2) + ['E_LOC']
        elif pos == 'nt':  # 组织机构名
            tags += ['B_ORG'] + ['M_ORG'] * (len(word) - 2) + ['E_ORG']
        elif pos == 'nz':  # 专有名词
            tags += ['B_PRO'] + ['M_PRO'] * (len(word) - 2) + ['E_PRO']
        elif pos == 'nr':  # 人名
            tags += ['S_PER'] if len(word) == 1 else ['B_PER'] + ['M_PER'] * (len(word) - 2) + ['E_PER']
        elif pos == 't':  # 时间
            tags += ['B_TIME'] + ['M_TIME'] * (len(word) - 2) + ['E_TIME']
        else:
            tags += ['O'] * len(word)

        return tags


    def tag_line(self, words_list, name_indicator, loc_indicator, org_indicator, time_indicator, pro_indicator):
        chars = []  # 词
        tags = []  # 命名实体标记
        pos = []  # 词性
        boundaries = []  # 词语边界
        name_in = []
        loc_in = []
        org_in = []
        time_in = []
        pro_in = []
        for word in words_list:
            w, h = word.split('/')
            flag = ['N']
            if word in name_indicator.keys():
                if name_indicator[word] >= 20:
                    flag = ['Y']
            name_in += flag * len(w)
            flag = ['N']
            if word in loc_indicator.keys():
                if loc_indicator[word] >= 20:
                    flag = ['Y']
            loc_in += flag * len(w)
            flag = ['N']
            if word in org_indicator.keys():
                if org_indicator[word] >= 20:
                    flag = ['Y']
            org_in += flag * len(w)
            flag = ['N']
            if word in time_indicator.keys():
                if time_indicator[word] >= 20:
                    flag = ['Y']
            time_in += flag * len(w)
            flag = ['N']
            if word in pro_indicator.keys():
                if pro_indicator[word] >= 20:
                    flag = ['Y']
            pro_in += flag * len(w)
            flag = ['N']

            chars.extend(w)
            tags += self.tagging(w, h)
            pos += len(w) * [h]
            boundaries += ['S'] if len(w) == 1 else ['B'] + ['M'] * (len(w) - 2) + ['E']

        return chars, pos, boundaries, tags, name_in, loc_in, org_in, time_in, pro_in

    def test_tagging(self, words_list, name_indicator, loc_indicator, org_indicator, time_indicator, pro_indicator):
        chars = []  # 词
        tags = []  # 命名实体标记
        pos = []  # 词性
        boundaries = []  # 词语边界
        name_in = []
        loc_in = []
        org_in = []
        time_in = []
        pro_in = []
        temp_words = ''
        for word in words_list:
            word = word.strip('\t ')
            test_word = re.sub(r'\[(.+)', '\\1', word)
            test_word = re.sub(r'(.+?)\][a-z]+', '\\1', test_word)
            w, h = test_word.split('/')
            flag = ['N']
            if word in name_indicator.keys():
                if name_indicator[word] >= 20:
                    flag = ['Y']
            name_in += flag * len(w)
            flag = ['N']
            if word in loc_indicator.keys():
                if loc_indicator[word] >= 20:
                    flag = ['Y']
            loc_in += flag * len(w)
            flag = ['N']
            if word in org_indicator.keys():
                if org_indicator[word] >= 20:
                    flag = ['Y']
            org_in += flag * len(w)
            flag = ['N']
            if word in time_indicator.keys():
                if time_indicator[word] >= 20:
                    flag = ['Y']
            time_in += flag * len(w)
            flag = ['N']
            if word in pro_indicator.keys():
                if pro_indicator[word] >= 20:
                    flag = ['Y']
            pro_in += flag * len(w)
            flag = ['N']

            if temp_words == '':
                bracket_pos = word.find('[')
                w, h = word.split('/')
                if bracket_pos == -1:
                    if len(w) == 0: continue
                    chars.extend(w)
                    tags += self.tagging(w, h)
                else:
                    w = w[bracket_pos + 1:]
                    temp_words += w
                pos += [h] * len(w)
                boundaries += ['S'] if len(w) == 1 else ['B'] + ['M'] * (len(w) - 2) + ['E']
            else:
                bracket_pos = word.find(']')
                w, h = word.split('/')
                boundaries += ['S'] if len(w) == 1 else ['B'] + ['M'] * (len(w) - 2) + ['E']
                if bracket_pos == -1:
                    temp_words += w
                    pos += [h] * len(w)
                else:
                    pos += [word[len(w) + 1:bracket_pos]] * len(w)
                    w = temp_words + w
                    h = word[bracket_pos + 1:]
                    temp_words = ''
                    if len(w) == 0: continue
                    chars.extend(w)
                    tags += self.tagging(w, h)
        assert temp_words == ''

        return chars, pos, boundaries, tags, name_in, loc_in, org_in, time_in, pro_in

    def load_copus(self, name_indicator, loc_indicator, org_indicator, time_indicator, pro_indicator):
        train_file = open(train_pretreat_path)
        tagging_file = open(train_tagging_path, 'w')

        for line in train_file:
            words = line.split()
            chars, pos, boundaries, tags, name_in, loc_in, org_in, time_in, pro_in = \
                self.tag_line(words, name_indicator, loc_indicator, org_indicator, time_indicator, pro_indicator)
            for k, v in enumerate(chars):
                tagging_file.write(v + '\t' + pos[k] + '\t' + boundaries[k] + '\t' + name_in[k] +
                                   '\t' + loc_in[k] + '\t' + org_in[k] + '\t' + time_in[k] + '\t' + pro_in[k]
                                   + '\t' + tags[k] + '\n')
            tagging_file.write('\n')

        train_file.close()
        tagging_file.close()


    def load_test_corpus(self, name_indicator, loc_indicator, org_indicator, time_indicator, pro_indicator):
        test_file = open(test_data_path)
        test_tagging_file = open(test_tagging_path, 'w')

        for line in test_file:
            line = re.sub(r'[0-9]{8}-[0-9]{2}-[0-9]{3}-[0-9]{3}/m', '', line).strip('\t\n\r ')
            line = re.sub(r'([\u4e00-\u9fa5]{1,3})/nr  ([\u4e00-\u9fa5]{1,3})/nr', '\\1\\2/nr', line)  # 将名字合并
            line = re.sub(r'(/[a-z]+)/[a-z%]+', '\\1', line).strip('\t\n\r ')
            if line == '': continue
            words_list = line.split()
            chars, pos, boundaries, tags, name_in, loc_in, org_in, time_in, pro_in = \
                self.test_tagging(words_list, name_indicator, loc_indicator, org_indicator, time_indicator, pro_indicator)

            # print(len(chars), len(name_in), len(loc_in), len(org_in), len(time_in), len(pro_in), len(tags))
            for k, v in enumerate(chars):
                test_tagging_file.write(v + '\t' + pos[k] + '\t' + boundaries[k] + '\t' + name_in[k] +
                                   '\t' + loc_in[k] + '\t' + org_in[k] + '\t' + time_in[k] + '\t' + pro_in[k]
                                   + '\t' + 'O' + '\n')
            test_tagging_file.write('\n')

        test_file.close()
        test_tagging_file.close()

    def recognize(self, tagger):
        test_file = open(test_data_path)
        test_result_file = open(test_result_path, 'w')


        for line in test_file:
            tagger.clear()
            for word in line.strip():
                word = word.strip()
                if word:
                    tagger.add((word + '\tO'))
            tagger.parse()
            size = tagger.size()
            xsize = tagger.xsize()
            for i in range(size):
                for j in range(xsize):
                    char = tagger.x(i, j)
                    tag = tagger.y2(i)
                    if tag == 'O':
                        test_result_file.write(char)
                    elif tag == 'B_LOC' or tag == 'B_ORG' or tag == 'B_PRO' or tag == 'B_PER' or tag == 'B_TIME':
                        test_result_file.write('(' + char)
                    elif tag == 'E_LOC' or tag == 'E_ORG' or tag == 'E_PRO' or tag == 'E_PER' or tag == 'E_TIME':
                        test_result_file.write(char + ')' + tag[2:])
                    else:
                        test_result_file.write(char)
            test_result_file.write('\n')

        test_file.close()
        test_result_file.close()

if __name__ == '__main__':
    pre = Pretreatment()
    pre.load_copus()
    name_indicator, loc_indicator, org_indicator, time_indicator, pro_indicator = pre.get_indicator()
    ner = NER()
    # ner.load_copus(name_indicator, loc_indicator, org_indicator, time_indicator, pro_indicator)
    ner.load_test_corpus(name_indicator, loc_indicator, org_indicator, time_indicator, pro_indicator)

    # tagger = CRFPP.Tagger('-m' + crf_model)
    # ner.recognize(tagger)