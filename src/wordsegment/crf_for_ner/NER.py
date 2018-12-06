import re
from config import *

class NER(object):
    def __init__(self):
        pass

    def tagging(self, word, pos):
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

    def tag_line(self, words):
        chars = [] #词
        tags = [] #命名实体标记
        pos = [] #词性
        boundaries = [] #词语边界
        temp_words = ''
        for word in words:
            word = word.strip('\t ')
            if temp_words == '':
                bracket_pos = word.find('[')
                w, h = word.split('/')
                if bracket_pos == -1:
                    if len(w) == 0: continue
                    chars.extend(w)
                    tags += self.tagging(w, h)
                else:
                    w = w[bracket_pos+1:]
                    temp_words += w
                pos += [h] * len(w)
                boundaries += ['S'] if len(w) == 1 else ['B'] + ['M'] * (len(w) - 2) + ['E']
            else:
                bracket_pos = word.find(']')
                w, h = word.split('/')
                boundaries += ['S'] if len(w) == 1 else ['B'] + ['M'] * (len(w) - 2) + ['E']
                if bracket_pos == -1:
                    temp_words += w
                    pos += [h]*len(w)
                else:
                    pos += [word[len(w) + 1:bracket_pos]] * len(w)
                    w = temp_words + w
                    h = word[bracket_pos+1:]
                    temp_words = ''
                    if len(w) == 0: continue
                    chars.extend(w)
                    tags += self.tagging(w, h)
        assert temp_words == ''

        return chars, tags, pos, boundaries


    def load_copus(self):
        train_file = open(train_data_path)
        tagging_file = open(pos_tagging_path, 'w')

        for line in train_file:
            line = re.sub(r'[0-9]{8}-[0-9]{2}-[0-9]{3}-[0-9]{3}/m', '', line)
            line = re.sub(r'([\u4e00-\u9fa5]{1,3})/nr  ([\u4e00-\u9fa5]{1,3})/nr', '\\1\\2/nr', line) #将名字合并
            line = line.strip('\t\n\r ')
            if line == '': continue
            words = line.split()
            if len(words) == 0: continue
            line_chars, line_tags, line_pos, line_boundries = self.tag_line(words)
            for k, v in enumerate(line_chars):
                tagging_file.write(v + '\t' + line_pos[k] + '\t' + line_boundries[k] + '\t' + line_tags[k] + '\n')
            tagging_file.write('\n')

        train_file.close()
        tagging_file.close()


if __name__ == '__main__':
    ner = NER()
    ner.load_copus()
    print