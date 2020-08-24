import os
import re

codePath = '/Users/bytedance/codebase/'

class Vocab(object):
    # 一个词典包含了word->idx 和 idx->word
    def __init__(self):
        self.word2id = {}
        self.id2word = {}

    # 构建词典
    def construct_vocab(self, sentence):
        # 统计sentences中所有词出现的频次
        word_counter = {}
        for word in sentence:
            word_counter[word] = word_counter.setdefault(word, 0) + 1

        # 用sentences中的word更新word2id和id2word
        for word in word_counter.keys():
            idx = len(self.word2id)
            # 如果word已经在词典中则idx不变，反之将其添加到字典中
            self.word2id.setdefault(word, idx)
            self.id2word[idx] = word

vocab_token = Vocab()

def get_words(sentence):
    result_list = re.findall('[a-zA-Z0-9]+', sentence)
    return result_list

if __name__ == '__main__':
    for curp in os.listdir(codePath):
        print("Start to read {}'s Methods...".format(curp))
        for f in os.listdir(codePath + curp + '/allJavaFuncs/'):
            cur = os.path.join(codePath + curp + '/allJavaFuncs/', f)
            with open(cur, 'r') as fx:
                allwords = get_words(fx.readline())
                vocab_token.construct_vocab(allwords)