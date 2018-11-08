# coding:utf-8
from . import Word2Vec
import os

class Dictionary():
    """
    提供word向量化接口，采用词窗口模型获取上下文。
    """
    dictionary = dict()
    vector_size = 0

    def __init__(self, path=None, flag=True):

        if path:
            self.load_dictionary(path, flag)

    def get_vector(self, word, default_vec):
        if not len(default_vec) == self.vector_size:
            raise ValueError(
                'default vector with size %d not equal to dictionary size %d' % (len(default_vec), self.vector_size))
        return list(self.dictionary.get(word, default_vec))

    def check(self, dic):
        if dic:
            return len(set([len(v) for k, v in dic.items()])) == 1

    def get_size(self, dic):
        return len(dic[list(dic.keys())[0]])

    def load_dictionary(self, path, flag):
        if flag:
            dictionary = dict()
            # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            # print(os.path.realpath(path))
            # print(os.path.realpath(__file__))
            # print(os.path.realpath("data/zh/zh.train"))     # 同级data目录
            # print(os.path.realpath("/data/zh/zh.train"))    # 变成该项目所在盘下的顶级目录
            # print(os.path.realpath("../data/zh/zh.train"))  # 到外层目录
            # print(os.path.realpath("../../data/zh/zh.train"))   # 到外两层目录

            w2v = Word2Vec.load(os.path.realpath(path))
            # w2v = Word2Vec.load(path)       # path如果是绝对路径会报错
            # NotImplementedError: unknown URI scheme 'd' in 'D://Program Files/Python/workspace/named_entity_recognition/data/tmp/word2vec/zh'
            # print('{}\n****************************************************\n{}'.format(w2v, w2v.wv.index2word))
            for key in w2v.wv.index2word:
            # for key in w2v.index2word:
            #  原代码报错：Exception: 'Word2Vec' object has no attribute 'index2word'
                dictionary[key] = w2v[key]

            self.vector_size = len(dictionary[list(dictionary.keys())[0]])
            self.dictionary = dictionary
            return
        else:
            dictionary = dict()
            with open(path, 'r') as rFile:
                for line in rFile.readlines():
                    arr = line.strip().split()
                    word = arr[0]
                    vec = arr[1:]
                    dictionary[word] = vec
                if self.check(dictionary):
                    self.vector_size = len(dictionary[list(dictionary.keys())[0]])
                    self.dictionary = dictionary

    def keys(self):
        return self.dictionary.keys()