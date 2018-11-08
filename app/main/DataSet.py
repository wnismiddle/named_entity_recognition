import warnings

warnings.filterwarnings("ignore")

import logging
import jieba

logger = logging.getLogger('jieba').setLevel(logging.WARNING)

import re
import numpy as np
from app.utils.utility import POS_sentence, dense_to_one_hot, construct_from_file, get_config, get_logger
from app.utils.Dictionary import Dictionary

from gensim.models import Word2Vec

from functools import reduce
from operator import or_
import itertools
import pickle


class Seq_DataSet(object):
    num_sequences = 0

    def __init__(self, path = None, data=None, ):
        self.sentences = []
        self.tags = []
        self.features = []

        if not path and not data:
            pass

        # load data to memory
        if path and not data:
            self.sentences, self.tags, self.features = construct_from_file(path)

        if data and not path:
            self.sentences, self.tags, self.features = data

        self.num_sequences = len(self.sentences)
        self.num_tokens = sum([len(sent) for sent in self.sentences])

        # construct dict
        tag_set = set()
        if len(self.tags) > 0:
            tag_set = reduce(or_, [set(tag) for tag in self.tags])
        self.labels = list(tag_set)
        self.labels.sort()
        if 'O' in self.labels:
            k = self.labels.index('O')
            self.labels[0], self.labels[k] = self.labels[k], self.labels[0]     # 将标注“O”的索引调至首位

        self.tag_dict = dict()
        for i, label in enumerate(self.labels):
            self.tag_dict[label] = i

    def __str__(self):
        return 'data set with %d sentences , %d tokens and %d tags %s' % (
            self.num_sequences, self.num_tokens, len(self.labels), str(self.labels))

    def set_data(self, path=None, data=None):
        if path :
            self.sentences, self.tags, self.features = construct_from_file(path)

        if data:
            self.sentences, self.tags, self.features = data

    def extract_document_from_sentence(self, sentence):
        return ''.join(sentence)

    def extract_documents(self, sentences, filename=None):
        documents = [self.extract_document_from_sentence(sent) for sent in sentences]
        if filename:
            with open(filename, 'w', encoding='utf-8') as wFile:
                for i in range(len(documents)):
                    wFile.write(documents[i])
                    wFile.write('\r\n')

        return documents

    def extract_entity_from_sentence(self, sentence, tag):
        if len(sentence) != len(tag):
            raise ValueError('not equal')
        step = 0
        entity = []
        entitys = []
        while step < len(sentence):
            if tag[step] != 'O':
                entity_value = tag[step].split('-')[-1]
                while step < len(sentence) and tag[step].split('-')[-1] == entity_value:
                    entity.append(sentence[step])
                    step += 1
                entitys.append(''.join(entity) + ' ' + entity_value)
                entity = []
            else:
                step += 1
        return entitys

    def extract_entity(self, sentences, tags, filename=None):
        num_sentences = len(sentences)
        entitys = []
        for i in range(num_sentences):
            entitys += self.extract_entity_from_sentence(sentences[i], tags[i])

        # 去除常用项
        entity_dict = dict()
        for i in range(len(entitys)):
            entity_value, entity_name = entitys[i].split()
            entity_dict[entity_value] = entity_name

        if filename:
            with open(filename, 'w', encoding='utf-8') as wFile:
                for key, value in entity_dict.items():
                    wFile.write(key + ' ' + value)
                    wFile.write('\r\n')

        return entitys

    def read_entity_dict(self, path):
        entity_dict = dict()

        with open(path, 'r', encoding='utf-8') as rFile:
            for line in rFile.readlines():
                key, value = line.split()
                entity_dict[key] = value
        return entity_dict

    def TAG_sentence(self, sentence, entity_dict):
        words = jieba.cut(sentence, cut_all=False)
        tags = []
        for word in words:
            if word in entity_dict.keys():
                tags.extend([entity_dict[word]] * len(word))
            else:
                tags.extend(['O'] * len(word))

        return tags

    def generate(self, sentences, entity_dict, filename=None, POS=True):
        """ generate NER-format data from raw documents and entity dict

        Arg:
            sentences : documents, ['我来到北京天安门', '这里的风景非常的漂亮'] list of sentences
            entity_dict : 自定义字典，通过将字典中key添加到jieba分词的字典中，保证能够分词能够分到所有的实体
            filename : 地址，生成的NER-format数据存放的地址

        Return:
            sentences:
            tags:
            features:

        Raise:


        """

        # sentences to 1-D format
        sentences_1d = []
        for i in range(len(sentences)):
            sentences_1d += sentences[i]

        print(len(sentences_1d))

        # pos stage
        POSs = [self.POS_sentence(sent) for sent in sentences]
        POS_1d = []
        for pos in POSs:
            POS_1d += pos

        print(len(POS_1d))

        # load self-entity-dict to jieba's dict
        for key, value in entity_dict.items():
            jieba.add_word(key, tag=value)

        # tag stage
        TAGs = [self.TAG_sentence(sent, entity_dict) for sent in sentences]
        TAG_1d = []
        for tag in TAGs:
            TAG_1d += tag

        print(len(TAG_1d))

        # write to the object file
        if filename:
            with open(filename, 'w', encoding='utf-8') as wFile:
                step = 0
                for num in [len(sent) for sent in sentences]:
                    for k in range(num):
                        wFile.write(sentences_1d[step] + ' ' + POS_1d[step] + ' ' + TAG_1d[step])
                        step += 1
                        wFile.write('\r\n')
                    wFile.write('\r\n')

        return


class w2v_DataSet(Seq_DataSet):
    dictionary = dict()

    def __init__(self,
                 path,
                 data,
                 window_size,
                 dictionary_path,
                 isWord2Vec,

                 ):
        super().__init__(path, data=data)

        self._windows_size = window_size
        self.dictionary = Dictionary(dictionary_path, isWord2Vec)

    def word2vec(self, word):
        """
        从字典中找到word对应的向量。
        如果是空格，则用全0做default
        如果是不存在的，则用uniform随机一个等长度的向量

        """
        default_vec = list(np.random.uniform(-0.25, 0.25, self.dictionary.vector_size))
        s = re.sub('[^0-9a-zA-Z]+', '', word)

        if word == 'space':
            return [0] * self.dictionary.vector_size

        if word in self.dictionary.keys():
            return self.dictionary.get_vector(word, default_vec)
        elif word.lower() in self.dictionary.keys():
            return self.dictionary.get_vector(word.lower(), default_vec)
        elif s in self.dictionary.keys():
            return self.dictionary.get_vector(s, default_vec)

        return list(default_vec)

    def extract_label_from_sentence(self, sentence, dense=False):
        return [self.tag_dict[tag] for tag in sentence]


        # return np.reshape(tags, (len(tags), -1))

    def extract_label(self, tags, dense=False):
        tags = [self.extract_label_from_sentence(tag) for tag in tags]
        labels = []
        for tag in tags:
            labels += tag

        if dense:
            return dense_to_one_hot(labels, len(self.tag_dict))

        return np.asarray(labels)

    def extract_feature_from_sentence(self, sentence):
        feature = []

        for i in range(len(sentence)):
            for k in [i + k - (self._windows_size - 1) / 2 for k in range(self._windows_size)]:
                if k in range(len(sentence)):
                    feature += self.word2vec(sentence[int(k)])

                else:
                    feature += self.word2vec('space')
        return feature

    def extract_features(self, sentences):
        features = []

        num_tokens = sum([len(sent) for sent in sentences])
        for sent in sentences:
            features += (self.extract_feature_from_sentence(sent))

        return np.reshape((features), (num_tokens, -1))
        # num_sentences = len(sentences)
        # if num_sentences == 1:
        #     return self.extract_feature_from_sentence(sentences[0])
        # else:
        #     features = self.extract_feature_from_sentence(sentences[0])
        #     for i in range(1, num_sentences):
        #         features = np.concatenate((features, self.extract_feature_from_sentence(sentences[i])))
        #
        #     return features.reshape(features.shape[0])

    def sent2features(self, batch_x, batch_y):
        """ according different models using different features selection methods
            in word window method
        :param batch_x:
        :param batch_y:
        :return:
        """

        features = []
        tags = []
        # sentences = get_sentences(filename)[sent_begin:sent_end]
        for sentence, label in zip(batch_x, batch_y):
            for i in range(len(sentence)):
                feature = []

                for j in [i + k - (self._windows_size - 1) / 2 for k in range(self._windows_size)]:
                    if j in range(len(sentence)):
                        feature.append(self.word2vec(sentence[int(j)]))
                    else:
                        feature.append(self.word2vec('space'))

                try:
                    if label[i].endswith('O'):
                        tag = np.asarray([1, 0, 0, 0, 0])
                    elif label[i].endswith('PER'):
                        tag = np.asarray([0, 1, 0, 0, 0])
                    elif label[i].endswith('LOC'):
                        tag = np.asarray([0, 0, 1, 0, 0])
                    elif label[i].endswith('ORG'):
                        tag = np.asarray([0, 0, 0, 1, 0])
                    elif label[i].endswith('MISC'):
                        tag = np.asarray([0, 0, 0, 0, 1])
                except Exception as e:
                    print(e)

                features.append(np.reshape(feature, (1, -1)))
                tags.append(tag)

        return np.reshape(features, (len(features), -1)), np.reshape(tags, (len(tags), -1))

    # def get_param(self):
    #     config = w2v_Config()
    #
    #     config.window_size = self._windows_size
    #
    #     return config

    def save(self, path):
        config = self.get_param()
        with open(path, 'wb') as w:
            pickle.dump(config, path)

    def load(self, path):
        with open(path, 'rb') as r:
            config = pickle.load(r)
            self.__init__(config)

    def next_batch(self, batch_size, shuffle=True):
        batch_x, batch_y = super().next_batch(batch_size, shuffle=False)
        batch_x, batch_y = self.sent2features(batch_x, batch_y)

        return batch_x, batch_y


class crf_DataSet(Seq_DataSet):
    def __init__(self,
                 path = None,
                 data=None,
                 language='zh'
                 ):
        self.count = 0
        super().__init__(path=path, data=data)
        self.language = language
        return

    def word2features(self, sent, pos_feature, i):
        self.count += 1
        if self.count % 10000 == 0:
            print(self.count)
        word = sent[i]
        pos_tag = pos_feature[i]
        # chunk_tag = sent[i][2]

        if self.language == 'zh':
            features = [
                'bias',
                'word = ' + word,
                'pos_tag = ' + pos_tag,
                'pos_tag[:2] = ' + pos_tag[:2],
            ]

            if i > 0:
                word1 = sent[i - 1]
                pos_tag1 = pos_feature[i - 1]
                features.extend([
                    '-1:word=' + word1,
                    '-1:pos_tag=' + pos_tag1,
                    '-1:pos_tag[:2]=' + pos_tag1[:2],

                ])
            else:
                features.append('BOS')

            if i < len(sent) - 1:
                word1 = sent[i + 1]
                pos_tag1 = pos_feature[i + 1]
                # chunktag1 = sent[i + 1][2]
                features.extend([
                    '+1:word=' + word1,
                    '+1:pos_tag=' + pos_tag1,
                    '+1:pos_tag[:2]=' + pos_tag1[:2],
                ])
            else:
                features.append('EOS')
            return features

        else:
            features = [
                'bias',
                'word.lower=' + word.lower(),
                'word[-3:]=' + word[-3:],
                'word[-2:]=' + word[-2:],
                'word.isupper=%s' % word.isupper(),
                'word.istitle=%s' % word.istitle(),
                'word.isdigit=%s' % word.isdigit(),
                'postag=' + pos_tag,
                # 'chunktag=' + chunk_tag,
                'postag[:2]=' + pos_tag[:2],
            ]

        if i > 0:
            word1 = sent[i - 1]
            postag1 = pos_feature[i - 1]

            # word1 = sent[i - 1][0]
            # postag1 = sent[i - 1][1]
            # chunktag1 = sent[i - 1][2]
            features.extend([
                '-1:word.lower=' + word1.lower(),
                '-1:word.istitle=%s' % word1.istitle(),
                '-1:word.isupper=%s' % word1.isupper(),
                '-1:postag=' + postag1,
                # '-1:chunktag=' + chunktag1,
                '-1:postag[:2]=' + postag1[:2],
            ])
        else:
            features.append('BOS')

        if i < len(sent) - 1:
            # print('*********************{}**********************'.format(sent[i+1]))

            # author : wn
            word1 = sent[i + 1]
            postag1 = pos_feature[i + 1]

            # word1 = sent[i + 1][0]
            # postag1 = sent[i + 1][1]

            # chunktag1 = sent[i + 1][2]
            features.extend([
                '+1:word.lower=' + word1.lower(),
                '+1:word.istitle=%s' % word1.istitle(),
                '+1:word.isupper=%s' % word1.isupper(),
                '+1:postag=' + postag1,
                # '+1:chunktag=' + chunktag1,
                '+1:postag[:2]=' + postag1
                ,
            ])
        else:
            features.append('EOS')

        return features

    def extract_features(self, sentences):
        # pos = self.POS_sentence(sentences)

        return [[self.word2features(sent, POS_sentence(''.join(sent)), i) for i in range(len(sent))] for sent in
                sentences]

    def extract_labels(self, tags):
        return [tag for tag in tags]


def get_window_data_set(path=None, data=None, language='zh', train = True):
    config = get_config()

    data_config = config['dataset']
    if not path and not data:
        path_train = data_config['zh_train'] if language == 'zh' else data_config['eng_train']
        path_test = data_config['zh_test'] if language == 'zh' else data_config['eng_test']
        path = path_train if train else path_test

    language_key = language + '_dictionary'
    dictionary = config[language_key]

    if config != -1:
        ww = w2v_DataSet(path=path, data=data, window_size=int(data_config['window_size']), dictionary_path=dictionary['path'], isWord2Vec=dictionary['flag'] == 'True')
        return ww


def get_crf_data_set(path = None, data = None, language = 'zh', train = True):
    config = get_config()

    data_config = config['dataset']
    if not path and not data:
        path_train = data_config['zh_train'] if language == 'zh' else data_config['eng_train']
        path_test = data_config['zh_test'] if language == 'zh' else data_config['eng_test']
        path = path_train if train else path_test

    return crf_DataSet(path = path, data = data, language=language)