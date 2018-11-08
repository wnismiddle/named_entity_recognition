# -*- encoding:utf-8 -*-
import configparser
import numpy as np
import jieba.posseg as pseg
import logging

from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from itertools import chain

from app.utils.config import mlp_config
from app.utils.conlleval import *

ANY_SPACE = '<SPACE>'


def print_config(config, logger):
    """
    Print configuration of the model
    """
    for k, v in config.items():
        logger.info("{}:\t{}".format(k.ljust(15), v))

def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    labels_dense = np.asarray(labels_dense)
    num_labels = len(labels_dense)
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def construct_from_file(path):
    """ load data from file
            """
    sentences = []
    features = []
    tags = []

    sentence = []
    feature = []
    tag = []

    if path:
        # print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        # print(os.path.realpath(path))
        # print(os.path.realpath(__file__))
        with open(path, 'r', encoding='utf-8') as rFile:
            for line in rFile.readlines():
                arr = line.strip().split()

                if len(arr) == 0:
                    sentences.append(sentence)
                    features.append(feature)
                    tags.append(tag)
                    sentence = []
                    feature = []
                    tag = []
                else:
                    sentence.append(arr[0])
                    tag.append(arr[-1])
                    if len(arr) > 2:
                        feature.append(arr[1:-1])

    return sentences, tags, features

def POS_sentence(sentence):
    """ 必须设定好格式，不然总会出问题"""
    words = pseg.cut(''.join(sentence))
    pos = []
    for word, flag in words:
        pos.extend([flag] * len(word))

    if len(pos) != len(sentence):
        raise ValueError('POS error')
    return pos

def precision_score(y_pred, y_true, n_classes):
    if len(y_pred) != len(y_true):
        raise ValueError()

    tp = [0] * n_classes
    p = [0] * n_classes
    for i in range(len(y_pred)):
        if y_true[i] == y_pred[i]:
            tp[y_pred[i]] += 1
        p[y_pred[i]] += 1
    print(tp)
    print(p)
    return np.sum(tp[1:] / np.sum(p[1:]))

def recall_score(y_pred, y_true, n_classes):
    if len(y_pred) != len(y_true):
        raise ValueError()

    tp = [0] * n_classes
    t = [0] * n_classes
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            tp[y_pred[i]] += 1
        t[y_true[i]] += 1

    return np.sum(tp[1:] / np.sum(t[1:]))

def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.

    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels=[class_indices[cls] for cls in tagset],
        target_names=tagset,
        digits=4
    )

def bio_classification_report_wn(y_true, y_pred):
    def evaluate(y_true, y_pred, options=None):
        if options is None:
            options = parse_args([])  # use defaults

        counts = EvalCounts()
        num_features = None  # number of features per line
        in_correct = False  # currently processed chunks is correct until now
        last_correct = 'O'  # previous chunk tag in corpus
        last_correct_type = ''  # type of previously identified chunk tag
        last_guessed = 'O'  # previously identified chunk tag
        last_guessed_type = ''  # type of previous chunk tag in corpus

        for i in range(len(y_true)):
            for x, y in zip(y_true[i], y_pred[i]):
                features = []
                features.append(x)
                features.append(y)

                if num_features is None:
                    num_features = len(features)
                elif num_features != len(features) and len(features) != 0:
                    raise FormatError('unexpected number of features: %d (%d)' %
                                      (len(features), num_features))

                if len(features) < 2:
                    raise FormatError('unexpected number of features in line %s' % line)

                guessed, guessed_type = parse_tag(features.pop())
                correct, correct_type = parse_tag(features.pop())

                end_correct = end_of_chunk(last_correct, correct,
                                           last_correct_type, correct_type)
                end_guessed = end_of_chunk(last_guessed, guessed,
                                           last_guessed_type, guessed_type)
                start_correct = start_of_chunk(last_correct, correct,
                                               last_correct_type, correct_type)
                start_guessed = start_of_chunk(last_guessed, guessed,
                                               last_guessed_type, guessed_type)

                if in_correct:
                    if (end_correct and end_guessed and
                                last_guessed_type == last_correct_type):
                        in_correct = False
                        counts.correct_chunk += 1
                        counts.t_correct_chunk[last_correct_type] += 1
                    elif (end_correct != end_guessed or guessed_type != correct_type):
                        in_correct = False

                if start_correct and start_guessed and guessed_type == correct_type:
                    in_correct = True

                if start_correct:
                    counts.found_correct += 1
                    counts.t_found_correct[correct_type] += 1
                if start_guessed:
                    counts.found_guessed += 1
                    counts.t_found_guessed[guessed_type] += 1
                if correct == guessed and guessed_type == correct_type:
                    counts.correct_tags += 1
                counts.token_counter += 1

                last_guessed = guessed
                last_correct = correct
                last_guessed_type = guessed_type
                last_correct_type = correct_type

            if in_correct:
                counts.correct_chunk += 1
                counts.t_correct_chunk[last_correct_type] += 1

        return counts
    counts = evaluate(y_true, y_pred)
    report = report_notprint(counts)
    return report

def init_config():
    try:
        cf = configparser.ConfigParser()
        cf.read(os.path.dirname(os.path.dirname(__file__)) + '/config.ini', encoding='utf-8')
    except Exception as e:
        print(e)
    return

# 读取config.ini 配置文件
def get_config():
    try:
        dic = dict()
        l_dic = dict()
        cf = configparser.ConfigParser()
        # print(os.path.dirname(os.path.dirname(__file__)) + '/config.ini')
        cf.read(os.path.dirname(os.path.dirname(__file__)) + '/config.ini', encoding='utf-8')

        for sec in cf.sections():
            for key in cf.items(sec):
                l_dic[key[0]] = key[1]
            dic[sec] = l_dic
            l_dic = dict()
        return dic
    except Exception as e:
        print(e)
    return -1

def get_default_mlp_config(train=True):
    conf = mlp_config()

    config = get_config()
    conf.batch_size = int(config['mlp']['batch_size'])
    conf.hidden_units = [int(token) for token in config['mlp']['hidden_units'].split(' ')]
    conf.learning_rate = float(config['mlp']['learning_rate'])
    conf.tf_path = config['mlp']['train_tf_path'] if train else mlp_config['test_tf_path']
    conf.train_epochs = int(config['mlp']['train_epochs'])
    return conf

def get_model_path(model_name):
    config = get_config()
    model = config['model']
    name = 'train_' + model_name

    return model[name]

def get_index_type(sent, tag):
    """
    利用NER将句子中的实体用<ORG> </ORG>的形式包裹
    :param sent: 
    :param tag: 
    :return: 
    """
    if len(sent) != len(tag):
        raise ValueError('')

    index = 0
    entity_info = []

    while index < len(tag):
        if tag[index].split('-')[0] == 'B':
            entity_type = tag[index].split('-')[-1]
            begin = index
            index += 1

            while index < len(tag) and tag[index].split('-')[0] == 'I':
                index += 1
            end = index

            entity_info.append([begin, end, entity_type])
        else:
            index += 1

    return entity_info

def transform_reformat(sentence, index_type):
    new_sent = []
    last_index = 0
    for begin, end, type in index_type:
        new_sent.append(sentence[last_index:begin])
        new_sent.append("<" + type + ">")
        new_sent.append(sentence[begin:end])
        new_sent.append("</" + type + ">")
        last_index = end
    new_sent.append(sentence[last_index:])
    return ''.join(new_sent)

def get_entity_sent(sent, tag):
    if len(sent) != len(tag):
        raise ValueError('')

    index = 0
    entitys = []
    entity = []
    while index < len(tag):
        if tag[index].split('-')[0] == 'B':
            entity_name = tag[index].split('-')[-1]
            entity.append(sent[index])
            index += 1

            while index < len(tag) and tag[index].split('-')[0] == 'I':
                entity.append(sent[index])
                index += 1

            entitys.append([''.join(entity), entity_name])
            entity = []
        else:
            index += 1

    return entitys

def get_entity(sentences, tags):
    entity = []
    for i in range(len(sentences)):
        entity += get_entity_sent(sentences[i], tags[i])

    en_set = set()
    entity_set = []
    for i in range(len(entity)):
        if entity[i][0] not in en_set and len(entity[i][0]) > 1:
            en_set.add(entity[i][0])
            entity_set.append([entity[i][0], entity[i][1]])

    return entity_set

def clear_single_tag_sent(tag):
    if len(tag) == 0:
        return []

    elif len(tag) == 1:
        return ['O']

    else:
        if tag[0] != 'O' and tag[1] == 'O':
            tag[0] = 'O'
        if tag[-1] != 'O' and tag[-2] == 'O':
            tag[-1] = 'O'

        for i in range(1, len(tag) - 1):
            if tag[i] != 'O' and tag[i - 1] == 'O' and tag[i + 1] == 'O':
                tag[i] = 'O'

        return tag

def clear_single_tag(tags):
    return [clear_single_tag_sent(tag) for tag in tags]


import os

print(os.path.dirname(__file__))

if __name__ == "__main__":
    sent = list("我来到北京")
    tag = ['O', 'O', 'O', 'B-LOC', 'I-LOC']

    print(get_entity_sent(sent, tag))
