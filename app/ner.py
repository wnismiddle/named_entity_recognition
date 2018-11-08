import numpy as np

from app.main import estimators

from app.utils.utility import get_default_mlp_config, get_model_path, classification_report, \
    precision_score, recall_score, POS_sentence
from app.main.DataSet import get_window_data_set, get_crf_data_set

import app.utils.utility as uti


def train(path=None, crf_flag=True, mlp_flag=False):
    """
    第一步，根据配置文件获得初始模型参数，并初始化模型
    第二部，数据模型导入输入和接口
    第二步，根据NER的模型，转换数据的格式
    第三步，训练NER模型，并保存到指定的位置。

    :return:
    """

    # 还需要根据配置文件初始化

    if mlp_flag:
        train_set = get_window_data_set(path)
        print(train_set)
        conf = get_default_mlp_config()
        conf.n_classes = len(train_set.labels)
        conf.n_input = train_set.dictionary.vector_size * train_set._windows_size
        conf.labels = train_set.labels
        conf.tag_dict = train_set.tag_dict

        mlp = estimators.MultiLayerPerceptron(config=conf)
        mlp.fit(train_set)

        test_set = get_window_data_set(path, language='zh', train=False)
        # mlp.evaluate_wn(test_set)       # 重写评估方式，按识别出的实体个数统计P、R、F值
        mlp.evaluate(test_set)

        model_dir = get_model_path('mlp')
        mlp.save(model_dir)

    if crf_flag:
        train_set = get_crf_data_set(path)
        print(train_set)
        test_set = get_crf_data_set(path, train=False)
        print(test_set)

        model_dir = get_model_path('crf')
        crf = estimators.CRF(model_dir=model_dir)

        # 如果存在训练好的crf模型，则无须再训练
        # if not os.listdir(model_dir):
        crf.fit(train_set)
        crf.evaluate_wn(test_set)   # 重写评估方式，按识别出的实体个数统计P、R、F值
        crf.evaluate(test_set)

    return


def boosting(sentences, ):
    pass


def seq_pos(sentences):
    return [POS_sentence(sent) for sent in sentences]


def seq_ner(sentences, model):
    # 默认的输入的格式是list of str
    sentences = [list(sent.strip().replace(' ', '')) for sent in sentences]

    if model == 'mlp':
        # mlp 预测
        model_path = get_model_path('mlp')
        conf = estimators.load_conf(model_path)
        mlp = estimators.MultiLayerPerceptron(conf)
        mlp.load()
        test = get_window_data_set(train=False)
        tags = []
        for sentence in sentences:
            test.set_data(data=(sentence, [], []))
            mlp_pro = mlp.proba(test)
            # del mlp
            rev_dict = {k: v for v, k in conf.tag_dict.items()}
            pred = [np.argmax(mlp_pro[i]) for i in range(len(mlp_pro))]  # 获取每个字对应类别概率最大的类别下标
            tag = [rev_dict[tag] for tag in pred]
            tags.append(tag)
        return sentences, tags, uti.get_entity(sentences, tags)
        # return sentences, tags, uti.get_entity_sent(list(chain.from_iterable(sentences)), tags)
    else:
        # crf 预测
        model_path = get_model_path('crf')
        crf = estimators.CRF(model_dir=model_path)
        test = get_crf_data_set(data=(sentences, [], []), train=False)
        # test.set_data(data=(sentences, [], []))
        crf_pro = crf.proba(test)
        tags = crf.predict(test)
        tags = uti.clear_single_tag(tags)
        return sentences, tags, uti.get_entity(sentences, tags)







def seq_label(sentence, flags, ensemble=False):
    sent = list(sentence)

    if not flags:
        raise ValueError()

    r1 = []
    r2 = []
    y_true = []
    labels = []
    rev_dict = {}

    if flags['mlp']:
        model_path = get_model_path('mlp')
        conf = estimators.load_conf(model_path)
        labels = conf.labels

        mlp = estimators.MultiLayerPerceptron(conf)
        mlp.load()

        test = get_window_data_set(train=False)
        y_true = test.extract_label(test.tags)
        # test.set_data(data=([sent], [], []))

        rev_dict = {k: v for v, k in conf.tag_dict.items()}
        # print(list(rev_dict[token] for token in mlp.predict(test)))
        r1 = mlp.proba(test)

        # mlp.evaluate(test)
        # mlp.proba(DataSet.w2v_DataSet(data=([sent], [], []), dictionary_path=dic_path, word2vec=True, window_size=7,
        #                               vector_size=100))

    if flags['crf']:
        model_path = get_model_path('crf')

        crf = estimators.CRF(
            model_dir=model_path)

        test = get_crf_data_set(train=False)
        # test.set_data(data=([sent], [], []))
        # re = crf.predict_sent(DataSet.crf_DataSet(path=None, data=([sent], [], [])))
        r2 = crf.proba(test)
        # crf.evaluate(test)

        r_pred = np.argmax(r2, 1)
        # print(len(r2))
        # print(len(y_true))
        # print(precision_score(r_pred, y_true, len(labels)))

    y_pred = r1 * 0.2 + r2 * 0.8
    y_pred = np.argmax(y_pred, 1)

    print(precision_score(y_pred, y_true, len(labels)))
    print(recall_score(y_pred, y_true, len(labels)))
    print(classification_report(y_true, np.argmax(r2, 1), labels=list(range(1, len(labels))), target_names=labels[1:],
                                digits=4))
    print(classification_report(y_true, y_pred, labels=list(range(1, len(labels))), target_names=labels[1:], digits=4))


def seq_ner_rel(in_sentences, model):
    mid_sentence, tags, _ = seq_ner(in_sentences, model)
    out_sentence = []
    for i in range(len(mid_sentence)):
        mid_result = uti.get_index_type(mid_sentence[i], tags[i])
        out_sentence.append(uti.transform_reformat(''.join(mid_sentence[i]), mid_result))
    return out_sentence


if __name__ == "__main__":
    # train()
    train(crf_flag=False, mlp_flag=True)
