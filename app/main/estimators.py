import tensorflow as tf
import numpy as np
import math
import random
import pickle
import pycrfsuite

from sklearn.metrics import classification_report
from itertools import chain

from . import DataSet
from ..utils import utility
from app.utils.config import mlp_config
from ..utils.utility import get_config

config = get_config()

class BaseEstimator(object):
    def __init__(self):
        return

    def fit(self, X, y):
        return

    def predict(self, X):
        return

    def save(self, path):
        return

    def load(self,):
        return


class MultiLayerPerceptron(BaseEstimator):
    graph = None
    x = y = None
    prediction = cost = optimizer = None
    weights = biases = layers = dict()
    init = None
    saver = None
    sess = None
    labels = []
    tag_dict = {}

    def __init__(self, config=mlp_config()):
        super().__init__()
        """" initialize the DNN network """
        self._hidden_units = config.hidden_units
        self.n_input = config.n_input
        self.n_classes = config.n_classes
        self._learning_rate = config.learning_rate
        self.tf_path = config.tf_path
        self.batch_size = config.batch_size
        self._train_epochs = config.train_epochs
        self.tag_dict = config.tag_dict
        self.labels = config.labels

        self.construct_graph()

    def __str__(self):
        params = self.get_params()
        return "multi-layer-perceptron with %d input %d output and hidden units %s , " \
               "learning_rate : %s batch_size : %s train_epochs : %s , " \
               "tensorflow model saved in %s " \
               %(params.n_input, params.n_classes, str(params.hidden_units),
                 params.learning_rate, params.batch_size, params.train_epochs,
                 params.tf_path
                 )

    def set(self, config):
        self._hidden_units = config.hidden_units
        self.n_input = config.n_input
        self.n_classes = config.n_classes
        self._learning_rate = config.learning_rate
        self.tf_path = config.tf_path
        self.batch_size = config.batch_size
        self._train_epochs = config.train_epochs

    def construct_graph(self):
        """

        :return:
        """
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.weights = dict()
            self.biases = dict()
            self.layers = dict()

            self.x = tf.placeholder('float', [None, self.n_input], name='input_x')
            self.y = tf.placeholder('float', [None, self.n_classes], name='input_y')
            self.prediction = self._construct_net()
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.prediction))
            # self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.prediction, self.y))

            # ValueError: Only call softmax_cross_entropy_with_logits with named arguments (labels=…, logits=…, …)
            # 调用tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels) 出现上述错误。
            # 解决办法：错误提示中已经给出，只能使用命名参数的方式来调用。调用函数改为：tf.nn.softmax_cross_entropy_with_logits(labels=logits, logits=tf_train_labels) 即可。

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(self.cost)

            self.init = tf.global_variables_initializer()

            self.saver = tf.train.Saver()
            self.sess = tf.Session(graph=self.graph)

        return

    def get_params(self):
        config = mlp_config()
        config.hidden_units = self._hidden_units
        config.batch_size = self.batch_size
        config.learning_rate = self._learning_rate
        config.n_classes = self.n_classes
        config.n_input = self.n_input
        config.train_epochs = self._train_epochs
        config.tf_path = self.tf_path
        config.labels = self.labels
        config.tag_dict = self.tag_dict

        return config

    def _full_connected_layer(self, in_op, input_unit, output_unit, layer_index, out_layer=False):

        if layer_index >= len(self._hidden_units):
            raise ValueError()

        if out_layer:
            layer_name = 'out'
        else:
            layer_name = 'hidden_' + str(layer_index)

        weight = tf.Variable(
            tf.random_uniform([input_unit, output_unit], minval=- math.sqrt(6) / math.sqrt(input_unit + output_unit),
                              maxval=math.sqrt(6) / math.sqrt(input_unit + output_unit)),
            name='w_' + layer_name)
        bias = tf.Variable(tf.random_normal([output_unit]), name='b_' + layer_name)

        layer = tf.add(tf.matmul(in_op, weight), bias)
        if not out_layer:
            layer = tf.nn.relu(layer)

        self.weights[layer_name] = weight
        self.biases[layer_name] = bias
        self.layers[layer_name] = layer

        return layer

    def _construct_net(self):
        n_layers = len(self._hidden_units)

        # input layer
        out_op = self._full_connected_layer(self.x, self.n_input, self._hidden_units[0], 0)

        # hidden layer
        for i in range(1, len(self._hidden_units)):
            out_op = self._full_connected_layer(out_op, self._hidden_units[i - 1], self._hidden_units[i], i)

        # out layer
        out_layer = self._full_connected_layer(out_op, self._hidden_units[-1], self.n_classes, i, out_layer=True)
        return out_layer

    # def fit_tf(self, filename):
    #     data, label = reader.read_decode_single_example(filename)
    #
    #     batch_data, batch_label = tf.train.shuffle_batch(
    #         [data, label], batch_size=self.batch_size,
    #         capacity=2000,
    #         min_after_dequeue=1000
    #     )
    #
    #     config = tf.ConfigProto(device_count={"CPU": 4},
    #                             inter_op_parallelism_threads=1,
    #                             intra_op_parallelism_threads=1,
    #                             log_device_placement=True
    #                             )
    #
    #     sess = tf.Session(graph=self.graph, config=config)
    #     sess.run(self.init)
    #
    #     tf.train.start_queue_runners(sess=sess)
    #
    #     step = 0
    #     while step < self._stop_step:
    #         batch_x, batch_y = sess.run([batch_data, batch_label])
    #
    #         batch_y = np.reshape(batch_y, (len(batch_y), -1))
    #         batch_y = conllner.dense_to_one_hot(batch_y, 5)
    #
    #         _, c = sess.run([self.optimizer, self.cost], feed_dict={
    #             self.x: batch_x,
    #             self.y: batch_y
    #         })
    #
    #         if step % 50 == 0:
    #             print('cost : ', "{:.9f}".format(c))
    #
    #         step += 1
    #     self._saver.save(sess, self.model_dir)
    #     return

    def fit(self, DataSet):
        """ fit the model by stochastic gradient decay
        """


        self.sess.run(self.init)

        for epoch in range(self._train_epochs):
            avg_cost = 0

            total_batch = math.ceil(DataSet.num_sequences / self.batch_size)

            batch_list = [index for index in range(total_batch)]
            random.shuffle(batch_list)

            for batch_i in batch_list:
                train_x = DataSet.extract_features(
                    DataSet.sentences[batch_i * self.batch_size:(batch_i + 1) * self.batch_size])
                train_y = DataSet.extract_label(DataSet.tags[batch_i * self.batch_size:(batch_i + 1) * self.batch_size],
                                                dense=True)

                _, c = self.sess.run([self.optimizer, self.cost], feed_dict={self.x: train_x, self.y: train_y})
                avg_cost += c / total_batch

            print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                  "{:.9f}".format(avg_cost))

        self.save_tf()

        return

    def save_tf(self):
        self.saver.save(self.sess, self.tf_path)

    def save(self, model_path):
        # self._saver.save(self._sess, model_path)
        # self.model_dir = model_path
        config = self.get_params()
        with open(model_path, 'wb') as w:
            pickle.dump(config, w)

        return



    def load(self, ):
        """ before load, you must initialize the session

        """

        #restore 模型的时候 不需要初始化
        # self.sess.run(self.init)
        self.saver.restore(self.sess, self.tf_path)
        return

    def evaluate(self, test_set):


        y_pred = self.predict(test_set)

        y_true = test_set.extract_label(test_set.tags, dense=False)

        # print(classification_report(y_true, y_pred, target_names=test_set.labels, digits=4))
        # print('precision: ', utility.precision_score(y_pred, y_true, len(test_set.tag_dict)))
        # print('recall   : ', utility.recall_score(y_pred, y_true, len(test_set.tag_dict)))

        # 打印结果日志
        logger = utility.get_logger(config['log']['train_mlp_log'])
        logger.info(str(classification_report(y_true, y_pred, target_names=test_set.labels, digits=4)))
        logger.info('precision: '+ str(utility.precision_score(y_pred, y_true, len(test_set.tag_dict))))
        logger.info('recall   : ' + str(utility.recall_score(y_pred, y_true, len(test_set.tag_dict))))
        return

    def predict(self, test_set):
        """ predict the label one_hot = False
            0 : 'O'
            1 : 'PER'
            2 : 'LOC'
            3 : 'ORG'
            4 : 'MISC'
        """
        return np.argmax(self.proba(test_set), 1)

    def proba(self, test_set):
        """ predict the unlabeled data
        Arg:
            X : to be predicted , (n_examples, n_features)

        Return:
            y_pred : predicted label, one hot (n_examples, n_classes)
        """

        sess = tf.Session(graph=self.graph)

        self.saver.restore(sess, self.tf_path)
        y_pred = sess.run(self.prediction, feed_dict={self.x: test_set.extract_features(test_set.sentences)})

        return y_pred

    def sentence_to_tags(self, string, dict_path='./tmp/word2vec'):
        """ convert prediction(label, int) to tag(string)

        Arg:
            X : sentence, string-format
        """
        sent = [string[i] for i in range(len(string))]

        fs = DataSet.w2v_DataSet(path=None, dictionary_path=dict_path, window_size=7, word2vec=True)

        feature = fs.extract_document_from_sentence(sent)

        y_pred = self.predict(feature)

        reverse_dict = {v: k for k, v in self.entity_dict.items()}

        return [reverse_dict[token] for token in y_pred]

    def predict_tag(self):

        return


class CRF(BaseEstimator):
    def __init__(self,

                 c1=0.5,
                 c2=5e-04,
                 max_iteration=100,
                 model_dir=''
                 ):
        self._nontrain = True
        self._trainer = pycrfsuite.Trainer(verbose=False)
        self._trainer.set_params(
            {
                'c1': 0.5,
                'c2': 5e-04,
                'max_iterations': 100,
                'feature.possible_transitions': True
            }
        )
        # self.tag_dict = tag_dict
        self._model_dir = model_dir

        return

    def tags_to_label(self, y_pred, dense=False):
        # first to array

        y_final = list(chain.from_iterable(y_pred))

        y_final = [self.tag_dict[token.split('-')[-1]] for token in y_final]

        if dense:
            return utility.dense_to_one_hot(y_final, len(self.tag_dict))
        return np.asarray(y_final)

    def fit(self, train_set):
        """train the model

        Arg:
            X : train data features
            y : train data labels
        """

        X = train_set.extract_features(train_set.sentences)
        Y = train_set.extract_labels(train_set.tags)
        for train_x, train_y in zip(X, Y):
            self._trainer.append(train_x, train_y)

        self._trainer.train(self._model_dir)
        self._nontrain = True

    def predict_sent(self, crf_data_set):
        """ predict the sequence tag

        Arg:
            X : to be predicted sequence , list of sequence

        Return:
            y_pred :
        """
        tagger = pycrfsuite.Tagger()
        tagger.open(self._model_dir)

        tmp = crf_data_set.extract_features(crf_data_set.sentences)

        y_pred = [tagger.tag(seq) for seq in crf_data_set.extract_features(crf_data_set.sentences)]
        return y_pred

    def predict(self, X):
        """

        :param X:
        :return:
        """
        y_pred = self.predict_sent(X)

        return y_pred
        # return self.tags_to_label(y_pred)

    def proba_sent(self, X):
        tagger = pycrfsuite.Tagger()
        tagger.open(self._model_dir)
        full_probability = []
        labels = list(tagger.labels())

        labels.sort()

        if 'O' in labels:
            k = labels.index('O')
            labels[0], labels[k] = labels[k], labels[0]

        tagger.set(X)
        for j in range(len(X)):
            # full_probability.append([tagger.marginal(tagger.labels()[k], j) for k in range(len(tagger.labels()))])
            full_probability.append([tagger.marginal(labels[index], j) for index in range(len(labels))])

        return np.reshape(full_probability, (len(full_probability), -1))

    def proba(self, test_set):
        """ probability distribution  (n_examples, n_classes)

        """
        # load the crf model

        X = test_set.extract_features(test_set.sentences)

        tagger = pycrfsuite.Tagger()
        tagger.open(self._model_dir)

        n_sentences = len(X)

        full_probability = []

        labels = list(tagger.labels())

        labels.sort()

        if 'O' in labels:
            k = labels.index('O')
            labels[0], labels[k] = labels[k], labels[0]



        for i in range(n_sentences):
            tagger.set(X[i])
            for j in range(len(X[i])):
                # full_probability.append([tagger.marginal(tagger.labels()[k], j) for k in range(len(tagger.labels()))])
                full_probability.append([tagger.marginal(labels[index], j) for index in range(len(labels))])

        # transform format
        full_probability = np.reshape(full_probability, (len(full_probability), -1))

        return full_probability

    def evaluate(self, crf_data_set):
        """ measure of the model including precision, recall and f1-score

        Arg:
            X : feature
            y : label

        """

        y_pred = self.predict_sent(crf_data_set)
        logger = utility.get_logger(config['log']['train_crf_log'])
        logger.debug(utility.bio_classification_report(crf_data_set.tags, y_pred))
        # print(utility.bio_classification_report(crf_data_set.tags, y_pred))

        return

    def evaluate_wn(self, crf_data_set):

        y_pred = self.predict_sent(crf_data_set)
        report = utility.bio_classification_report_wn(crf_data_set.tags, y_pred)
        logger = utility.get_logger(config['log']['train_crf_log'])
        for line in report:
            logger.info(line)
        return

est_dict = {'crf': CRF, 'mlp': MultiLayerPerceptron}


def load_conf(model_path):
    with open(model_path, 'rb') as r:
        conf = pickle.load(r)
    return conf

def get_estimators(est_name, train = True):
    config = get_config()
    mlp_conf = config['mlp']

    if est_name == 'mlp':
        conf = mlp_config()
        conf.batch_size = int(mlp_conf['batch_size'])
        conf.hidden_units = [int(token) for token in mlp_conf['hidden_units'].split(' ')]
        conf.learning_rate = float(mlp_conf['learning_rate'])
        conf.tf_path = mlp_conf['train_tf_path'] if train else mlp_config['test_tf_path']
        conf.train_epochs = int(mlp_conf['train_epochs'])

        return MultiLayerPerceptron(conf)
