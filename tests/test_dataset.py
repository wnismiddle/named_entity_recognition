import unittest

from app.main.DataSet import get_window_data_set, Seq_DataSet,crf_DataSet, get_crf_data_set


class TestDataSet(unittest.TestCase):

    def test_init(self):
        seq = Seq_DataSet(path=None, data=([],[],[]))
        seq = Seq_DataSet()
        print(seq)

    def test_load_from_config(self):
        ww = get_window_data_set(train = True)
        print(ww)

        self.assertTrue(ww.num_tokens > 0)
        self.assertTrue(ww.num_tokens > 0)


        ww2 = get_window_data_set('../data/esp/esp.train')
        # ww2 = get_window_data_set('/home/user/dsf/named_entity_recognition/data/esp/esp.train')
        print(ww2)

        ww3 = get_window_data_set(data = (['wo', 'bu', 'shi', 'zhong', 'wen'],[],[]))
        print(ww3)

    def test_load_default_ww(self):
        ww = get_window_data_set(train = True)
        print(ww)

        ww = get_window_data_set(train = False)
        print(ww)

    def test_word2vec(self):
        word = '中'
        ww = get_window_data_set()
        fea = ww.word2vec(word)
        print(fea)

        self.assertTrue(isinstance(fea, list))


        fea = ww.word2vec('nn')
        self.assertTrue(isinstance(fea, list))

        fea = ww.word2vec('space')
        self.assertTrue(isinstance(fea, list))

        self.assertTrue(sum(fea) == 0)

    def test_extract_sentence(self):

        sent = list('我喜欢中国，但我更爱北京')
        ww = get_window_data_set()
        ww.extract_feature_from_sentence(sent)

    def test_crf_dataset(self):
        crf_set = get_crf_data_set()
        print(crf_set)

    def test_crf_extract_sentence(self):
        crf_set = get_crf_data_set()
        print(crf_set.extract_features(crf_set.sentences[:4]))
        print(crf_set.extract_labels(crf_set.tags[:4]))

    def test_ww_extract_sentence(self):
        ww = get_window_data_set()
        tag = ww.extract_label_from_sentence(ww.tags[12])
        print('tag' , tag)