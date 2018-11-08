import unittest

from app.utils.utility import get_config


class TestInit(unittest.TestCase):

    def test_all(self):
        config = get_config()
        self.assertTrue(isinstance(config, dict))

        self.assertTrue('zh_dictionary' in config.keys())
        zh_dictionary = config['zh_dictionary']
        self.assertTrue(isinstance(zh_dictionary, dict))
        self.assertTrue(zh_dictionary['path'], '/Users/duanshangfu/PycharmProjects/named_entity_recognition/data/tmp/word2vec/zh')
        self.assertTrue(zh_dictionary['flag'], 'True')

        self.assertTrue('eng_dictionary' in config.keys())
        eng_dictionary = config['eng_dictionary']
        self.assertTrue(eng_dictionary['path'], '/Users/duanshangfu/PycharmProjects/named_entity_recognition/data/tmp/word2vec/eng')

        self.assertTrue('dataset' in config.keys())
        dataset = config['dataset']
        self.assertEqual(dataset['zh_train'], '/Users/duanshangfu/PycharmProjects/named_entity_recognition/data/zh/zh.train')
