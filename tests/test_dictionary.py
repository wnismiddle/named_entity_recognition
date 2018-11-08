import unittest
import numpy as np

from app.utils.utility import get_config
import app.utils.Dictionary as Dic

class TestDictionary(unittest.TestCase):

    def setUp(self):
        self.dic = Dic.Dictionary()
        return

    def test_void_ini(self):
        dic = Dic.Dictionary()
        self.assertTrue(dic, dict())

    def test_check(self):

        dic = {'a':[1,2,4,5], 'b' : [2.3, 3.2, 3, 1]}
        self.assertTrue(self.dic.check(dic))

    def test_vec_size(self):
        dic = {'a': [1, 2, 4, 5], 'b': [2.3, 3.2, 3, 1]}
        self.assertTrue(self.dic.get_size(dic), 4)

    def test_load(self):
        config = get_config()

        zh_config = config['zh_dictionary']
        self.dic.load_dictionary(zh_config['path'], zh_config['flag'] == 'True')
        self.assertTrue(self.dic.vector_size == 100)

        eng_config = config['eng_dictionary']
        self.dic.load_dictionary(eng_config['path'], eng_config['flag'] == 'True')
        self.assertTrue(self.dic.vector_size == 300)


    def test_zh_get_vec(self):
        config = get_config()
        zh_config = config['zh_dictionary']
        self.dic.load_dictionary(zh_config['path'], zh_config['flag'] == 'True')


        default = [0] * 10
        self.assertRaises(ValueError, self.dic.get_vector, '中', default_vec=default )

        default = [0] * 100
        vec = self.dic.get_vector('中', default_vec=default)
        self.assertTrue(len(vec), 300)

        vec = self.dic.get_vector('nimh', default_vec=default)
        self.assertTrue(vec == default)


        return


    def test_keys(self):
        keys = list(self.dic.keys())
        self.assertTrue(len(keys) == 0)

        config = get_config()
        zh_config = config['zh_dictionary']
        self.dic.load_dictionary(zh_config['path'], zh_config['flag'] == 'True')

        keys = list(self.dic.keys())
        self.assertTrue(len(keys) > 0)
