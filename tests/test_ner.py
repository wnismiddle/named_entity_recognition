import unittest

from app import ner

class TestNer(unittest.TestCase):

    def test_train(self):
        ner.train()     # 默认crf训练
        # ner.train(path=None, crf_flag=False, mlp_flag=True)     # mlp训练