import unittest

import app.utils.utility as uti

from . import get_window_data_set

class TestUtility(unittest.TestCase):

    def test_get_model_path(self):
        dir = uti.get_model_path('mlp')
        print(dir)
        print(uti.get_model_path('crf'))

    def test_get_window_set(self):
        train_set = get_window_data_set()
        print(train_set)


