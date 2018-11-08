import unittest

from app.main.estimators import MultiLayerPerceptron, get_estimators, load_conf
from app.main.DataSet import get_window_data_set
from app.utils.utility import get_config, get_model_path
from app.utils.utility import get_default_mlp_config

class TestEstimators(unittest.TestCase):


    def test_init(self):
        config = get_config()
        est = MultiLayerPerceptron()

    def test_load_from_config(self):
        mlp = get_estimators('mlp')
        print(mlp)
        self.assertTrue(isinstance(mlp, MultiLayerPerceptron))


    def test_save(self):
        path = get_model_path('mlp')
        default_conf = get_default_mlp_config(train=True)

        mlp = MultiLayerPerceptron(default_conf)


    def test_load(self):
        pass

    def test_predict(self):
        model_path = get_model_path('mlp')
        default_conf = load_conf(model_path= model_path)
        mlp = MultiLayerPerceptron(default_conf)
        mlp.load()

        ww = get_window_data_set()
        ww.set_data(data=([list('我来到中国'), list('今天我和张建国去东南大学做报告')], [], []))

        print(mlp.predict(ww))