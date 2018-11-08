import unittest
import json
import http.client


class TestRest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # cls.conn = http.client.HTTPConnection('192.168.3.110', 5001)
        cls.conn = http.client.HTTPConnection('localhost', 5001)

    def test_tag_pos(self):
        headers = {"Content-Type": "application/json"}

        params = {
            "task" : "pos",
            "data" : [
                {"content": "阳光很好。"},
                {"content": "新的一年即将到来。"}
            ]
        }

        self.__class__.conn.request('POST', '/tag', json.JSONEncoder().encode(params), headers)
        response = self.__class__.conn.getresponse()
        print(type(response))

        print(json.loads(response.read().decode()), 'utf-8')

    def test_tag_ner(self):
        headers = {"Content-Type": "application/json"}

        # params = {
        #     'task': 'ner',
        #     'data': [
        #         {'content': '阳光很好。'},
        #         {'content': '新的一年即将到来。'}
        #     ]
        # }

        params = {
            "task" : "ner",
            "data" : "2018年5月31日,公司与云南锡业集团有限责任公司(以下简称“锡业集团”或“甲方”)签署了《股权收购意向协议》"
        }

        self.__class__.conn.request('POST', '/ner_rel', json.JSONEncoder().encode(params), headers)
        response = self.__class__.conn.getresponse()
        print(type(response))

        print(json.loads(response.read().decode()), 'utf-8')


    def test_train(self):
        headers = {"Content-Type": "application/json"}

        params = {
            'flag' : 'train'
            # 'flag' : 'predict'
        }

        self.__class__.conn.request('POST', '/ner_train', json.JSONEncoder().encode(params), headers)
        response = self.__class__.conn.getresponse()
        print(type(response))

        print(json.loads(response.read().decode()))

if __name__ == '__main__':
    unittest.main()