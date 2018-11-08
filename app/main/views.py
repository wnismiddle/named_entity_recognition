# coding:utf-8
from flask import render_template, session, redirect, url_for, current_app, request, jsonify

from . import main

# from ..ner import seq_pos, seq_ner, seq_ner_rel, train
from ..ner import *
import traceback
import json


def get_request():
    return

@main.route("/")
def index():
    return jsonify(
            {
                'ret_code': 0
            }
        )

@main.route('/test/<username>')
def hello(username):
    try:
        request_data = request.get_data().decode('utf-8')
        json_request_data = json.loads(request_data, 'utf-8')
        flag = json_request_data['flag']
    except Exception as e:
        print(e)
    return 'service test successful, ' + username

# 词性标注任务 RESTful API
@main.route('/pos', methods=['POST'])
def pos():
    try:
        request_data = request.get_data().decode('utf-8')
        json_request_data = json.loads(request_data, encoding='utf-8')

        data = json_request_data['data']
        tag = seq_pos(data)
        # print(tag)
        return jsonify({
            'ret_code': 0,
            'task': 'pos',
            'result': [{'id': i, 'tagger': tag[i]} for i in range(len(data))]
        })
    except Exception as e:
        return jsonify(
            {
                "ret_code": -1,
                "task": "pos",
                "err_info": traceback.format_exc()
            }
        )
        print(traceback.format_exc())

# 命名实体识别任务 RESTful API
@main.route('/ner', methods=['POST'])
def ner():
    try:
        request_data = request.get_data().decode('utf-8')
        json_request_data = json.loads(request_data, encoding='utf-8')
        # print(json_request_data)

        data = json_request_data['data']
        model = json_request_data['model']

        data = [tmp for tmp in data.split('。') if len(tmp) > 0]
        # print(len(data))
        sentences, tags, entity_set = seq_ner(data, model)

        return jsonify({
            'ret_code': 0,
            'task': 'ner',
            'sentences': sentences,
            'tags': tags,
            # 'tags': [[token.split('-')[-1] for token in tag] for tag in tags],
            'entity_set': entity_set
        })

    except Exception as e:
        return jsonify(
            {
                "ret_code": -1,
                "task": "ner",
                "err_info": traceback.format_exc()
            }
        )
        print(traceback.format_exc())

# 该方法已删
# @main.route('/tag', methods=['POST'])
def tag():
    try:
        request_data = request.get_data().decode('utf-8')
        json_request_data = json.loads(request_data, encoding='utf-8')
        # print(json_request_data)
        task = json_request_data['task']

        if task == 'pos':
            data = [dic['content'] for dic in json_request_data['data']]
            tag = seq_pos(data)
            print(tag)
            return jsonify({
                'ret_code': 0,
                'task': 'pos',
                'result': [{'id': i, 'tagger': tag[i]} for i in range(len(data))]
            })

        if task == 'ner':
            data = json_request_data['data']
            data = [tmp for tmp in data.split('。') if len(tmp) > 0]
            print(len(data))
            sentences, tags, entity_set = seq_ner(data)

            return jsonify({
                'ret_code': 0,
                'task': 'ner',
                'sentences': sentences,
                'tags': [[token.split('-')[-1] for token in tag] for tag in tags],
                'entity_set': entity_set
            })

    except Exception as e:
        return jsonify(
            {
                "ret_code": -1,
                "task": "ner",

            }
        )
        print(traceback.format_exc())

# 实体识别并标注 RESTful API
@main.route('/ner_rel', methods=['POST'])
def ner_rel():
    try:
        request_data = request.get_data().decode('utf-8')
        json_request_data = json.loads(request_data, encoding='utf-8')
        data = json_request_data['data']
        model = json_request_data['model']

        data = [tmp for tmp in data.split('。') if len(tmp) > 0]     # 将文本按句号分割

        out_sentence = seq_ner_rel(data, model)    # 标注每个句子

        return jsonify({
            "ret_code": 0,
            "task": "ner_rel",
            "sentences": [sent for sent in out_sentence],
        })
    except Exception:
        return jsonify({
            "ret_code": -1,
            "err_info": traceback.format_exc()
        })

# 训练模型 RESTful API
@main.route('/ner_train', methods=['POST'])
def ner_train():
    try:
        request_data = request.get_data().decode('utf-8')
        json_request_data = json.loads(request_data, encoding='utf-8')
        model = json_request_data['model']

        if model == "crf":
            # train(path="../../data/zh/zh.train")
            train()  # 默认使用条件随机场方式训练
        elif model == "mlp":
            train(crf_flag=False, mlp_flag=True)    # 使用神经网络训练

        return jsonify({
            "ret_code": 0,
            "task": "ner_train",
        })
    except Exception:
        return jsonify({
            "ret_code": -1,
            "task": "ner_train",
            "err_info": traceback.format_exc()
        })
    return