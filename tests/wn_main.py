
from app import ner
from app.utils.utility import get_config, get_model_path
import os
from app.ner import *


def train():
    ner.train()     # 默认crf训练
    # ner.train(path=None, crf_flag=False, mlp_flag=True)     # mlp训练

def clean_logs():
    config = get_config()
    train_crf_log_path = config['log']['train_crf_log']
    test_crf_log_path = config['log']['test_crf_log']
    train_mlp_log_path = config['log']['train_mlp_log']
    test_mlp_log_path = config['log']['test_mlp_log']
    if os.path.exists(train_crf_log_path):
        os.remove(train_crf_log_path)
    if os.path.exists(test_crf_log_path):
        os.remove(test_crf_log_path)
    if os.path.exists(train_mlp_log_path):
        os.remove(train_mlp_log_path)
    if os.path.exists(test_mlp_log_path):
        os.remove(test_mlp_log_path)

def clean_models(model_name='crf'):
    config = get_config()
    model = config['model']
    name = 'train_' + model_name
    if os.path.exists(model[name]):
        os.remove(model[name])
    return

def test_ner_rel():
    data = u"江门甘蔗化工厂(集团)股份有限公司(以下简称“公司”)正在筹划重大事项,涉及收购国内两家军工行业标的公司的控股权,包括用于能源、电力、大型装备、军工等领域的中高端金属制品制造企业及为军队武器装备提供关键零部件的企业。目前相关方正在积极磋商交易的具体方案,该事项可能涉及重大资产重组。公司将按照有关规定,组织开展各项工作,同时根据事项的进展情况及时履行相应程序,并进行信息披露。公司筹划的重大事项尚存较大不确定性,敬请广大投资者注意投资风险。2018年6月5日,公司与主要交易对方签订了《关于购买资产之意向书》,1、标的公司。本次交易的标的公司为四川升华电源科技有限公司(以下简称“升华电源”),升华电源主要从事军用开关电源、模块电源的研发、生产和销售。2、标的资产。本次交易的标的资产为升华电源的控股权。3、交易对方。本次交易的交易对方为冯骏、彭玫等升华电源现有股东。交易对方为独立第三方,本次交易不构成关联交易。4、交易方式。本次交易的交易方式为由上市公司支付现金的方式购买标的资产。同时,交易对方须将所获对价的一定金额用于增持上市公司股份,建立双方合作的长效机制。"
    data = [tmp for tmp in data.split('。') if len(tmp) > 0]  # 将文本按句号分割
    out_sentence = seq_ner_rel(data)  # 标注每个句子

def test_ner():
    data = u"江门甘蔗化工厂(集团)股份有限公司(以下简称“公司”)正在筹划重大事项,涉及收购国内两家军工行业标的公司的控股权,包括用于能源、电力、大型装备、军工等领域的中高端金属制品制造企业及为军队武器装备提供关键零部件的企业。目前相关方正在积极磋商交易的具体方案,该事项可能涉及重大资产重组。公司将按照有关规定,组织开展各项工作,同时根据事项的进展情况及时履行相应程序,并进行信息披露。公司筹划的重大事项尚存较大不确定性,敬请广大投资者注意投资风险。2018年6月5日,公司与主要交易对方签订了《关于购买资产之意向书》,1、标的公司。本次交易的标的公司为四川升华电源科技有限公司(以下简称“升华电源”),升华电源主要从事军用开关电源、模块电源的研发、生产和销售。2、标的资产。本次交易的标的资产为升华电源的控股权。3、交易对方。本次交易的交易对方为冯骏、彭玫等升华电源现有股东。交易对方为独立第三方,本次交易不构成关联交易。4、交易方式。本次交易的交易方式为由上市公司支付现金的方式购买标的资产。同时,交易对方须将所获对价的一定金额用于增持上市公司股份,建立双方合作的长效机制。"
    data = [tmp for tmp in data.split('。') if len(tmp) > 0]
    print(len(data))
    sentences, tags, entity_set = seq_ner(data, 'crf')


if __name__ == '__main__':
    # clean_logs()
    # clean_models(model_name='crf')
    # clean_models(model_name='mlp')
    # train()
    # test_ner_rel()
    test_ner()