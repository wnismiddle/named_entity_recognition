[TOC]

# REST API

URL 结构 http://localhost:5000/method

### 通用输入参数

```python
{
  	"task" : "" # 需要处理的任务
}
```

### 通用返回结果

```python
{
  	"ret_code" : 0 # 返回码，0为ok，其余表示错误
  	"err_msg" : "ok" # 具体错误信息
  	"task" : "/:ner" # 输入的任务，':'前是url，后面是task
    "result" : "" # 返回的结果，一般是list格式
}
```



## 数据格式

本系统主要用于处理序列标注问题，训练模型时需要提供序列数据和对应的标签，预测时提供纯文本。目前在中文/英文命名实体上取得较好的效果。

- NER-format

```python
# 每一行都是一个token,一句话之间用一个空行隔开。
# 第一列对应token(中文单个字，英文单个词(最小单词))		**必须有**
# 最后一列是token对应的tag，基础的是PER-LOC-ORG三种	**必须有**
# 中间列为任意的特征，常用的有POS, CHUNK等				**可选**
EU 			NNP 	I-NP 	I-ORG
rejects 	VBZ 	I-VP 	O
German 		JJ 		I-NP 	I-MISC
call 		NN 		I-NP 	O
to 			TO 		I-VP 	O
boycott 	VB 		I-VP 	O
British 	JJ 		I-NP 	I-MISC
lamb 		NN 		I-NP 	O
. 			. 		O 	 	O
```

- raw data + entity

  如果用户只能提供原始文件，需要按照**一行一句话**的格式提供，同时提供实体字典。示例如下：

```python
## raw data
藏书本来就是所有传统收藏门类中的第一大户，只是我们结束温饱的时间太短而已。
以家乡的历史文献、特定历史时期书刊、某一名家或名著的多种出版物为专题，注意精品、非卖品、纪念品，集成系列，那收藏的过程就已经够您玩味无穷了。
精品、专题、系列、稀见程度才是质量的核心。
...
```

```python
## entity dict
车泽武 PER
英国 LOC
刘杰 PER
青岛电冰箱总厂 ORG
爱知县丰田市 LOC
杰杰 PER
广州美晨股份有限公司 ORG
袁微 PER
西单商场 LOC
```

## 序列标注

### 词性标注

- **描述**：使用**jieba**提供的分词器，对序列中的每个token的词性进行标注。

- **URL**：`/tag` 

- **示例**：

  ```python
  # user输入list of sentence，以及需要的任务
  {
    	"task" : "pos", # 需要处理的序列标注任务，包括pos(词性标注)和ner(命名实体识别)
    	"data_format" : "file", # 数据
    	"data" : [
        "content" : "this is a sentence",
        "content" : "this is a sentence"
    	]
  ```

### NER

- **描述**：使用本系统的NER方法，对序列中的token进行命名实体识别

- **URL**：`/tag` 

- **示例**：

  ```json
  # user输入list of sentence，以及需要的任务
  {
    	"task" : "ner",
    	"data_format" : "file",
    	"language" : 'zh' # 模型的语言，默认为中文，可选英文
    	"data" : [
        "content" : "this is a sentence",
        "content" : "this is a sentence"
    	]
  }
  ```

## 模型训练

### NER模型训练

- **描述**：返回当前系统的配置信息，以及提供修改配置信息的接口

- **URL**：`/train` 

- **示例**：

  ```python
  {
    	"task" : "ner",
      "data_format" : 0/1,
      "entity_dict_path" : "" # /相对地址
  	"data_path" : "" # /相对地址
      "estimators" : "mlp crf" # 目前提供mlp和crf两种模型
      "boosting" : "adaboosting"# 目前只提供boosting 空则不进行boosting
      
  }
  ```

  ​

## 配置

- **描述**：返回当前系统的配置信息，以及提供修改配置信息的接口

- **URL**：`/config` 

- **示例**：

  ```python
  # 输入:获得当前配置信息
  {
    	"task" : "search",
      "config list" : "" # 空则返回所有配置信息，不空则显示请求的信息，多个config用空格隔开
  }
  # 返回:返回相应的配置结果
  {
    	"result" : [
        "learning rate" : 0.05， # 所有的配置信息参考设计手册
    	]
  }
  ```

  ```python
  # 输入:修改指定配置信息
  {
    	"task" : "update",
      "config list" : "learning_rate,train_epochs"
      "data" : [
        	"learning rate" : 0.5,
          "train epochs" : 10
      ]
  }
  # 返回:通用返回。
  ```

  ​

