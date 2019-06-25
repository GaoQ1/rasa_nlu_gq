# Rasa NLU GQ
Rasa NLU (Natural Language Understanding) 是一个自然语义理解的工具，举个官网的例子如下：

> *"I'm looking for a Mexican restaurant in the center of town"*

And returning structured data like:

```
  intent: search_restaurant
  entities: 
    - cuisine : Mexican
    - location : center
```

## Introduction
原来的项目在分支0.2.7上，可自由切换。这个版本的修改是基于最新版本的rasa，将原来rasa_nlu_gao里面的component修改了下，并没有做新增。并且之前做法有些累赘，并不需要在rasa源码中修改。可以直接将原来的component当做addon加载，继承最新版本的rasa，可实时更新。

## New features
目前新增的特性如下（请下载最新的rasa-nlu-gao版本）(edit at 2019.06.24)：
  - 新增了实体识别的模型，一个是bilstm+crf，一个是idcnn+crf膨胀卷积模型，对应的yml文件配置如下：
  ```
    language: "zh"

    pipeline:
    - name: "JiebaTokenizer"
    - name: "CountVectorsFeaturizer"
      token_pattern: "(?u)\b\w+\b"
    - name: "EmbeddingIntentClassifier"
    - name: "rasa_nlu_gao.extractors.bilstm_crf_entity_extractor.BilstmCRFEntityExtractor"
      lr: 0.001
      char_dim: 100
      lstm_dim: 100
      batches_per_epoch: 10
      seg_dim: 20
      num_segs: 4
      batch_size: 200
      tag_schema: "iobes"
      model_type: "bilstm" # 模型支持两种idcnn膨胀卷积模型或bilstm双向lstm模型
      clip: 5
      optimizer: "adam"
      dropout_keep: 0.5
      steps_check: 100
  ```
  - 新增了jieba词性标注的模块，可以方便识别名字，地名，机构名等等jieba能够支持的词性，对应的yml文件配置如下：
  ```
    language: "zh"

    pipeline:
    - name: "JiebaTokenizer"
    - name: "CRFEntityExtractor"
    - name: "rasa_nlu_gao.extractors.jieba_pseg_extractor.JiebaPsegExtractor"
      part_of_speech: ["nr", "ns", "nt"]
    - name: "CountVectorsFeaturizer"
      OOV_token: oov
      token_pattern: "(?u)\b\w+\b"
    - name: "EmbeddingIntentClassifier"
  ```
  - 新增了根据实体反向修改意图，对应的文件配置如下：
  ```
    language: "zh"

    pipeline:
    - name: "JiebaTokenizer"
    - name: "CRFEntityExtractor"
    - name: "JiebaPsegExtractor"
    - name: "CountVectorsFeaturizer"
      OOV_token: oov
      token_pattern: '(?u)\b\w+\b'
    - name: "EmbeddingIntentClassifier"
    - name: "rasa_nlu_gao.classifiers.entity_edit_intent.EntityEditIntent"
      entity: ["nr"]
      intent: ["enter_data"]
      min_confidence: 0
  ```
  - 新增了bert模型提取词向量特征，对应的配置文件如下：
  ```
    language: "zh"

    pipeline:
    - name: "JiebaTokenizer"
    - name: "rasa_nlu_gao.featurizers.bert_vectors_featurizer.BertVectorsFeaturizer"
      ip: '127.0.0.1'
      port: 5555
      port_out: 5556
      show_server_config: True
      timeout: 10000
    - name: "EmbeddingIntentClassifier"
    - name: "CRFEntityExtractor"
  ```
  - 新增了对CPU和GPU的利用率的配置，主要是`EmbeddingIntentClassifier`和`ner_bilstm_crf`这两个使用到tensorflow的组件，配置如下（当然config_proto可以不配置，默认值会将资源全部利用）：
  ```
    language: "zh"

    pipeline:
    - name: "JiebaTokenizer"
    - name: "CountVectorsFeaturizer"
      token_pattern: '(?u)\b\w+\b'
    - name: "EmbeddingIntentClassifier"
      config_proto: {
        "device_count": 4,
        "inter_op_parallelism_threads": 0,
        "intra_op_parallelism_threads": 0,
        "allow_growth": True
      }
    - name: "rasa_nlu_gao.extractors.bilstm_crf_entity_extractor.BilstmCRFEntityExtractor"
      config_proto: {
        "device_count": 4,
        "inter_op_parallelism_threads": 0,
        "intra_op_parallelism_threads": 0,
        "allow_growth": True
      }
  ```
  - 新增了`embedding_bert_intent_classifier`分类器，对应的配置文件如下：
  ```
    language: "zh"

    pipeline:
    - name: "JiebaTokenizer"
    - name: "rasa_nlu_gao.featurizers.bert_vectors_featurizer.BertVectorsFeaturizer"
      ip: '127.0.0.1'
      port: 5555
      port_out: 5556
      show_server_config: True
      timeout: 10000
    - name: "rasa_nlu_gao.classifiers.embedding_bert_intent_classifier.EmbeddingBertIntentClassifier"
    - name: "CRFEntityExtractor"
  ```
  
   - 在基础词向量使用bert的情况下，后端的分类器使用tensorflow高级api完成，tf.estimator,tf.data,tf.example,tf.saved_model
   `intent_estimator_classifier_tensorflow_embedding_bert`分类器，对应的配置文件如下：
  ```
  language: "zh"

  pipeline:
  - name: "JiebaTokenizer"
  - name: "rasa_nlu_gao.featurizers.bert_vectors_featurizer.BertVectorsFeaturizer"
    ip: '127.0.0.1'
    port: 5555
    port_out: 5556
    show_server_config: True
    timeout: 10000
  - name: "rasa_nlu_gao.classifiers.embedding_bert_intent_estimator_classifier.EmbeddingBertIntentEstimatorClassifier"
  - name: "SpacyNLP"
  - name: "CRFEntityExtractor"
  ```

## Quick Install
```
pip install rasa-nlu-gao
```

## Some Examples
具体的例子请看[rasa_chatbot_cn](https://github.com/GaoQ1/rasa_chatbot_cn)

## TODO
 - Add more awesome and userful components, like intent classify and slot extract, or combime this two tickets.
 - Keep updating need your contribute