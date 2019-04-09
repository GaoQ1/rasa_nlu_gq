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

## Intent of this project
这个项目的目的和初衷，是由于官方的rasa nlu里面提供的components和models并不能满足实际需求。所以我自定义了一些components，并发布到Pypi上。可以通过`pip install rasa-nlu-gao`下载。后续会不断往里面填充和优化组件，也欢迎大家贡献。

## New features
目前新增的特性如下（请下载最新的rasa-nlu-gao版本）：
  - 新增了实体识别的模型，一个是bilstm+crf，一个是idcnn+crf膨胀卷积模型，对应的yml文件配置如下：
  ```
    language: "zh"

    pipeline:
    - name: "tokenizer_jieba"
    - name: "intent_featurizer_count_vectors"
      token_pattern: "(?u)\b\w+\b"
    - name: "intent_classifier_tensorflow_embedding"
    - name: "ner_bilstm_crf"
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
    - name: "tokenizer_jieba"
    - name: "ner_crf"
    - name: "jieba_pseg_extractor"
      part_of_speech: ["nr", "ns", "nt"]
    - name: "intent_featurizer_count_vectors"
      OOV_token: oov
      token_pattern: "(?u)\b\w+\b"
    - name: "intent_classifier_tensorflow_embedding"
  ```
  - 新增了根据实体反向修改意图，对应的文件配置如下：
  ```
    language: "zh"

    pipeline:
    - name: "tokenizer_jieba"
    - name: "ner_crf"
    - name: "jieba_pseg_extractor"
    - name: "intent_featurizer_count_vectors"
      OOV_token: oov
      token_pattern: '(?u)\b\w+\b'
    - name: "intent_classifier_tensorflow_embedding"
    - name: "entity_edit_intent"
      entity: ["nr"]
      intent: ["enter_data"]
      min_confidence: 0
  ```
  - 新增了word2vec提取词向量特征，对应的配置文件如下：
  ```
    language: "zh"

    pipeline:
    - name: "tokenizer_jieba"
    - name: "intent_featurizer_wordvector"
      vector: "data/vectors.txt"
    - name: "intent_classifier_tensorflow_embedding"
    - name: "ner_crf"
    - name: "jieba_pseg_extractor"
  ```
  - 新增了bert模型提取词向量特征，对应的配置文件如下：
  ```
    language: "zh"

    pipeline:
    - name: "tokenizer_jieba"
    - name: "bert_vectors_featurizer"
      ip: '172.16.10.46'
      port: 5555
      port_out: 5556
      show_server_config: True
      timeout: 10000
    - name: "intent_classifier_tensorflow_embedding"
    - name: "ner_crf"
    - name: "jieba_pseg_extractor"
  ```
  - 新增了对CPU和GPU的利用率的配置，主要是`intent_classifier_tensorflow_embedding`和`ner_bilstm_crf`这两个使用到tensorflow的组件，配置如下（当然config_proto可以不配置，默认值会将资源全部利用）：
  ```
    language: "zh"

    pipeline:
    - name: "tokenizer_jieba"
    - name: "intent_featurizer_count_vectors"
      token_pattern: '(?u)\b\w+\b'
    - name: "intent_classifier_tensorflow_embedding"
      config_proto: {
        "device_count": 4,
        "inter_op_parallelism_threads": 0,
        "intra_op_parallelism_threads": 0,
        "allow_growth": True
      }
    - name: "ner_bilstm_crf"
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
    - name: "tokenizer_jieba"
    - name: "bert_vectors_featurizer"
      ip: '172.16.10.46'
      port: 5555
      port_out: 5556
      show_server_config: True
      timeout: 10000
    - name: "intent_classifier_tensorflow_embedding_bert"
    - name: "ner_crf"
    - name: "jieba_pseg_extractor"
  ```
  
   - 在基础词向量使用bert的情况下，后端的分类器使用tensorflow高级api完成，tf.estimator,tf.data,tf.example,tf.saved_model
   `intent_estimator_classifier_tensorflow_embedding_bert`分类器，对应的配置文件如下：
  ```
  language: "zh"

  pipeline:
  - name: "tokenizer_jieba"
  - name: "bert_vectors_featurizer"
    ip: '127.0.0.1'
    port: 5555
    port_out: 5556
    show_server_config: True
    timeout: 10000
  - name: "intent_estimator_classifier_tensorflow_embedding_bert"
  - name: "nlp_spacy"
  - name: "ner_crf"
  ```

## Quick Install
```
pip install rasa-nlu-gao
```

## 🤖 Running of the bot
To train the NLU model:
```
python3 -m rasa_nlu_gao.train -c sample_configs/config_embedding_bilstm.yml --data data/examples/rasa/rasa_dataset_training.json --path models
```

To run the NLU model:
```
python3 -m rasa_nlu_gao.server -c sample_configs/config_embedding_bilstm.yml -P 8000 --path models
```

## Some Examples
具体的例子请看[rasa_chatbot_cn](https://github.com/GaoQ1/rasa_chatbot_cn)