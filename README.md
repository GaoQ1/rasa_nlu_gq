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
这个项目的目的和初衷，是由于官方的rasa nlu里面提供的components和models有点out of time，并且精确率有时候不是很乐观。所以我自定义了几个基于tensorflow的能够兼容rasa框架的models，而为什么不直接提个pr到rasa nlu呢，因为要写太多test我懒癌犯了。所以在我自己的github上开源并发布到Pypi上，这样后续也能不断往里面填充和优化模型，方便别人也方便自己。

## New models
这里新增的models主要是做实体识别的模型，主要有两个一个是bilstm+crf，一个是idcnn+crf膨胀卷积模型

## Quick Install
```
pip install rasa_nlu_gao
```

## 🤖 Running of the bot
To train the NLU model:
```
python -m rasa_nlu_gao.train -c sample_configs/config_embedding_bilstm.yml --data data/examples/rasa/rasa_dataset_training.json --path models
```

To run the NLU model:
```
python -m rasa_nlu_gao.server -c sample_configs/config_embedding_bilstm.yml --path models
```
