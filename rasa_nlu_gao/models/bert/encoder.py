#!/usr/bin/python
# coding:utf-8
"""
replaces bert-as-service encoding as a function
"""

import os,time
import tensorflow as tf
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.run_config import RunConfig
from multiprocessing import cpu_count
from . import modeling, tokenization
from .extract_features import model_fn_builder, convert_lst_to_features, PoolingStrategy


class Encoder(object):
    def __init__(self, model_dir, max_seq_len=10):
        self.model_dir = model_dir
        self.max_seq_len = max_seq_len
        self.estimator, self.tokenizer = self.create_estimator_and_tokenizer()

    def create_estimator_and_tokenizer(self):
        config_fp = os.path.join(self.model_dir, 'bert_config.json')
        checkpoint_fp = os.path.join(self.model_dir, 'bert_model.ckpt')
        vocab_fp = os.path.join(self.model_dir, 'vocab.txt')

        tokenizer = tokenization.FullTokenizer(vocab_file=vocab_fp)

        model_fn = model_fn_builder(
            bert_config=modeling.BertConfig.from_json_file(config_fp),
            init_checkpoint=checkpoint_fp,
            pooling_strategy=PoolingStrategy.NONE,
            pooling_layer=[-2]
        )

        config = tf.ConfigProto(
            device_count={
                'CPU': cpu_count()
            },
            inter_op_parallelism_threads=0,
            intra_op_parallelism_threads=0,
            gpu_options={
                'allow_growth': True
            }
        )

        estimator = Estimator(model_fn, config=RunConfig(
            session_config=config), model_dir=None)

        return estimator, tokenizer

    def input_fn_builder(self, msg):
        def gen():
            for _ in range(1):
                tmp_f = list(convert_lst_to_features(msg, self.max_seq_len, self.tokenizer))
                yield {
                    'input_ids': [f.input_ids for f in tmp_f],
                    'input_mask': [f.input_mask for f in tmp_f],
                    'input_type_ids': [f.input_type_ids for f in tmp_f]
                }

        def input_fn():
            return (tf.data.Dataset.from_generator(
                gen,
                output_types={'input_ids': tf.int32,
                            'input_mask': tf.int32,
                            'input_type_ids': tf.int32,
                            },
                output_shapes={
                    'input_ids': (None, self.max_seq_len),
                    'input_mask': (None, self.max_seq_len),
                    'input_type_ids': (None, self.max_seq_len)}).prefetch(10))

        return input_fn

    def encode(self, questions):
        input_fn = self.input_fn_builder(questions)
        result = self.estimator.predict(input_fn)

        for rq in result:
            query_vec = rq['encodes']
        return query_vec
