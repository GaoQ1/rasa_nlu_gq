from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import re
import io

import typing
import numpy as np

try:
    import cPickle as pickle
except ImportError:
    import pickle

from builtins import str
from typing import Any, Dict, List, Optional, Text, Tuple

from rasa_nlu_gao.extractors import EntityExtractor
from rasa_nlu_gao.model import Metadata
from rasa_nlu_gao.training_data import Message

from rasa_nlu_gao.utils.bilstm_utils import char_mapping, tag_mapping, prepare_dataset, BatchManager, iob_iobes, iob2, save_model, create_model, input_from_line

from rasa_nlu_gao.models.model import Model

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import numpy as np
    import tensorflow as tf
    import tensorflow.contrib

try:
    import tensorflow as tf
except ImportError:
    tf = None

class BilstmCRFEntityExtractor(EntityExtractor):
    name = "ner_bilstm_crf"

    provides = ["entities"]

    requires = ["tokens"]

    defaults = {
        "lr": 0.001,
        "char_dim": 100,
        "lstm_dim": 100,
        "batches_per_epoch": 10,
        "seg_dim": 20,
        "num_segs": 4,
        "batch_size": 20,
        "zeros": True,
        "tag_schema": "iobes",
        "lower": False,
        "model_type": "idcnn",
        "clip": 5,
        "optimizer": "adam",
        "dropout_keep": 0.5,
        "steps_check": 100
    }
    

    def __init__(self,
                component_config=None,
                ent_tagger=None,
                session=None,
                char_to_id=None,
                id_to_tag=None):
        super(BilstmCRFEntityExtractor, self).__init__(component_config)

        self.component_config = component_config
        self.ent_tagger = ent_tagger # 指的是训练好的model
        self.session = session
        self.char_to_id = char_to_id
        self.id_to_tag = id_to_tag

    def train(self, training_data, config, **kwargs):
        self.component_config = config.for_component(self.name, self.defaults)

        if training_data.entity_examples:
            filtered_entity_examples = self.filter_trainable_entities(training_data.training_examples)

            train_sentences = self._create_dataset(filtered_entity_examples)

            # 检测并维护数据集的tag标记
            self.update_tag_scheme(
                train_sentences, self.component_config["tag_schema"])

            _c, char_to_id, id_to_char = char_mapping(
                train_sentences, self.component_config["lower"])

            tag_to_id, id_to_tag = tag_mapping(train_sentences)
            
            self.char_to_id = char_to_id
            self.id_to_tag = id_to_tag

            self.component_config["num_chars"] = len(char_to_id)
            self.component_config["num_tags"] = len(tag_to_id)
            
            train_data = prepare_dataset(
                train_sentences, char_to_id, tag_to_id, self.component_config["lower"]
            )

            # 获取可供模型训练的单个批次数据
            train_manager = BatchManager(
                train_data, self.component_config["batch_size"])

            self._train_model(train_manager)


    def _create_dataset(self, examples):
        dataset = []
        for example in examples:
            entity_offsets = self._convert_example(example)
            dataset.append(self._predata(
                example.text, entity_offsets, self.component_config["zeros"]))
        return dataset


    @staticmethod
    def _convert_example(example):
        def convert_entity(entity):
            return entity["start"], entity["end"], entity["entity"]

        return [convert_entity(ent) for ent in example.get("entities", [])]

    @staticmethod
    def _predata(text, entity_offsets, zeros):
        value = 'O'
        bilou = [value for _ in text]
        # zero_digits函数的用途是将所有数字转化为0

        def zero_digits(s):
            return re.sub('\d', '0', s)

        text = zero_digits(text.rstrip()) if zeros else text.rstrip()

        cooked_data = []

        for (start, end, entity) in entity_offsets:
            if start is not None and end is not None:
                bilou[start] = 'B-' + entity
                for i in range(start+1, end):
                    bilou[i] = 'I-' + entity

        for index, achar in enumerate(text):
            if achar.strip():
                temp = []
                temp.append(achar)
                temp.append(bilou[index])

                cooked_data.append(temp)
            else:
                continue

        return cooked_data

    def update_tag_scheme(self, sentences, tag_scheme):
        for i, s in enumerate(sentences):
            tags = [w[1] for w in s]
            # Check that tags are given in the IOB format
            if not iob2(tags):
                s_str = '\n'.join(' '.join(w) for w in s)
                raise Exception('Sentences should be given in IOB format! ' +
                                'Please check sentence %i:\n%s' % (i, s_str))
            if tag_scheme == 'iob':
                # If format was IOB1, we convert to IOB2
                for word, new_tag in zip(s, tags):
                    word[1] = new_tag
            elif tag_scheme == 'iobes':
                new_tags = iob_iobes(tags)
                for word, new_tag in zip(s, new_tags):
                    word[1] = new_tag
            else:
                raise Exception('Unknown tagging scheme!')

    def _train_model(self, train_manager):
        tf_config = tf.ConfigProto()
        # tf_config.gpu_options.allow_growth = True

        # 训练集全量跑一次需要迭代的次数
        steps_per_epoch = train_manager.len_data

        sess = tf.Session(config=tf_config)

        self.session = sess

        # 此处模型创建为项目最核心代码
        model = create_model(sess, Model, self.component_config, logger)
        self.model = model

        logger.warning("start training")
        loss_slot = []

        for _ in range(self.component_config["batches_per_epoch"]):
            for batch in train_manager.iter_batch(shuffle=True):
                step, batch_loss_slot = model.run_step(
                    sess, True, batch)
                loss_slot.append(batch_loss_slot)

                if step % self.component_config["steps_check"] == 0:
                    iteration = step // steps_per_epoch + 1

                    logger.warning("iteration:{} step:{}/{}, "
                                "NER loss:{:>9.6f}".format(
                                    iteration, step % steps_per_epoch, steps_per_epoch, np.mean(loss_slot)))
                    loss_slot = []


    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        extracted = self.add_extractor_name(self.extract_entities(message))
        message.set("entities", message.get("entities", []) + extracted, add_to_output=True)

    def extract_entities(self, message):
        # type: (Message) -> List[Dict[Text, Any]]
        """Take a sentence and return entities in json format"""
        if self.ent_tagger is not None:
            result =  self.ent_tagger.evaluate_line(
                self.session, input_from_line(message.text, self.char_to_id), self.id_to_tag)
            return result.get("entities", [])
        else:
            return []


    @classmethod
    def load(cls,
             model_dir=None,  # type: Text
             model_metadata=None,  # type: Metadata
             cached_component=None,  # type: Optional[CRFEntityExtractor]
             **kwargs  # type: **Any
             ):
        meta = model_metadata.for_component(cls.name)

        tf_config = tf.ConfigProto()
        # tf_config.gpu_options.allow_growth = True

        sess = tf.Session(config=tf_config)

        model = Model(meta)
        if model_dir and meta.get("classifier_file"):
            file_name = meta.get("classifier_file")
            checkpoint = os.path.join(model_dir, file_name)
            model.saver.restore(sess, checkpoint)

            with io.open(os.path.join(
                    model_dir,
                    cls.name + "_char_to_id.pkl"), 'rb') as f:
                char_to_id = pickle.load(f)
            with io.open(os.path.join(
                    model_dir,
                    cls.name + "_id_to_tag.pkl"), 'rb') as f:
                id_to_tag = pickle.load(f)

            return BilstmCRFEntityExtractor(
                component_config=meta,
                ent_tagger=model,
                session=sess,
                char_to_id=char_to_id,
                id_to_tag=id_to_tag)

        else:
            return BilstmCRFEntityExtractor(meta)

    def persist(self, model_dir):
        # type: (Text) -> Optional[Dict[Text, Any]]
        """Persist this model into the passed directory.
        Return the metadata necessary to load the model again."""
        if self.session is None:
            return {"classifier_file": None}
        checkpoint = os.path.join(model_dir, self.name + ".ckpt")

        try:
            os.makedirs(os.path.dirname(checkpoint))
        except OSError as e:
            # be happy if someone already created the path
            import errno
            if e.errno != errno.EEXIST:
                raise

        save_model(self.session, self.model, checkpoint, logger)

        with io.open(os.path.join(
                model_dir,
                self.name + "_char_to_id.pkl"), 'wb') as f:
            pickle.dump(self.char_to_id, f)
        with io.open(os.path.join(
                model_dir,
                self.name + "_id_to_tag.pkl"), 'wb') as f:
            pickle.dump(self.id_to_tag, f)

        return {"classifier_file": self.name + ".ckpt"}
