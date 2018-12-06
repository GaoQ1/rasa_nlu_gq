from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import

import logging
import os
import re
from typing import Any, Dict, List, Optional, Text

from rasa_nlu_gao import utils
from rasa_nlu_gao.featurizers import Featurizer
from rasa_nlu_gao.training_data import Message
from rasa_nlu_gao.components import Component
from rasa_nlu_gao.model import Metadata
from rasa_nlu_gao.models.bert_client import BertClient

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

class BertVectorsFeaturizer(Featurizer):
    name = "bert_vectors_featurizer"

    provides = ["text_features"]

    requires = ["tokens"]

    defaults = {
        "ip": 'localhost',
        "port": 5555
    }

    @classmethod
    def required_packages(cls):
        return ["numpy"]

    def __init__(self, component_config=None):
        super(BertVectorsFeaturizer, self).__init__(component_config)
        ip = self.component_config['ip']
        port = self.component_config['port']
        self.bc = BertClient(ip=ip, port=int(port))

    @classmethod
    def create(cls, cfg):
        component_conf = cfg.for_component(cls.name, cls.defaults)
        return BertVectorsFeaturizer(component_conf)

    @staticmethod
    def _replace_number(text):
        return re.sub(r'\b[0-9]+\b', '0', text)

    def _get_message_text(self, message):
        all_tokens = []

        for t in message.get("tokens"):
            text = self._replace_number(t.text)
            all_tokens.append(text)

        bert_embedding = self.bc.encode([' '.join(all_tokens)])

        return np.squeeze(bert_embedding)


    def train(self, training_data, cfg=None, **kwargs):
        tokens_text = [self._get_message_text(example) for example in tqdm(training_data.intent_examples)]

        X = np.array(tokens_text)

        for i, example in enumerate(training_data.intent_examples):
            example.set("text_features", self._combine_with_existing_text_features(example, X[i]))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        message_text = self._get_message_text(message)

        message.set("text_features", self._combine_with_existing_text_features(message, message_text))

    @classmethod
    def load(cls,
             model_dir=None,  # type: Text
             model_metadata=None,  # type: Metadata
             cached_component=None,  # type: Optional[Component]
             **kwargs  # type: **Any
             ):

        meta = model_metadata.for_component(cls.name)

        return cls(meta)
