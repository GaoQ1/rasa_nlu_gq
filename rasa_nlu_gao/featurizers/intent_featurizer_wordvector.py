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
import numpy as np

logger = logging.getLogger(__name__)

class WordVectorsFeaturizer(Featurizer):
    name = "intent_featurizer_wordvector"

    provides = ["text_features"]

    requires = ["tokens"]

    defaults = {
        "vector": None,
        "elmo": None,
        "limit": None
    }

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["gensim", "numpy", "torch"]

    def __init__(self, component_config=None, model=None, category=None):
        """Construct a new count vectorizer using the sklearn framework."""

        super(WordVectorsFeaturizer, self).__init__(component_config)
        self.model = model
        self.category = category

    @classmethod
    def create(cls, cfg):
        component_conf = cfg.for_component(cls.name, cls.defaults)

        vector_file = component_conf.get("vector")
        elmo_file = component_conf.get("elmo")
        
        if not vector_file and not elmo_file:
            raise Exception("The WordVectorsFeaturizer component needs "
                            "the configuration value either word2vec vector or elmo model.")
        
        if vector_file:
            import gensim
            model = gensim.models.KeyedVectors.load_word2vec_format(vector_file, binary=False, limit=component_conf.get("limit"))
            category = 'word2vec'
        elif elmo_file:
            from rasa_nlu_gao.models.elmo_cn import Embedder
            model = Embedder(elmo_file)
            category = 'elmo'

        return WordVectorsFeaturizer(component_conf, model, category)

    @staticmethod
    def _replace_number(text):
        return re.sub(r'\b[0-9]+\b', '0', text)

    def _chunk_max_pooling(self,data):
        import torch
        import torch.nn as  nn
        all_data = np.zeros((len(data), data[0].shape[1]), dtype=np.float32)
        for i in range(len(data)):
            stride = data[i].shape[0]
            m = nn.MaxPool1d(stride, stride)
            arr = torch.from_numpy(data[i]).reshape(1, 1, -1)
            arr = m(arr)
            arr = arr.reshape(1, -1)
            all_data[i] = arr[0].data.numpy()
        return np.squeeze(all_data)

    def _k_max_pooling(self,data):
        import torch
        all_data = np.zeros((len(data), data[0].shape[1]), dtype=np.float32)
        for i in range(len(data)):
            arr = torch.from_numpy(data[i]).reshape(-1)
            stride = data[i].shape[1]
            ind = arr.topk(stride)[1].sort()
            all_data[i] = arr[ind[0]].data.numpy()
        return np.squeeze(all_data)


    def _get_message_text(self, message):
        all_tokens = []
        all_t=[]
        for t in message.get("tokens"):
            text = self._replace_number(t.text)
            all_t.append(text)
            if self.category == 'word2vec':
                unk_vec = np.zeros((self.model.vector_size,))
                all_tokens.append(unk_vec)

                if text in self.model.vocab:
                    all_tokens.append(self.model[text])

        all_tokens = [np.vstack(tuple(all_tokens))]

        if self.category == 'elmo':
            all_tokens = self.model.sents2elmo([all_t])

        single_token = self._k_max_pooling(all_tokens)

        return single_token

    def train(self, training_data, cfg=None, **kwargs):
        tokens_text = [self._get_message_text(example) for example in training_data.intent_examples]
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

        if model_dir:
            vector_file = meta.get("vector")
            elmo_file = meta.get("elmo")

            if vector_file:
                import gensim
                model = gensim.models.KeyedVectors.load_word2vec_format(vector_file, binary=False, limit=meta.get("limit"))
                category = 'word2vec'
            elif elmo_file:
                from rasa_nlu_gao.models.elmo_cn import Embedder
                model = Embedder(elmo_file)
                category = 'elmo'

            return WordVectorsFeaturizer(
                component_config=meta,
                model=model,
                category=category)
        else:
            logger.warning("Failed to load featurizer. Maybe path {} "
                           "doesn't exist".format(os.path.abspath(model_dir)))
            return WordVectorsFeaturizer(meta)
