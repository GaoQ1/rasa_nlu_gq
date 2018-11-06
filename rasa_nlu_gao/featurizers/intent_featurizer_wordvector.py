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
from rasa_nlu_gao.training_data import TrainingData
from rasa_nlu_gao.components import Component
from rasa_nlu_gao.config import RasaNLUModelConfig
from rasa_nlu_gao.model import Metadata

logger = logging.getLogger(__name__)

import code

class WordVectorsFeaturizer(Featurizer):
    name = "intent_featurizer_wordvector"

    provides = ["text_features"]

    requires = ["tokens"]

    defaults = {
        "vector": os.path.join("data", "vectors.txt"),
    }

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["gensim", "numpy"]

    def __init__(self, component_config=None, model=None):
        """Construct a new count vectorizer using the sklearn framework."""

        super(WordVectorsFeaturizer, self).__init__(component_config)
        self.model = model

    @classmethod
    def create(cls, cfg):
        import gensim

        component_conf = cfg.for_component(cls.name, cls.defaults)
        vector_file = component_conf.get("vector")
        if not vector_file:
            raise Exception("The WordVectorsFeaturizer component needs "
                            "the configuration value for 'vectors'.")
        model = gensim.models.KeyedVectors.load_word2vec_format(vector_file, binary=False)

        return WordVectorsFeaturizer(component_conf, model)

    @staticmethod
    def _replace_number(text):
        text = re.sub(r'\b[0-9]+\b', '__NUMBER__', text)
        return text

    def _get_message_text(self, message):
        for t in message.get("tokens"):
            
            text = self._replace_number(t.text)

            code.interact(local=locals())
            pass


        return ' '.join([self._replace_number(t.text) for t in message.get("tokens")])
    

    def train(self, training_data, cfg=None, **kwargs): # 将现有词向量取出来
        import numpy as np

        tokens_text = [self._get_message_text(example) for example in training_data.intent_examples]

        # X = self.vect.fit_transform(lem_exs).toarray()

        # rt = [self.model.get_vector(token.split(" ")) for token in tokens_text]
        rt = [token.split(" ") for token in tokens_text]


        # code.interact(local=locals())


        for i, example in enumerate(training_data.intent_examples):
            # create bag for each example
            example.set("text_features",
                        self._combine_with_existing_text_features(example,
                                                                  X[i]))

        


    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        pass

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        pass

    @classmethod
    def load(cls,
             model_dir=None,  # type: Text
             model_metadata=None,  # type: Metadata
             cached_component=None,  # type: Optional[Component]
             **kwargs  # type: **Any
             ):

        meta = model_metadata.for_component(cls.name)
