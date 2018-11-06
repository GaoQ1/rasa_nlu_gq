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
        "vector": os.path.join("data", "vectors.txt")
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
        return re.sub(r'\b[0-9]+\b', '__NUMBER__', text)

    def _get_message_text(self, message):
        unk_vec = np.zeros((self.model.vector_size,))
        all_tokens = [unk_vec]

        for t in message.get("tokens"):
            text = self._replace_number(t.text)
            if text in self.model.vocab:
                all_tokens.append(self.model[text])

        return np.array(all_tokens).mean(axis=0)

    def train(self, training_data, cfg=None, **kwargs):
        tokens_text = [self._get_message_text(example) for example in training_data.intent_examples]
        X = np.array(tokens_text)

        for i, example in enumerate(training_data.intent_examples):
            example.set("text_features", self._combine_with_existing_text_features(example, X[i]))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        message_text = self._get_message_text(message)

        message.set("text_features", self._combine_with_existing_text_features(message, message_text))

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        """Persist this model into the passed directory.
        Returns the metadata necessary to load the model again."""

        featurizer_file = os.path.join(model_dir, self.name + ".pkl")
        utils.pycloud_pickle(featurizer_file, self)
        return {"featurizer_file": self.name + ".pkl"}

    @classmethod
    def load(cls,
             model_dir=None,  # type: Text
             model_metadata=None,  # type: Metadata
             cached_component=None,  # type: Optional[Component]
             **kwargs  # type: **Any
             ):

        meta = model_metadata.for_component(cls.name)

        if model_dir and meta.get("featurizer_file"):
            file_name = meta.get("featurizer_file")
            featurizer_file = os.path.join(model_dir, file_name)
            return utils.pycloud_unpickle(featurizer_file)
        else:
            logger.warning("Failed to load featurizer. Maybe path {} "
                           "doesn't exist".format(os.path.abspath(model_dir)))
            return WordVectorsFeaturizer(meta)
