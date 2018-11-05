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

    }

    def __init__(self, component_config=None):
        """Construct a new count vectorizer using the sklearn framework."""

        super(WordVectorsFeaturizer, self).__init__(component_config)

    def train(self, training_data, cfg=None, **kwargs):

        code.interact(local=locals())

        pass


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
