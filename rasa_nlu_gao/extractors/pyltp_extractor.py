from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import warnings

from builtins import str
from typing import Any
from typing import Dict
from typing import Optional
from typing import Text

from rasa_nlu_gao import utils
from rasa_nlu_gao.extractors import EntityExtractor
from rasa_nlu_gao.model import Metadata
from rasa_nlu_gao.training_data import Message
from rasa_nlu_gao.training_data import TrainingData
from rasa_nlu_gao.utils import write_json_to_file
from pyltp import Segmentor, Postagger


class PyLTPEntityExtractor(EntityExtractor):
    name = 'PyLTP_entity_extractor'

    provides = ["entities"]

    requires = ['tokens']

    defaults = {
        "model_path": None,  # Nh: name Ni: organization Ns: place
        "part_of_speech": ['nh'],
        "rename_to_entity": ['username'],  # rename 'nh' to 'username'
        "dictionary_path": None  # customize dictionary
    }

    def __init__(self, component_config=None):
        # type: (Optional[Dict[Text, Text]]) -> None

        super(PyLTPEntityExtractor, self).__init__(component_config)
        self.model_path = self.component_config.get('model_path')
        self.dictionary_path = self.component_config.get('dictionary_path')

        self.segmentor = Segmentor()
        self.postagger = Postagger()
        if self.dictionary_path is None:
            self.segmentor.load(self.model_path + "/cws.model")
            self.postagger.load(self.model_path+"/pos.model")
        else:
            self.segmentor.load_with_lexicon(
                self.model_path + "/cws.model", self.dictionary_path)
            self.postagger.load_with_lexicon(
                self.model_path+"/pos.model", self.dictionary_path)

    @classmethod
    def create(cls, cfg):
        component_conf = cfg.for_component(cls.name, cls.defaults)
        return PyLTPEntityExtractor(component_conf)

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["pyltp"]

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        extracted = self.add_extractor_name(self.extract_entities(message))
        message.set("entities", extracted, add_to_output=True)

    def extract_entities(self, message):
        # type: (Message) -> List[Dict[Text, Any]]
                # Set your own model path
        sentence = message.text
        words = self.segmentor.segment(sentence)
        postags = self.postagger.postag(words)
        result = zip(words, postags)

        raw_entities = message.get("entities", [])

        for word, postag in result:
            part_of_speech = self.component_config["part_of_speech"]
            rename_to_entity = self.component_config["rename_to_entity"]

            if postag in part_of_speech:
                start = sentence.index(word)
                end = start + len(word)

                entity_index = part_of_speech.index(postag)
                rename_entity = rename_to_entity[entity_index] or postag

                hasAlreadyExtractor = False

                for obj in raw_entities:
                    if obj and obj['value'] == word:
                        hasAlreadyExtractor = True

                if not hasAlreadyExtractor:
                    raw_entities.append({
                        'start': start,
                        'end': end,
                        'value': word,
                        'entity': rename_entity
                    })
        return raw_entities

    @classmethod
    def load(cls,
             model_dir=None,  # type: Optional[Text]
             model_metadata=None,  # type: Optional[Metadata]
             cached_component=None,  # type: Optional[Component]
             **kwargs  # type: **Any
             ):

        meta = model_metadata.for_component(cls.name)

        return cls(meta)
