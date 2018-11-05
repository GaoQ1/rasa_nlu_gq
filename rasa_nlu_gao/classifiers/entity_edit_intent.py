from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import map
from typing import Any
from typing import Dict
from typing import Text

from rasa_nlu_gao.components import Component
from rasa_nlu_gao.training_data import Message

class EntityEditIntent(Component):
    name = "entity_edit_intent"

    provides = ["intent"]

    defaults = {
        "entity": ["nr"],
        "intent": ["enter_data"],
        "min_confidence": 0
    }

    def __init__(self, component_config=None):
        super(EntityEditIntent, self).__init__(component_config)

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        last_entities = message.get("entities", [])
        last_intent = message.get("intent", {})
    
        if last_intent["confidence"] <= self.component_config["min_confidence"]:
            for item in last_entities:
                if item["entity"] in self.component_config["entity"]:
                    entity_index = self.component_config["entity"].index(item["entity"])
                    intent_name = self.component_config["intent"][entity_index]

                    intent = {"name": intent_name, "confidence": 1.0}

                    message.set("intent", intent, add_to_output=True)
