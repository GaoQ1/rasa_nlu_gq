import logging
from typing import List, Text, Any, Optional, Dict
from rasa_nlu_gao.classifiers import INTENT_RANKING_LENGTH

from rasa.nlu.components import Component
from rasa.nlu.model import Metadata
from rasa.nlu.training_data import Message

import os
import shutil
import kashgari
from kashgari.embeddings import BERTEmbedding
import kashgari.tasks.classification as clf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

logger = logging.getLogger(__name__)


class KashgariIntentClassifier(Component):
    provides = ["intent", "intent_ranking"]

    defaults = {
        "bert_model_path": None,
        "sequence_length": "auto",
        "layer_nums": 4,
        "trainable": False,
        "classifier_model": "BiLSTM_Model",
        "epochs": 10,
        "batch_size": 32,
        "validation_split": 0.2,
        "patience": 5,
        "factor": 0.5, # factor of reduce learning late everytime
        "verbose": 1
    }

    def __init__(self,
                 component_config=None,
                 model = None):
        super(KashgariIntentClassifier, self).__init__(component_config)

        bert_model_path = self.component_config.get('bert_model_path')
        sequence_length = self.component_config.get('sequence_length')
        layer_nums = self.component_config.get('layer_nums')
        trainable = self.component_config.get('trainable')

        self.classifier_model = self.component_config.get('classifier_model')
        

        self.bert_embedding = BERTEmbedding(bert_model_path,
                                        task=kashgari.CLASSIFICATION,
                                        layer_nums = layer_nums,
                                        trainable=trainable,
                                        sequence_length=sequence_length)

        self.tokenizer = self.bert_embedding.tokenizer

        self.model = model

    async def train(self, training_data, cfg, **kwargs):
        classifier_model = eval("clf." + self.classifier_model)

        epochs = self.component_config.get('epochs')
        batch_size = self.component_config.get('batch_size')
        validation_split = self.component_config.get('validation_split')
        patience = self.component_config.get('patience')
        factor = self.component_config.get('factor')
        verbose = self.component_config.get('verbose')

        X, Y = [], []
        for msg in training_data.intent_examples:
            X.append(self.tokenizer.tokenize(msg.text))
            Y.append(msg.get('intent'))

        train_x, validate_x, train_y, validate_y = train_test_split( X, Y, test_size=validation_split, random_state=100)

        self.bert_embedding.processor.add_bos_eos = False

        self.model = classifier_model(self.bert_embedding)

        checkpoint = ModelCheckpoint(
            'intent_weights.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=verbose)
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience)
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=factor,
            patience=patience,
            verbose=verbose)

        self.model.fit(
            train_x,
            train_y,
            validate_x,
            validate_y,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint, early_stopping, reduce_lr]
        )

    async def process(self, message, **kwargs):
        intent_ranks = self.get_intent_score(message)
        intent = intent_ranks[0]

        message.set("intent", intent, add_to_output=True)
        message.set("intent_ranking", intent_ranks, add_to_output=True)

    def get_intent_score(self, message):
        intent_top_k = self.model.predict_top_k_class(
            [self.tokenizer.tokenize(message.text)],
            top_k = INTENT_RANKING_LENGTH
        )[0]

        intent_ranks = [{
            'name': intent_top_k['label'],
            'confidence': float(intent_top_k['confidence'])
        }]

        for item in intent_top_k['candidates']:
            intent_ranks.append({'name': item['label'], 'confidence': float(item['confidence'])})

        return intent_ranks

    def persist(self,
                file_name: Text,
                model_dir: Text) -> Optional[Dict[Text, Any]]:
        model_path = os.path.join(model_dir, file_name)
        self.model.save(model_path)
        
        remove_file = os.path.join(model_path, 'model_weights.h5')
        os.remove(remove_file)
        shutil.move('intent_weights.h5', model_path)
        os.rename(os.path.join(model_path, 'intent_weights.h5'), os.path.join(model_path, 'model_weights.h5'))

        return {"file": file_name}

    @classmethod
    def load(cls,
             meta: Dict[Text, Any],
             model_dir: Optional[Text]=None,
             model_metadata: Optional['Metadata']=None,
             cached_component: Optional[Component]=None,
             **kwargs: Any
             ) -> 'KashgariIntentClassifier':

        if model_dir and meta.get("file"):
            file_name = meta.get("file")
            classifier_model = os.path.join(model_dir, file_name)
            loaded_model = kashgari.utils.load_model(classifier_model)
            
            return cls(component_config=meta,
                       model=loaded_model)
        else:
            logger.warning("Failed to load classifier model. Maybe path {} "
                           "doesn't exist"
                           "".format(os.path.abspath(model_dir)))
            return cls(component_config=meta)
