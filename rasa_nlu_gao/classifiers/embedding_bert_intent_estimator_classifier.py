from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import logging
import os

import typing
from typing import List, Text, Any, Optional, Dict

from rasa_nlu_gao.classifiers import INTENT_RANKING_LENGTH
from rasa_nlu_gao.components import Component
from multiprocessing import cpu_count
from tensorflow.contrib import predictor as Pred
import numpy as np

try:
    import cPickle as pickle
except ImportError:
    import pickle

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import tensorflow as tf
    from rasa_nlu_gao.config import RasaNLUModelConfig
    from rasa_nlu_gao.training_data import TrainingData
    from rasa_nlu_gao.model import Metadata
    from rasa_nlu_gao.training_data import Message

try:
    import tensorflow as tf
except ImportError:
    tf = None


class EmbeddingBertIntentEstimatorClassifier(Component):
    """Intent classifier using supervised bert embeddings."""

    name = "intent_estimator_classifier_tensorflow_embedding_bert"

    provides = ["intent", "intent_ranking"]

    requires = ["text_features"]

    defaults = {
        # nn architecture
        "batch_size": 256,
        "epochs": 200,

        # flag if tokenize intents
        "intent_tokenization_flag": False,
        "intent_split_symbol": '_',

        "config_proto": {
            "device_count": cpu_count(),
            "inter_op_parallelism_threads": 0,
            "intra_op_parallelism_threads": 0,
            "allow_growth": True,
            "allocator_type": 'BFC',               # best-fit with coalescing algorithm 内存分配、释放、碎片管理
            "per_process_gpu_memory_fraction": 0.5 # this means use 50% of your gpu memory in max
        }
    }

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["tensorflow"]

    def _load_nn_architecture_params(self):
        self.batch_size = self.component_config['batch_size']
        self.epochs = self.component_config['epochs']

    def _load_flag_if_tokenize_intents(self):
        self.intent_tokenization_flag = self.component_config['intent_tokenization_flag']
        self.intent_split_symbol = self.component_config['intent_split_symbol']
        if self.intent_tokenization_flag and not self.intent_split_symbol:
            logger.warning("intent_split_symbol was not specified, "
                           "so intent tokenization will be ignored")
            self.intent_tokenization_flag = False

    @staticmethod
    def _check_tensorflow():
        if tf is None:
            raise ImportError(
                'Failed to import `tensorflow`. '
                'Please install `tensorflow`. '
                'For example with `pip install tensorflow`.')

    def __init__(self,
                 component_config=None,  # type: Optional[Dict[Text, Any]]
                 inv_intent_dict=None,  # type: Optional[Dict[int, Text]]
                 encoded_all_intents=None,  # type: Optional[np.ndarray]
                 estimator=None,
                 predictor=None,
                 feature_columns=None
                 ):
        # type: (...) -> None
        """Declare instant variables with default values"""
        self._check_tensorflow()
        super(EmbeddingBertIntentEstimatorClassifier, self).__init__(component_config)

        # nn architecture parameters
        self._load_nn_architecture_params()

        # flag if tokenize intents
        self._load_flag_if_tokenize_intents()

        # transform numbers to intents
        self.inv_intent_dict = inv_intent_dict
        # encode all intents with numbers
        self.encoded_all_intents = encoded_all_intents

        # tf related instances
        self.estimator = estimator
        self.predictor = predictor
        self.feature_columns = feature_columns

    # training data helpers:
    @staticmethod
    def _create_intent_dict(training_data):
        """Create intent dictionary"""

        distinct_intents = set([example.get("intent")
                               for example in training_data.intent_examples])
        return {intent: idx
                for idx, intent in enumerate(sorted(distinct_intents))}

    @staticmethod
    def _create_intent_token_dict(intents, intent_split_symbol):
        """Create intent token dictionary"""

        distinct_tokens = set([token
                               for intent in intents
                               for token in intent.split(
                                        intent_split_symbol)])
        return {token: idx
                for idx, token in enumerate(sorted(distinct_tokens))}

    def _create_encoded_intents(self, intent_dict):
        """Create matrix with intents encoded in rows as bag of words,
        if intent_tokenization_flag = False this is identity matrix"""

        if self.intent_tokenization_flag:
            intent_token_dict = self._create_intent_token_dict(
                list(intent_dict.keys()), self.intent_split_symbol)

            encoded_all_intents = np.zeros((len(intent_dict),
                                            len(intent_token_dict)))
            for key, idx in intent_dict.items():
                for t in key.split(self.intent_split_symbol):
                    encoded_all_intents[idx, intent_token_dict[t]] = 1

            return encoded_all_intents
        else:
            return np.eye(len(intent_dict))

    # data helpers:
    def _create_all_Y(self, size):
        # stack encoded_all_intents on top of each other
        # to create candidates for training examples
        # to calculate training accuracy
        all_Y = np.stack([self.encoded_all_intents[0] for _ in range(size)])

        return all_Y

    def _prepare_data_for_training(self, training_data, intent_dict):
        """Prepare data for training"""

        X = np.stack([e.get("text_features")
                      for e in training_data.intent_examples])

        intents_for_X = np.array([intent_dict[e.get("intent")]
                                  for e in training_data.intent_examples])

        Y = np.stack([self.encoded_all_intents[intent_idx]
                      for intent_idx in intents_for_X])

        return X, Y, intents_for_X

    def input_fn(self,features, labels, batch_size, shuffle_num, mode):
        """
         build tf.data set for input pipeline

        :param features: type dict() , define input x structure for parsing
        :param labels: type np.array input label
        :param batch_size: type int number ,input batch_size
        :param shuffle_num: type int number , random select the data
        :param mode: type string ,tf.estimator.ModeKeys.TRAIN or tf.estimator.ModeKeys.PREDICT
        :return: set() with type of (tf.data , and labels)
        """
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(shuffle_num).batch(batch_size).repeat(self.epochs)
        else:
            dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        data, labels = iterator.get_next()
        return data, labels


    def train(self, training_data, cfg=None, **kwargs):
        # type: (TrainingData, Optional[RasaNLUModelConfig], **Any) -> None
        """Train the embedding intent classifier on a data set."""

        intent_dict = self._create_intent_dict(training_data)

        if len(intent_dict) < 2:
            logger.error("Can not train an intent classifier. "
                         "Need at least 2 different classes. "
                         "Skipping training of intent classifier.")
            return

        self.inv_intent_dict = {v: k for k, v in intent_dict.items()}
        self.encoded_all_intents = self._create_encoded_intents(intent_dict)

        X, Y, intents_for_X = self._prepare_data_for_training(training_data, intent_dict)

        num_classes = len(intent_dict)

        # define classes number to classified
        head = tf.contrib.estimator.multi_class_head(n_classes=num_classes)

        # define feature spec for input x parsing
        feature_names = ['a_in']
        self.feature_columns = [tf.feature_column.numeric_column(key=k,shape=[1, X.shape[1]]) for k in feature_names]

        x_tensor = {'a_in': X}
        intents_for_X = intents_for_X.astype(np.int32)

        # set gpu and tf graph confing
        tf.logging.set_verbosity(tf.logging.INFO)
        config_proto = self.get_config_proto(self.component_config)

        # sparse_softmax_cross_entropy , build linear classified model
        self.estimator = tf.contrib.estimator.LinearEstimator(
                                                     head = head,
                                                     feature_columns=self.feature_columns,
                                                     optimizer='Ftrl',
                                                     config=tf.estimator.RunConfig(session_config=config_proto)
                                                 )
        # train model
        self.estimator.train(input_fn=lambda: self.input_fn(x_tensor,
                                                  intents_for_X,
                                                  self.batch_size,
                                                  shuffle_num=1000,
                                                  mode = tf.estimator.ModeKeys.TRAIN),
                                                  max_steps=2000)
        # evaluate model
        results = self.estimator.evaluate(input_fn=lambda: self.input_fn(x_tensor,
                                                  intents_for_X,
                                                  self.batch_size,
                                                  shuffle_num=1000,
                                                  mode = tf.estimator.ModeKeys.PREDICT))

        print(results)


    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        """Return the most likely intent and its similarity to the input."""

        intent = {"name": None, "confidence": 0.0}
        intent_ranking = []

        if self.predictor is None:
            logger.error("There is no trained tf.session: "
                         "component is either not trained or "
                         "didn't receive enough training data")

        else:
            X = message.get("text_features").tolist()
            examples = []
            feature = {}
            # convert input x to tf.feature with float feature spec
            feature['a_in'] = tf.train.Feature(float_list=tf.train.FloatList(value=X))
            # build tf.example for prediction
            example = tf.train.Example(
                features=tf.train.Features(
                    feature=feature
                )
            )
            # serialize tf.example to string
            examples.append(example.SerializeToString())

            # Make predictions.
            result_dict = self.predictor({'inputs': examples})
            result_score_list = result_dict['scores'][0]
            max_score = np.max(result_dict['scores'][0])
            max_index = np.argmax(result_dict['scores'][0])

            # if X contains all zeros do not predict some label
            if len(X)>0:
                intent = {
                    "name": self.inv_intent_dict[max_index], "confidence": float(max_score)
                }
                ranking = result_score_list[:INTENT_RANKING_LENGTH]
                intent_ranking = [{"name": self.inv_intent_dict[intent_idx],
                                   "confidence": float(score)}
                                  for intent_idx, score in enumerate(ranking)]

                intent_ranking = sorted(intent_ranking, key=lambda s: s['confidence'], reverse=True)

        message.set("intent", intent, add_to_output=True)
        message.set("intent_ranking", intent_ranking, add_to_output=True)

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        """Persist this model into the passed directory.
        Return the metadata necessary to load the model again."""
        if self.estimator is None:
            return {"classifier_file": None}

        # build feature spec for tf.example parsing
        feature_spec = tf.feature_column.make_parse_example_spec(self.feature_columns)
        # build tf.example parser
        serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
        # export tf model
        path = self.estimator.export_savedmodel(model_dir, serving_input_receiver_fn)
        # decode model path to string
        file_dir = os.path.basename(path).decode('utf-8')


        with io.open(os.path.join(
                model_dir,
                self.name + "_inv_intent_dict.pkl"), 'wb') as f:
            pickle.dump(self.inv_intent_dict, f)
        with io.open(os.path.join(
                model_dir,
                self.name + "_encoded_all_intents.pkl"), 'wb') as f:
            pickle.dump(self.encoded_all_intents, f)

        return {"classifier_file": file_dir}

    @staticmethod
    def get_config_proto(component_config):
        # 配置configProto
        config = tf.ConfigProto(
            device_count={
                'CPU': component_config['config_proto']['device_count']
            },
            inter_op_parallelism_threads=component_config['config_proto']['inter_op_parallelism_threads'],
            intra_op_parallelism_threads=component_config['config_proto']['intra_op_parallelism_threads'],
            gpu_options={
                'allow_growth': component_config['config_proto']['allow_growth']
            }
        )
        config.gpu_options.per_process_gpu_memory_fraction= component_config['config_proto']['per_process_gpu_memory_fraction']
        config.gpu_options.allocator_type = component_config['config_proto']['allocator_type']
        return config

    @classmethod
    def load(cls,
             model_dir=None,  # type: Text
             model_metadata=None,  # type: Metadata
             cached_component=None,  # type: Optional[Component]
             **kwargs  # type: **Any
             ):
        # type: (...) -> EmbeddingBertIntentAdanetClassifier

        meta = model_metadata.for_component(cls.name)
        config_proto = cls.get_config_proto(meta)

        print("bert model loaded")

        if model_dir and meta.get("classifier_file"):
            file_name = meta.get("classifier_file")
            # tensorflow.contrib.predictor to load the model file which may has 10x speed up in predict time
            predict = Pred.from_saved_model(export_dir=os.path.join(model_dir,file_name),config=config_proto)

            with io.open(os.path.join(
                    model_dir,
                    cls.name + "_inv_intent_dict.pkl"), 'rb') as f:
                inv_intent_dict = pickle.load(f)
            with io.open(os.path.join(
                    model_dir,
                    cls.name + "_encoded_all_intents.pkl"), 'rb') as f:
                encoded_all_intents = pickle.load(f)

            return EmbeddingBertIntentEstimatorClassifier(
                    component_config=meta,
                    inv_intent_dict=inv_intent_dict,
                    encoded_all_intents=encoded_all_intents,
                    predictor=predict
            )

        else:
            logger.warning("Failed to load nlu model. Maybe path {} "
                           "doesn't exist"
                           "".format(os.path.abspath(model_dir)))
            return EmbeddingBertIntentEstimatorClassifier(component_config=meta)





