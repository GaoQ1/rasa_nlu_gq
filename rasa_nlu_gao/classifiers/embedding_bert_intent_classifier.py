from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import logging
import os
from tqdm import tqdm

import typing
from typing import List, Text, Any, Optional, Dict

from rasa_nlu_gao.classifiers import INTENT_RANKING_LENGTH
from rasa_nlu_gao.components import Component
from multiprocessing import cpu_count
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

import GPUtil
from bert_serving.client import BertClient
from rasa_nlu_gao.models.lenet import conv_net

class EmbeddingBertIntentClassifier(Component):
    """Intent classifier using supervised bert embeddings."""

    name = "intent_classifier_tensorflow_embedding_bert"

    provides = ["intent", "intent_ranking"]

    requires = ["text_features"]

    defaults = {
        # nn architecture
        "num_hidden_layers": 2,
        "hidden_layer_size": [1024, 256],
        "batch_size": 256,
        "epochs": 300,
        "learning_rate": 0.001,

        # regularization
        "C2": 0.002,
        "droprate": 0.2,

        # flag if tokenize intents
        "intent_tokenization_flag": False,
        "intent_split_symbol": '_',

        # visualization of accuracy
        "evaluate_every_num_epochs": 10,  # small values may hurt performance
        "evaluate_on_num_examples": 1000,  # large values may hurt performance

        "config_proto": {
            "device_count": cpu_count(),
            "inter_op_parallelism_threads": 0,
            "intra_op_parallelism_threads": 0,
            "allow_growth": True
        }
    }

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["tensorflow"]

    def _load_nn_architecture_params(self):
        self.num_hidden_layers = self.component_config['num_hidden_layers']
        self.hidden_layer_size = self.component_config['hidden_layer_size']

        self.batch_size = self.component_config['batch_size']
        self.epochs = self.component_config['epochs']
        self.learning_rate = self.component_config['learning_rate']

    def _load_regularization_params(self):
        self.C2 = self.component_config['C2']
        self.droprate = self.component_config['droprate']

    def _load_flag_if_tokenize_intents(self):
        self.intent_tokenization_flag = self.component_config['intent_tokenization_flag']
        self.intent_split_symbol = self.component_config['intent_split_symbol']
        if self.intent_tokenization_flag and not self.intent_split_symbol:
            logger.warning("intent_split_symbol was not specified, "
                           "so intent tokenization will be ignored")
            self.intent_tokenization_flag = False

    def _load_visual_params(self):
        self.evaluate_every_num_epochs = self.component_config[
                                            'evaluate_every_num_epochs']
        if self.evaluate_every_num_epochs < 1:
            self.evaluate_every_num_epochs = self.epochs
        self.evaluate_on_num_examples = self.component_config[
                                            'evaluate_on_num_examples']

    @staticmethod
    def _check_hidden_layer_sizes(num_layers, layer_size, name=''):
        num_layers = int(num_layers)

        if num_layers < 0:
            logger.error("num_hidden_layers_{} = {} < 0."
                         "Set it to 0".format(name, num_layers))
            num_layers = 0

        if isinstance(layer_size, list) and len(layer_size) != num_layers:
            if len(layer_size) == 0:
                raise ValueError("hidden_layer_size_{} = {} "
                                 "is an empty list, "
                                 "while num_hidden_layers_{} = {} > 0"
                                 "".format(name, layer_size,
                                           name, num_layers))

            logger.error("The length of hidden_layer_size_{} = {} "
                         "does not correspond to num_hidden_layers_{} "
                         "= {}. Set hidden_layer_size_{} to "
                         "the first element = {} for all layers"
                         "".format(name, len(layer_size),
                                   name, num_layers,
                                   name, layer_size[0]))

            layer_size = layer_size[0]

        if not isinstance(layer_size, list):
            layer_size = [layer_size for _ in range(num_layers)]

        return num_layers, layer_size

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
                 session=None,  # type: Optional[tf.Session]
                 graph=None,  # type: Optional[tf.Graph]
                 message_placeholder=None,  # type: Optional[tf.Tensor]
                 intent_placeholder=None,  # type: Optional[tf.Tensor]
                 y_predict=None   # type: Optional[tf.Tensor]
                 ):
        # type: (...) -> None
        """Declare instant variables with default values"""
        self._check_tensorflow()
        super(EmbeddingBertIntentClassifier, self).__init__(component_config)

        # nn architecture parameters
        self._load_nn_architecture_params()

        # regularization
        self._load_regularization_params()
        # flag if tokenize intents
        self._load_flag_if_tokenize_intents()
        # visualization of accuracy
        self._load_visual_params()

        # check if hidden_layer_sizes are valid
        (self.num_hidden_layers,
         self.hidden_layer_size) = self._check_hidden_layer_sizes(
                                        self.num_hidden_layers,
                                        self.hidden_layer_size,
                                        name='hidden_layer')

        # transform numbers to intents
        self.inv_intent_dict = inv_intent_dict
        # encode all intents with numbers
        self.encoded_all_intents = encoded_all_intents

        # tf related instances
        self.session = session
        self.graph = graph
        self.a_in = message_placeholder
        self.b_in = intent_placeholder
        self.y_predict = y_predict

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

    def _output_training_stat(self, X, intents_for_X, is_training):
        """Output training statistics"""
        n = self.evaluate_on_num_examples
        ids = np.random.permutation(len(X))[:n]
        all_Y = self._create_all_Y(X[ids].shape[0])

        train_sim = self.session.run(self.y_predict,
                                     feed_dict={self.a_in: X[ids],
                                                self.b_in: all_Y,
                                                is_training: False})

        train_acc = np.mean(np.argmax(train_sim, -1) == intents_for_X[ids])
        return train_acc

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

        self.graph = tf.Graph()
        with self.graph.as_default():

            self.a_in = tf.placeholder(tf.float32, (None, X.shape[-1]), name='a')
            self.b_in = tf.placeholder(tf.float32, (None, Y.shape[-1]), name='b')

            is_training = tf.placeholder_with_default(False, shape=())

            # Create a graph for training
            logits_train = conv_net(self.a_in, num_classes, self.num_hidden_layers, self.hidden_layer_size, self.C2, self.droprate, is_training=True)

            # Define loss and optimizer (with train logits, for dropout to take effect)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_train, labels=self.b_in)) + tf.losses.get_regularization_loss()

            self.y_predict = tf.nn.softmax(logits_train)

            train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)

            # train tensorflow graph
            config_proto = self.get_config_proto(self.component_config)
            self.session = tf.Session(graph=self.graph, config=config_proto)
            self.session.run(tf.global_variables_initializer())

            pbar = tqdm(range(self.epochs), desc="Epochs")
            train_acc = 0
            last_loss = 0
            for ep in pbar:
                indices = np.random.permutation(len(X))

                batch_size = self.batch_size
                batches_per_epoch = (len(X) // batch_size + int(len(X) % batch_size > 0))

                ep_loss = 0
                for i in range(batches_per_epoch):
                    end_idx = (i + 1) * batch_size
                    start_idx = i * batch_size
                    batch_a = X[indices[start_idx:end_idx]]
                    batch_b = Y[indices[start_idx:end_idx]]

                    sess_out = self.session.run(
                        {'loss': loss, 'train_op': train_op},
                        feed_dict={self.a_in: batch_a,
                                self.b_in: batch_b,
                                is_training: True}
                    )

                    ep_loss += sess_out.get('loss') / batches_per_epoch

                if self.evaluate_on_num_examples:
                    if (ep == 0 or
                            (ep + 1) % self.evaluate_every_num_epochs == 0 or
                            (ep + 1) == self.epochs):
                        train_acc = self._output_training_stat(X, intents_for_X,
                                                            is_training)
                        last_loss = ep_loss

                        pbar.set_postfix({
                            "loss": "{:.3f}".format(ep_loss),
                            "acc": "{:.3f}".format(train_acc)
                        })
                else:
                    pbar.set_postfix({
                        "loss": "{:.3f}".format(ep_loss)
                    })

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        """Return the most likely intent and its similarity to the input."""

        intent = {"name": None, "confidence": 0.0}
        intent_ranking = []

        if self.session is None:
            logger.error("There is no trained tf.session: "
                         "component is either not trained or "
                         "didn't receive enough training data")

        else:

            # get features (bag of words) for a message
            X = message.get("text_features").reshape(1, -1)

            # stack encoded_all_intents on top of each other
            # to create candidates for test examples
            all_Y = self._create_all_Y(X.shape[0])

            with self.graph.as_default():
                y_predict = self.session.run(self.y_predict, feed_dict={self.a_in: X, self.b_in: all_Y})
                
                intent_ids = y_predict[0][0]
                intent_id_argmax = np.argmax(intent_ids, -1)

            # if X contains all zeros do not predict some label
            if X.any():
                intent = {
                    "name": self.inv_intent_dict[intent_id_argmax], "confidence": float(intent_ids[intent_id_argmax])
                }
                ranking = intent_ids[:INTENT_RANKING_LENGTH]
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
        with self.graph.as_default():
            self.graph.clear_collection('message_placeholder')
            self.graph.add_to_collection('message_placeholder',
                                         self.a_in)

            self.graph.clear_collection('intent_placeholder')
            self.graph.add_to_collection('intent_placeholder',
                                         self.b_in)

            self.graph.clear_collection('y_predict')
            self.graph.add_to_collection('y_predict',
                                         self.y_predict)

            saver = tf.train.Saver()
            saver.save(self.session, checkpoint)

        with io.open(os.path.join(
                model_dir,
                self.name + "_inv_intent_dict.pkl"), 'wb') as f:
            pickle.dump(self.inv_intent_dict, f)
        with io.open(os.path.join(
                model_dir,
                self.name + "_encoded_all_intents.pkl"), 'wb') as f:
            pickle.dump(self.encoded_all_intents, f)

        return {"classifier_file": self.name + ".ckpt"}

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
        return config

    @classmethod
    def load(cls,
             model_dir=None,  # type: Text
             model_metadata=None,  # type: Metadata
             cached_component=None,  # type: Optional[Component]
             **kwargs  # type: **Any
             ):
        # type: (...) -> EmbeddingIntentClassifier

        meta = model_metadata.for_component(cls.name)
        config_proto = cls.get_config_proto(meta)

        if model_dir and meta.get("classifier_file"):
            file_name = meta.get("classifier_file")
            checkpoint = os.path.join(model_dir, file_name)
            graph = tf.Graph()
            with graph.as_default():
                sess = tf.Session(config=config_proto)
                saver = tf.train.import_meta_graph(checkpoint + '.meta')
                saver.restore(sess, checkpoint)

                a_in = tf.get_collection('message_placeholder')[0]
                b_in = tf.get_collection('intent_placeholder')[0]

                y_predict = tf.get_collection('y_predict')

            with io.open(os.path.join(
                    model_dir,
                    cls.name + "_inv_intent_dict.pkl"), 'rb') as f:
                inv_intent_dict = pickle.load(f)
            with io.open(os.path.join(
                    model_dir,
                    cls.name + "_encoded_all_intents.pkl"), 'rb') as f:
                encoded_all_intents = pickle.load(f)

            return EmbeddingBertIntentClassifier(
                    component_config=meta,
                    inv_intent_dict=inv_intent_dict,
                    encoded_all_intents=encoded_all_intents,
                    session=sess,
                    graph=graph,
                    message_placeholder=a_in,
                    intent_placeholder=b_in,
                    y_predict=y_predict
            )

        else:
            logger.warning("Failed to load nlu model. Maybe path {} "
                           "doesn't exist"
                           "".format(os.path.abspath(model_dir)))
            return EmbeddingBertIntentClassifier(component_config=meta)
