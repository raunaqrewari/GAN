# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

import utils

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("function", "train",
                    "What should the model do: train, evaluate")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_bool("mode", "generator",
                  "What do you want the model to do. To train generator only, use --generator; to train discriminator only, use --discriminator; to train them together like GANs, use --gan")
FLAGS = flags.FLAGS

class SmallConfig(object):
  """Small config."""
  embed_size = 100 
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 10
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 0.75
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000
  lr = 0.001
  generate_length = 20


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8
  batch_size = 20
  vocab_size = 10000


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10000

class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000

def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32

# class Input(object):
#   """The input data."""
#   def __init__(self, config, data, name=None):
#     self.batch_size = batch_size = config.batch_size
#     self.num_steps = num_steps = config.num_steps
#     minibatch = utils.ptb_producer(data, batch_size, num_steps, name=name)
#     self.input_data = minibatch[0] 
#     self.targets = minibatch[1]
#     print (minibatch)


class Generator(object):
  def __init__(self, config, embeddings, is_training):
    self.config = config
    self.batch_size = config.batch_size
    self.num_steps = config.num_steps
    self.size = config.hidden_size
    self.vocab_size = config.vocab_size
    self.embeddings = embeddings
    self.is_training = is_training
    self.grad_norm = None
    self.build()


  def build(self):
    self.add_placeholders()
    self.hidden, self.pred = self.add_prediction_op()
    self.loss = self.add_loss_op(self.pred)
    self.train_op = self.add_training_op(self.loss)


  def add_placeholders(self):
    self.input_placeholder = tf.placeholder(tf.int32, shape=(self.batch_size, self.num_steps))
    self.labels_placeholder = tf.placeholder(tf.int32, shape = (self.batch_size, self.num_steps))


  def create_feed_dict(self, inputs_batch, labels_batch = None):
    feed_dict = {
            self.input_placeholder: inputs_batch,
        }
    if labels_batch != None:
        feed_dict[self.labels_placeholder] = labels_batch
    return feed_dict

  def add_embedding(self):
    embedding_tensor = tf.Variable(self.embeddings)
    embeddings = tf.nn.embedding_lookup(embedding_tensor, self.input_placeholder)
    # embeddings = tf.get_variable("embedding", [vocab_size, size], dtype=data_type())
    embeddings = tf.reshape(embeddings, shape = [-1, self.num_steps, self.config.embed_size])
    return embeddings

  def add_prediction_op(self):
    inputs = self.add_embedding()

    def lstm_cell():
      return tf.contrib.rnn.BasicLSTMCell(self.size, forget_bias=1.0, state_is_tuple=True)
    
    attn_cell = lstm_cell
    if self.is_training and self.config.keep_prob < 1:
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob = self.config.keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell([attn_cell() for _ in range(self.config.num_layers)], state_is_tuple=True)

    self._initial_state = cell.zero_state(self.batch_size, data_type())

    if self.is_training and self.config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, self.config.keep_prob)

    xavier_initializer = tf.contrib.layers.xavier_initializer()
    W = tf.get_variable("W", shape = [self.size, self.vocab_size], dtype=data_type(), initializer= xavier_initializer)
    b = tf.get_variable("b", [self.vocab_size], dtype=data_type())

    hidden_states = []
    logits = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(self.num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (hidden, state) = cell(inputs[:, time_step, :], state)
        logit = tf.matmul(hidden, W) + b
        hidden_states.append(hidden)
        logits.append(logit)

    logits = tf.stack(logits)
    logits = tf.transpose(logits, perm = [1, 0, 2])
    return hidden_states, logits

  def add_loss_op(self, logits):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = self.labels_placeholder))
    return loss

  def add_training_op(self, loss):
    # self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    # tf.gradients(cost, tvars) computes the gradient of cost with respect to all variables in tvars
    # tf.clip_by_global_norm() takes a list of gradients to perform gradient clipping on
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
                                      self.config.max_grad_norm)
    optimizer = tf.train.AdamOptimizer(self.config.lr)
    train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())

    # self._new_lr = tf.placeholder(
    #     tf.float32, shape=[], name="new_learning_rate")
    # self._lr_update = tf.assign(self._lr, self._new_lr)

    return train_op

  def train_on_batch(self, sess, inputs_batch, labels_batch):
    """Perform one step of gradient descent on the provided batch of data.
    This version also returns the norm of gradients.
    """
    feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch)
    _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
    return loss

  def run_epoch(self, sess, train_data):
    losses = 0
    count = 0
    for input_batch in utils.ptb_producer(train_data, self.batch_size, self.num_steps):
      count += 1
      # minibatch = Input(config=self.config, data=train_data)
      loss = self.train_on_batch(sess, input_batch[0], input_batch[1])
      losses += loss
      # print ("Loss after 1 batch : ", np.sum(losses)/(count*self.batch_size))
      # print("perplexity: ", np.exp(np.sum(losses) / (self.batch_size * count)))

    return losses

  def fit(self, saver, sess, train_data):
    losses = 0
    self.epoch_size = ((len(train_data) // self.batch_size) - 1) // self.num_steps
    for epoch in range(self.config.max_max_epoch):
      loss = self.run_epoch(sess, train_data)
      losses += loss
      print ("Loss after 1 EPOCH = ", loss)
      # print ("Loss after 1 EPOCH = ", np.sum(losses[epoch])/(self.batch_size * self.epoch_size))
      # print("perplexity after an epoch: ", np.exp(np.sum(losses) / (self.epoch_size * epoch * self.batch_size)))
    save_path = saver.save(sess, "/Users/Raunaq/Dropbox/courses/Spring_16/CS224N/Project/code/ptb/checkpoints/model.ckpt")
    return losses

  def predict_on_batch(self, sess, inputs_batch, labels_batch):
    """Make predictions for the provided batch of data

    Args:
        sess: tf.Session()
        input_batch: np.ndarray of shape (n_samples, n_features)
    Returns:
        predictions: np.ndarray of shape (n_samples, n_classes)
    """
    predictions = []
    dictionary = utils.get_word_dict(FLAGS.data_path)
    for i in xrange(self.config.generate_length):
      feed = self.create_feed_dict(inputs_batch)
      preds = sess.run(self.pred, feed_dict=feed)
      inputs_batch = np.argmax(preds, axis = 2)
      predictions.append(([dictionary[int(inputs_batch[j,:])] for j in xrange(inputs_batch.shape[0])]))
    for i in xrange(self.config.batch_size):
      print ([predictions[j][i] for j in xrange(len(predictions))])


    # dictionary = utils.get_word_dict(FLAGS.data_path)
    # feed = self.create_feed_dict(inputs_batch)
    # predictions = sess.run(self.pred, feed_dict=feed)
    # for i in xrange(predictions.shape[0]):
    #   actual_sentence = " ".join([dictionary[j] for j in labels_batch[i,:]])
    #   predicted_sentence = " ".join([dictionary[j] for j in np.argmax(predictions[i,:,:], axis = 1)])
    #   print ("Actual Sentence = ", actual_sentence)
    #   print ("Predicted sentence = ", predicted_sentence)

    # return predictions

  def predict_on_data(self, session, test_data):
    print ("Here")
    for input_batch in utils.ptb_producer(test_data, self.batch_size, self.num_steps):
      # if is_gan == True:
      #   yield self.predict_on_batch(session, input_batch[0], input_batch[1], is_gan)
      self.predict_on_batch(session, input_batch[0], input_batch[1])



def do_train(model, train_data):
  init = tf.global_variables_initializer()
  saver = tf.train.Saver()

  with tf.Session() as session:
    session.run(init)
    losses = model.fit(saver, session, train_data)

    # print ("Loss = ", np.sum(losses)/len(train_data))


def do_evaluate(model, test_data):
  init = tf.global_variables_initializer()
  saver = tf.train.Saver()

  with tf.Session() as session:
    session.run(init)
    saver.restore(session, "/Users/Raunaq/Dropbox/courses/Spring_16/CS224N/Project/code/ptb/checkpoints/model.ckpt")
    print ("Session restored")
    model.predict_on_data(session, test_data)


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  train_data, valid_data, test_data, _ = utils.ptb_raw_data(FLAGS.data_path)
  config = get_config()
  embeddings = utils.get_embedding_matrix(FLAGS.data_path)

  if FLAGS.function == "train":
    with tf.name_scope("Train"):
      model = Generator(config, embeddings, is_training = True)
      do_train(model, train_data)

  if FLAGS.function == "evaluate":
    eval_config = get_config()
    eval_config.num_steps = 1
    model = Generator(eval_config, embeddings, is_training = False)
    do_evaluate(model, valid_data)
    # if FLAGS.mode == "gan":
    #   do_evaluate(model, valid_data, is_gan = True)
    # else:
    #   do_evaluate(model, valid_data, is_gan = False)

  if FLAGS.function == "test":
    eval_config = get_config()
    eval_config.num_steps = 1
    eval_config.batch_size = 1
    model = Generator(eval_config, embeddings, is_training = False)
    do_evaluate(model, test_data)
  print ("Done")

if __name__ == "__main__":
  tf.app.run()


