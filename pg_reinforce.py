# NB. For Mountain Car try more episodes


import random
import numpy as np
import tensorflow as tf

class PolicyGradientREINFORCE(object):

  def __init__(self, session,
                     optimizer,
                     policy_network,
                     state_dim,
                     num_actions,
                     gru_unit_size,
                     num_step,
                     num_layers,
                     save_path,
                     global_step,
                     max_gradient=5,       # max gradient norms
                     entropy_bonus=0.001,
                     summary_writer=None,
                     summary_every=100,
                     loss_function="l2"):

    # tensorflow machinery
    self.session        = session
    self.optimizer      = optimizer
    self.summary_writer = summary_writer
    self.summary_every  = summary_every
    self.gru_unit_size  = gru_unit_size
    self.num_step       = num_step
    self.num_layers     = num_layers
    self.no_op          = tf.no_op()

    # model components
    self.policy_network = policy_network
    self.state_dim = state_dim
    self.num_actions = num_actions
    self.loss_function = loss_function

    # training parameters
    self.max_gradient    = max_gradient
    self.entropy_bonus   = entropy_bonus

    #counter
    self.global_step = global_step
    self.reward_stats = []

    # create and initialize variables
    self.create_variables()
    var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    self.session.run(tf.variables_initializer(var_lists))

    # make sure all variables are initialized
    self.session.run(tf.assert_variables_initialized())

    # try load saved model
    self.saver = tf.train.Saver(tf.global_variables())
    self.save_path = save_path
    self.load_model()

    if self.summary_writer is not None:
      # graph was not available when journalist was created
      self.summary_writer.add_graph(self.session.graph)
      self.summary_every = summary_every

  def create_input_placeholders(self):
    with tf.name_scope("inputs"):
      self.states = tf.placeholder(tf.float32, (None, None, self.state_dim), name="states")
      self.actions = tf.placeholder(tf.int32, (None, None), name="actions")
      self.returns = tf.placeholder(tf.float32, (None, None), name="returns")
      self.advantages = tf.placeholder(tf.float32, (None, None), name="advantages")
      self.init_states = tuple(tf.placeholder(tf.float32, (None, self.gru_unit_size),
                                                                 name="init_states")
                                                    for _ in range(self.num_layers))
      self.seq_len = tf.placeholder(tf.int32, (None,), name="seq_len")
      self.reward = tf.placeholder(tf.float32, None, name="reward")

  def create_variables_for_actions(self):
    with tf.name_scope("generating_actions"):
      with tf.variable_scope("policy_network"):
        self.logit, self.final_state, self.value = self.policy_network(self.states,
                                                    self.init_states, self.seq_len)
      self.probs = tf.nn.softmax(self.logit)
      with tf.name_scope("computing_entropy"):
        self.entropy = tf.reduce_sum(
                tf.multiply(self.probs, -1.0 * tf.log(self.probs)), axis=1)

  def create_variables_for_optimization(self):
    with tf.name_scope("optimization"):
      with tf.name_scope("masker"):
          self.mask = tf.sequence_mask(self.seq_len, self.num_step)
          self.maskA = tf.zeros([tf.shape(self.mask)[0], self.num_step // 2])
          self.maskB = tf.ones([tf.shape(self.mask)[0],
                               self.num_step - self.num_step // 2])
          self.truncated_mask = tf.concat([self.maskA, self.maskB], axis=1)

          self.mask = tf.reshape(tf.cast(self.mask, tf.float32), (-1,))
          self.truncated_mask = tf.reshape(self.truncated_mask, (-1,))

      if self.loss_function == "cross_entropy":
        self.loss_applied = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                            logits=self.logit,
                                            labels=tf.reshape(self.actions, (-1,)))
      elif self.loss_function == "l2":
        self.one_hot_actions = tf.one_hot(tf.reshape(self.actions, (-1,)),
                                                        self.num_actions)
        self.loss_applied = tf.reduce_mean((self.probs - self.one_hot_actions) ** 2,
                                            axis=1)
      else:
          raise ValueError("loss function type is not defined")

      self.masked_loss_applied = tf.multiply(self.loss_applied, self.mask)
      self.masked_loss_applied = tf.multiply(self.masked_loss_applied,
                                                self.truncated_mask)

      self.masked_entropy = tf.multiply(self.entropy, self.mask)
      self.maksed_entropy = tf.multiply(self.masked_entropy,
                                            self.truncated_mask)
      self.masked_entropy = tf.reduce_mean(self.masked_entropy)

      self.value_loss = (self.value - tf.reshape(self.returns, (-1,))) ** 2
      self.masked_value_loss = tf.multiply(self.value_loss, self.mask)
      self.masked_value_loss = tf.multiply(self.masked_value_loss,
                                           self.truncated_mask)


      self.loss = (tf.reduce_mean(tf.multiply(self.masked_loss_applied,
                                 tf.reshape(self.advantages, (-1,))))
                   + tf.reduce_mean(self.masked_value_loss))

      self.loss -= self.entropy_bonus * self.masked_entropy

      self.gradients = self.optimizer.compute_gradients(self.loss)
      self.clipped_gradients = [(tf.clip_by_norm(grad, self.max_gradient), var)
                                  for grad, var in self.gradients
                                  if grad is not None]

      self.train_op = self.optimizer.apply_gradients(self.clipped_gradients,
                                                     self.global_step)

  def create_summaries(self):
    self.loss_summary = tf.summary.scalar("loss", self.loss)
    self.reward_summary = tf.summary.scalar("reward", self.reward)
    self.histogram_summaries = []
    for grad, var in self.gradients:
        if grad is not None:
            histogram_summary = tf.summary.histogram(var.name + "/gradient", grad)
            self.histogram_summaries.append(histogram_summary)
    self.entropy_summary = tf.summary.scalar("entropy", self.masked_entropy)
    self.weight_summaries = []
    weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="policy_network")
    for w in weights:
        w_summary = tf.summary.histogram("policy_weights/" + w.name, w)
        self.weight_summaries.append(w_summary)

  def merge_summaries(self):
    self.summarize = tf.summary.merge([self.loss_summary + self.entropy_summary]
                                      + self.histogram_summaries
                                      + self.weight_summaries
                                      + [self.reward_summary])

  def load_model(self):
    try:
        save_dir = '/'.join(self.save_path.split('/')[:-1])
        ckpt = tf.train.get_checkpoint_state(save_dir)
        load_path = ckpt.model_checkpoint_path
        self.saver.restore(self.session, load_path)
    except:
        print("no saved model to load. starting new session")
    else:
        print("loaded model: {}".format(load_path))
        self.saver = tf.train.Saver(tf.global_variables())

  def create_variables(self):
    self.create_input_placeholders()
    self.create_variables_for_actions()
    self.create_variables_for_optimization()
    self.create_summaries()
    self.merge_summaries()

  def sampleAction(self, states, init_states, seq_len=[1]):
    probs, final_state, value = self.session.run([self.probs,
                                                       self.final_state, self.value],
                                 {self.states: states, self.init_states: init_states,
                                  self.seq_len: seq_len})
    return np.random.choice(self.num_actions, p=probs[0]), final_state, value

  def compute_action_probabilities(self, states, init_states, seq_len):
    return self.session.run(self.probs, {self.states: states,
                                         self.init_states: init_states,
                                         self.seq_len: seq_len})

  def update_parameters(self, states, actions, returns, advantages,
                        init_states, seq_len, reward):
    train_itr = self.session.run(self.global_step)
    write_summary = train_itr % self.summary_every == 0
    self.reward_stats.append(reward)
    _, summary = self.session.run([self.train_op,
                                   self.summarize if write_summary else self.no_op],
                                  {self.states: states,
                                   self.actions: actions,
                                   self.returns: returns,
                                   self.advantages: advantages,
                                   self.init_states: init_states,
                                   self.seq_len: seq_len,
                                   self.reward: np.mean(self.reward_stats)})

    if write_summary:
        self.summary_writer.add_summary(summary, train_itr + 1)
        self.saver.save(self.session, self.save_path, global_step=self.global_step)
        print ("SAVED MODEL #{}".format(train_itr + 1))
        self.reward_stats = []
