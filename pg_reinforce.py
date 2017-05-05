import random
import numpy as np
import tensorflow as tf

class PolicyGradientREINFORCE(object):

  def __init__(self, session,
                     optimizer,
                     policy_network,
                     observation_space_dim,
                     num_actions,
                     gru_unit_size,
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
    self.no_op          = tf.no_op()

    # model components
    self.policy_network = policy_network
    self.observation_space_dim = observation_space_dim
    self.num_actions = num_actions
    self.loss_function = loss_function

    # training parameters
    self.max_gradient    = max_gradient
    self.entropy_bonus   = entropy_bonus

    #counter
    self.global_step = global_step
    self.summary_counter = 0

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
      self.observations = tf.placeholder(tf.float32, (None,) + self.observation_space_dim,
                                    name="observations")
      self.actions = tf.placeholder(tf.int32, (None,), name="actions")
      self.returns = tf.placeholder(tf.float32, (None,), name="returns")
      self.advantages = tf.placeholder(tf.float32, (None,), name="advantages")
      self.init_state = tf.placeholder(tf.float32, (None, self.gru_unit_size),
                                                                 name="init_state")

  def create_variables_for_actions(self):
    with tf.name_scope("generating_actions"):
      with tf.variable_scope("policy_network"):
        self.logit, self.final_state, self.value = self.policy_network(self.observations,
                                                    self.init_state)
      self.probs = tf.nn.softmax(self.logit)
      self.entropy = - tf.reduce_sum(self.probs * tf.log(self.probs))

  def create_variables_for_optimization(self):
    with tf.name_scope("optimization"):
      if self.loss_function == "cross_entropy":
        self.negative_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                            logits=self.logit,
                                            labels=self.actions)
        self.pl_loss = tf.reduce_sum(self.negative_log_prob * self.advantages)
      elif self.loss_function == "l2":
        self.one_hot_actions = tf.one_hot(self.actions, self.num_actions)
        self.prob_diff = tf.reduce_sum((self.probs - self.one_hot_actions) ** 2,
                                            axis=1)
        self.pl_loss = tf.reduce_sum(self.prob_diff * self.advantages)
      else:
          raise ValueError("loss function type is not defined")


      self.value_loss = 0.5 * tf.reduce_sum((self.value - self.returns) ** 2)

      self.loss = self.pl_loss + 0.5 * self.value_loss - self.entropy_bonus * self.entropy

      inc_step = self.global_step.assign_add(tf.shape(self.observations)[0])

      self.gradients = self.optimizer.compute_gradients(self.loss)
      self.clipped_gradients = [(tf.clip_by_norm(grad, self.max_gradient), var)
                                  for grad, var in self.gradients]

      self.grad_norm = tf.global_norm([grad for grad, var in self.gradients])
      self.var_norm = tf.global_norm(tf.trainable_variables())

      self.train_op = tf.group(
            self.optimizer.apply_gradients(self.clipped_gradients), inc_step)

  def create_summaries(self):
    batch_size = tf.to_float(tf.shape(self.observations)[0])
    self.policy_loss_summary = tf.summary.scalar("loss/policy_loss", self.pl_loss / batch_size)
    self.entropy_loss_summary = tf.summary.scalar("loss/entropy_loss", self.entropy / batch_size)
    self.value_loss_summary  = tf.summary.scalar("loss/value_loss", self.value_loss / batch_size)
    self.total_loss_summary = tf.summary.scalar("loss/total_loss", self.loss / batch_size)
    self.grad_norm_summary = tf.summary.scalar("loss/grad_norm", self.grad_norm)
    self.var_norm_summary = tf.summary.scalar("loss/var_norm", self.var_norm)

  def merge_summaries(self):
    self.summarize = tf.summary.merge([self.policy_loss_summary,
                                      self.entropy_loss_summary,
                                      self.value_loss_summary,
                                      self.total_loss_summary,
                                      self.grad_norm_summary,
                                      self.var_norm_summary])

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

  def sampleAction(self, observations, init_state):
    probs, final_state, value = self.session.run(
                [self.probs, self.final_state, self.value],
                {self.observations: observations, self.init_state: init_state})
    return np.random.choice(self.num_actions, p=probs[0]), final_state, value

  def compute_action_probabilities(self, observations, init_state):
    return self.session.run(self.probs, {self.observations: observations,
                                         self.init_state: init_state})

  def update_parameters(self, observations, actions, returns, advantages,
                        init_state):
    self.summary_counter += 1
    write_summary = self.summary_counter == self.summary_every
    _, summary, global_step = self.session.run([self.train_op,
                                   self.summarize if write_summary else self.no_op,
                                   self.global_step],
                                  {self.observations: observations,
                                   self.actions: actions,
                                   self.returns: returns,
                                   self.advantages: advantages,
                                   self.init_state: init_state})

    if write_summary:
        self.summary_writer.add_summary(summary, global_step)
        self.saver.save(self.session, self.save_path, global_step=self.global_step)
        self.summary_counter = 0
