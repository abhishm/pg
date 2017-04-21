import numpy as np
import json
import os
import gym
import universe
import tensorflow as tf
from envs import create_env
from tqdm import tqdm
import matplotlib.pyplot as plt
from pg_reinforce import PolicyGradientREINFORCE
from sampler import Sampler

config = json.load(open("configuration.json"))
train = config["train"]

# env configuration
if "Pong" in config["env_name"]:
    env = create_env(config["env_name"], 0, 1)
else:
    env = gym.make(config["env_name"])

state_dim = np.prod(env.observation_space.shape)
num_actions = env.action_space.n

# RNN configuration
global_step = tf.Variable(0, name="global_step", trainable=False)
gru_unit_size = config["gru_unit_size"]
num_step = config["num_step"]
max_step = config["max_step"]
batch_size = config["batch_size"]
max_gradient = config["max_gradient_norm"]
loss_function = config["loss_function"]
entropy_bonus = config["entropy_bonus"]
num_layers = config["num_layers"]
learning_adaptive = config["learning"]["learning_adaptive"]
if learning_adaptive:
    learning_rate = tf.train.exponential_decay(
                      config["learning"]["learning_rate"],
                      global_step,
                      config["learning"]["decay_steps"],
                      config["learning"]["decay_rate"],
                      staircase=True)
else:
    learning_rate = config["learning"]["learning_rate"]

#tensorflow
sess = tf.Session()
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

# checkpointing
base_file = "_".join([i + "-" + str(v) for i, v in sorted(config.items())
                                    if i not in ["train", "learning"]])
os.makedirs(base_file, exist_ok=True)
json.dump(config, open(base_file + "/configuration.json", "w"))
writer = tf.summary.FileWriter(base_file + "/summary/")
save_path= base_file + '/models/'
os.makedirs(save_path, exist_ok=True)

# iterations
num_itr = config["num_itr"]


def show_image(array):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.plot(array)
    plt.title("Reward Progress")
    plt.grid()
    plt.savefig(base_file + "/summary/" + "reward.png")
    plt.show()


def policy_network(states, init_states, seq_len):
    """ define policy neural network """
    with tf.variable_scope("rnn"):
        gru_cell = tf.contrib.rnn.GRUCell(gru_unit_size)
        gru_cells = tf.contrib.rnn.MultiRNNCell([gru_cell] * num_layers)
        output, final_state = tf.nn.dynamic_rnn(gru_cells, states,
                initial_state=init_states, sequence_length=seq_len)
        output = tf.nn.relu(output)

    with tf.variable_scope("softmax"):
        w_softmax = tf.get_variable("w_softmax", shape=[gru_unit_size, num_actions],
            initializer=tf.contrib.layers.xavier_initializer())
        b_softmax = tf.get_variable("b_softmax", shape=[num_actions],
            initializer=tf.constant_initializer(0))

    logit = (tf.matmul(tf.reshape(output, [-1, gru_unit_size]), w_softmax)
             + b_softmax)

    with tf.variable_scope("value_function"):
        w_value = tf.get_variable("w_value", shape=[gru_unit_size, 1],
            initializer=tf.contrib.layers.xavier_initializer())
        b_value = tf.get_variable("b_value", shape=[1],
            initializer=tf.constant_initializer(0))

    value = (tf.matmul(tf.reshape(output, [-1, gru_unit_size]), w_value)
             + b_value)
    return logit, final_state, value


pg_reinforce = PolicyGradientREINFORCE(sess,
                                       optimizer,
                                       policy_network,
                                       state_dim,
                                       num_actions,
                                       gru_unit_size,
                                       num_step,
                                       num_layers,
                                       save_path + env.spec.id,
                                       global_step,
                                       max_gradient=max_gradient,
                                       entropy_bonus=entropy_bonus,
                                       summary_writer=writer,
                                       summary_every=10,
                                       loss_function=loss_function)

sampler = Sampler(pg_reinforce, env, gru_unit_size, num_step, num_layers,
               max_step, batch_size)

# reward = []
# for _ in tqdm(range(num_itr)):
#     if train:
#         batch = sampler.samples()
#         # progress tracking
#         r = batch["rewards"].sum()
#         reward.append(r)
#         # updates
#         pg_reinforce.update_parameters(batch["states"], batch["actions"],
#                                         batch["monte_carlo_returns"],
#                                         batch["init_states"], batch["seq_len"],
#                                         r)
#     else:
#         episode = sampler.collect_one_episode(render=True)
#         print("reward is {0}".format(np.sum(episode["rewards"])))
#
# show_image(reward)
