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
from model import policy_network

config = json.load(open("configuration.json"))
train = config["train"]

# env configuration
if "Pong" in config["env_name"]:
    env = create_env(config["env_name"], 0, 1)
else:
    env = gym.make(config["env_name"])

observation_space_dim = env.observation_space.shape
num_actions = env.action_space.n

# RNN configuration
global_step = tf.Variable(0, name="global_step", trainable=False)
gru_unit_size = config["gru_unit_size"]
num_step = config["num_step"]
max_gradient = config["max_gradient_norm"]
loss_function = config["loss_function"]
entropy_bonus = config["entropy_bonus"]
discount = config["discount"]
lambda_ = config["lambda_"]
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



pg_reinforce = PolicyGradientREINFORCE(sess,
                                       optimizer,
                                       policy_network,
                                       observation_space_dim,
                                       num_actions,
                                       gru_unit_size,
                                       save_path + env.spec.id,
                                       global_step,
                                       max_gradient=max_gradient,
                                       entropy_bonus=entropy_bonus,
                                       summary_writer=writer,
                                       summary_every=100,
                                       loss_function=loss_function)

sampler = Sampler(pg_reinforce, env, gru_unit_size, num_step, gamma=discount,
                  lambda_=lambda_, summary_writer=writer)


for _ in tqdm(range(num_itr)):
    if train:
        batch = sampler.samples()
        # updates
        pg_reinforce.update_parameters(batch["observations"],
                                        batch["actions"],
                                        batch["returns"],
                                        batch["advantages"],
                                        batch["init_state"])
    else:
        episode = sampler.collect_batch(render=True)

show_image(reward)
