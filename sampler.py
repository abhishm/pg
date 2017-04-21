import numpy as np

class Sampler(object):
    def __init__(self,
                 policy,
                 env,
                 gru_unit_size=16,
                 num_step=10,
                 num_layers=1,
                 max_step=2000,
                 batch_size=10000,
                 discount=0.99):
        self.policy = policy
        self.env = env
        self.gru_unit_size = gru_unit_size
        self.num_step = num_step
        self.num_layers = num_layers
        self.max_step = max_step
        self.batch_size = batch_size
        self.state = self.env.reset()
        self.discount = discount

    def compute_monte_carlo_returns(self, rewards):
        return_so_far = 0
        returns = []
        for reward in rewards[::-1]:
            return_so_far = reward + self.discount * return_so_far
            returns.append(return_so_far)
        return returns[::-1]

    def collect_one_episode(self, render=False):
        self.state = self.env.reset() # NB. remove it for Pong
        states, actions, rewards, values, dones = [], [], [], [], []
        init_states = tuple([] for _ in range(self.num_layers))
        init_state = tuple(
             [np.zeros((1, self.gru_unit_size)) for _ in range(self.num_layers)])
        for t in range(self.max_step):
            if render:
                self.env.render()
            self.state = self.preprocessing(self.state)
            action, final_state, value = self.policy.sampleAction(
                                        self.state[np.newaxis, np.newaxis, :],
                                        init_state)
            next_state, reward, done, _ = self.env.step(action)
            # appending the experience
            states.append(self.state)
            actions.append(action)
            rewards.append(reward)
            values.append(value[0, 0])
            [init_states[i].append(init_state[i][0]) for i in
                                           range(self.num_layers)]
            dones.append(done)
            # going to next state
            self.state = next_state
            init_state = final_state
            if done:
                self.state = self.env.reset()
            if reward == 0:
                break
        returns = self.compute_monte_carlo_returns(rewards)
        #returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        advantages = np.array(returns) - np.array(values)
        #advantages = ((advantages - np.mean(advantages))
        #              / (np.std(advantages) + 1e-8))
        episode = dict(
                    states = np.array(states),
                    actions = np.array(actions),
                    rewards = np.array(rewards),
                    monte_carlo_returns = np.array(returns),
                    advantages = advantages,
                    init_states = tuple(np.array(init_states[i])
                                   for i in range(self.num_layers)),
                    )
        return self.expand_episode(episode)

    def collect_one_batch(self):
        episodes = []
        len_samples = 0
        while len_samples < self.batch_size:
            episode = self.collect_one_episode()
            episodes.append(episode)
            len_samples += np.sum(episode["seq_len"])
        # prepare input
        states = np.concatenate([episode["states"] for episode in episodes])
        actions = np.concatenate([episode["actions"] for episode in episodes])
        rewards = np.concatenate([episode["rewards"] for episode in episodes])
        monte_carlo_returns = np.concatenate([episode["monte_carlo_returns"]
                                 for episode in episodes])
        advantages = np.concatenate([episode["advantages"]
                                      for episode in episodes])

        init_states = tuple(
                       np.concatenate([episode["init_states"][i]
                                       for episode in episodes])
                       for i in range(self.num_layers))
        seq_len = np.concatenate([episode["seq_len"] for episode in episodes])
        batch = dict(
                    states = states,
                    actions = actions,
                    rewards = rewards,
                    monte_carlo_returns = monte_carlo_returns,
                    advantages = advantages,
                    init_states = init_states,
                    seq_len = seq_len
                    )
        return batch

    def expand_episode(self, episode):
        episode_size = len(episode["rewards"])
        if episode_size % self.num_step:
            batch_from_episode = (episode_size // self.num_step + 1)
        else:
            batch_from_episode = (episode_size // self.num_step)

        extra_length = batch_from_episode * self.num_step - episode_size
        last_batch_size = episode_size - (batch_from_episode - 1) * self.num_step

        batched_episode = {}
        for key, value in episode.items():
            if key == "init_states":
                truncated_value = tuple(value[i][::self.num_step] for i in
                                        range(self.num_layers))
                batched_episode[key] = truncated_value
            else:
                expanded_value = np.concatenate([value, np.zeros((extra_length,) +
                                                     value.shape[1:])])
                batched_episode[key] = expanded_value.reshape((-1, self.num_step) +
                                                         value.shape[1:])

        seq_len = [self.num_step] * (batch_from_episode - 1) + [last_batch_size]
        batched_episode["seq_len"] = np.array(seq_len)
        return batched_episode

    def samples(self):
        return self.collect_one_batch()

    def preprocessing(self, image):
        """ preprocess 42x42x1 uint8 frame into 1764 (42x42) 1D float vector """
        return image.astype(np.float).ravel()
