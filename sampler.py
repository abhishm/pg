import tensorflow as tf
import numpy as np
import scipy.signal

class Sampler(object):
    def __init__(self,
                 policy,
                 env,
                 gru_unit_size=256,
                 num_step=20,
                 gamma=0.99,
                 lambda_=1.00,
                 summary_writer=None):
        self.policy = policy
        self.env = env
        self.gru_unit_size = gru_unit_size
        self.num_step = num_step
        self.gamma = gamma
        self.lambda_ = lambda_
        self.summary_writer = summary_writer
        self.observation = self.env.reset()
        self.init_state = np.zeros((1, self.gru_unit_size))

    def discounted_x(self, x, gamma):
        return scipy.signal.lfilter([1], [1, -gamma], x[::-1])[::-1]


    def n_step_returns(self, rewards, values, final_value):
        """
        returns the value estime and GAE estimate
        """
        reward_plus = np.array(rewards + [final_value])
        value_plus = np.array(values + [final_value])

        n_step_pred = self.discounted_x(reward_plus, self.gamma)[:-1]

        delta_t = rewards + self.gamma * value_plus[1:] - value_plus[:-1]
        # this formula for the advantage comes "Generalized Advantage Estimation":
        # https://arxiv.org/abs/1506.02438
        adv = self.discounted_x(delta_t, self.gamma * self.lambda_)
        return n_step_pred, adv


    def collect_batch(self, render=False):
        observations, actions, rewards, values = [], [], [], []
        init_state = self.init_state.copy()

        for t in range(self.num_step):
            if render:
                self.env.render()
            action, next_state, value = self.policy.sampleAction(
                                        self.observation[np.newaxis, :],
                                        self.init_state)
            next_ob, reward, done, info = self.env.step(action)
            # appending the experience
            observations.append(self.observation)
            actions.append(action)
            rewards.append(reward)
            values.append(value[0, 0])
            # going to next state
            self.observation = next_ob
            self.init_state = next_state

            if info:
                summary = tf.Summary()
                global_step = self.policy.session.run(self.policy.global_step)
                for k, v in info.items():
                    summary.value.add(tag=k, simple_value=float(v))
                self.summary_writer.add_summary(summary, global_step)
                self.summary_writer.flush()

            if done:
                self.observation = self.env.reset()
                self.init_state = np.zeros((1, self.gru_unit_size))
                break

        if done:
            final_value = 0.0
        else:
            _, _, final_value = self.policy.sampleAction(
                                        self.observation[np.newaxis, :],
                                        self.init_state)
            final_value = final_value[0, 0]

        returns, advantages = self.n_step_returns(rewards, values, final_value)

        episode = dict(
                    observations = np.array(observations),
                    actions = np.array(actions),
                    returns = returns,
                    advantages = advantages,
                    init_state = init_state,
                    )
        return episode

    def samples(self):
        return self.collect_batch()
