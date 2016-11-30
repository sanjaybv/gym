import numpy as np

class HillClimbing:
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
        self.observation_length = len(self.observation_space.low)
        # random initial parameters
        rand_nums = np.random.rand(self.observation_length) * 2 - 1
        self.theta = rand_nums / np.linalg.norm(rand_nums)

        # eps - range of noise
        self.eps = 0.5
        self.cum_reward = 0.0
        self.best_reward = 0.0
        self.best_theta = self.theta

    def act(self, observation, reward, done):
        self.cum_reward += reward
        action_value = np.array(observation).dot(self.theta)
        return 1 if action_value >= 0 else 0

    def set_theta(self, theta):
        self.theta = theta

    def get_cum_reward(self):
        return self.cum_reward
    
    def start_episode(self):
        self.cum_reward = 0
        # add noise to theta
        rand_nums = np.random.rand(self.observation_length)
        self.theta = self.best_theta + rand_nums * self.eps
        self.theta /= np.linalg.norm(self.theta)
        print 'theta:', self.theta,

    def end_episode(self):
        print("reward:", self.cum_reward)
        if self.cum_reward > self.best_reward:
            self.best_reward = self.cum_reward
            self.best_theta = self.theta

    def get_best_params(self):
        return self.best_theta, self.best_reward

