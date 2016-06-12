import gym
import time
import numpy as np

# class of the agent
class HillClimbingAgent():
    def __init__(self, action_space):
        self.action_space = action_space
        # random initial parameters
        rand_nums = np.random.rand(4) * 2 - 1
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
        rand_nums = np.random.rand(4)
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

def main():
    # initialize the enironment
    env = gym.make('CartPole-v0')
    env.monitor.start('cartpole', force=True, seed=0)

    # initialize agent
    agent = HillClimbingAgent(env.action_space)

    # set up run parameters
    episode_count = 200
    max_steps = env.spec.timestep_limit
    reward = 0
    done = False

    # run the episodes
    for i_episode in range(episode_count):
        print i_episode,
        agent.start_episode()
        ob = env.reset()
        for t in range(max_steps):
            #env.render()
            # get action from the agent
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break
        agent.end_episode()

    #env.monitor.close()

if __name__ == '__main__':
    main()
