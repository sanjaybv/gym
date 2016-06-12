import gym
import time
import numpy as np

# class of the agent
class RandomAgent():
    def __init__(self, action_space):
        self.action_space = action_space
        self.theta = np.zeros(4)
        self.cum_reward = 0.0
        self.max_reward = 0.0
        self.max_theta = np.zeros(4)

    def act(self, observation, reward, done):
        self.cum_reward += reward
        action_value = np.array(observation).dot(self.theta)
        return 1 if action_value >= 0 else 0

    def set_theta(self, theta):
        self.theta = theta
    
    def start_episode(self):
        rand_nums = np.random.rand(4)
        self.theta = rand_nums / np.linalg.norm(rand_nums)

    def end_episode(self):
        if self.cum_reward > self.max_reward:
            self.max_reward = self.cum_reward
            self.max_theta = self.theta

    def get_max_params(self):
        return self.max_theta, self.max_reward

def main():
    # initialize the enironment
    env = gym.make('CartPole-v0')
    #env.monitor.start('/tmp/cartpole-experiment-1', force=True, seed=0)

    # initialize agent
    agent = RandomAgent(env.action_space)

    # set up run parameters
    episode_count = 1000
    max_steps = 200
    reward = 0
    done = False

    # run the episodes
    for i_episode in range(episode_count):
        agent.start_episode()
        ob = env.reset()
        for t in range(max_steps):
            # get action from the agent
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break
        agent.end_episode()

    theta, _ = agent.get_max_params()
    reward = 0
    agent.set_theta(theta)
    ob = env.reset()
    done = False
    while not done:
        env.render()
        action = agent.act(ob, reward, done)
        ob, reward, done, _ = env.step(action)

    #env.monitor.close()

if __name__ == '__main__':
    main()
