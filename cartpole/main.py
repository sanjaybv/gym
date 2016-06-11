import gym
import time

class RandomAgent():
    def __init__(self, action_space):
        self.action_space = action_space
    def act(self, observation, reward, done):
        return self.action_space.sample()

def main():
    env = gym.make('CartPole-v0')
    #env.monitor.start('/tmp/cartpole-experiment-1', force=True, seed=0)

    agent = RandomAgent(env.action_space)

    episode_count = 100
    max_steps = 200
    reward = 0
    done = False

    for i_episode in range(episode_count):
        ob = env.reset()
        for t in range(max_steps):
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            print('reward:', reward)
            if done:
                break
            time.sleep(1)

    #env.monitor.close()

if __name__ == '__main__':
    main()
