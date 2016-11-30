import gym
import time
import numpy as np
from hill_climbing import HillClimbing

def main():
    # initialize the enironment
    env = gym.make('Acrobot-v0')
    env.monitor.start('acrobot', force=True, seed=0)

    # initialize agent
    agent = HillClimbing(env.action_space, env.observation_space)

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
