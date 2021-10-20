import gym
import random
from tile import *
import matplotlib.pyplot as plt
import numpy as np
from gym import wrappers
import random


class Agent:
    def __init__(self, env, alpha=0.01, gamma=0.98, epsilon=0.6, episodes=2000):
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.POSITION_UPPER_BOUND, self.VELOCITY_UPPER_BOUND = tuple(
            env.observation_space.high)
        self.POSITION_LOWER_BOUND, self.VELOCITY_LOWER_BOUND = tuple(
            env.observation_space.low)
        self.env = env
        self.tilings = 8
        self.maxtiles = 2048
        self.episodes = episodes
        self.thetas = np.zeros(self.maxtiles)
        self.hash_table = IHT(self.maxtiles)

    def chooseAction(self, observation):
        if random.random() > self.epsilon:
            return np.argmax([self.fun_approx(observation, action) for action in range(self.env.action_space.n)])
        else:
            return random.randint(0, 2)

    def fun_approx(self, observation, action):
        temp = np.matmul(self.getFeature(
        observation[0], observation[1], action), self.thetas)
        return temp

    def getFeature(self, position, velocity, action):
        indices = tiles(self.hash_table, self.tilings,
                        [self.tilings * position / (self.POSITION_UPPER_BOUND - self.POSITION_LOWER_BOUND),
                         self.tilings * velocity / (self.VELOCITY_UPPER_BOUND - self.VELOCITY_LOWER_BOUND)],
                        [action])

        features = [0] * self.maxtiles
        for index in indices:
            features[index] = 1
        return features

    def delta(self, observation, action):
        return self.getFeature(observation[0], observation[1], action)

    def run(self):
        for i_episode in range(self.episodes):
            observation = self.env.reset()
            self.env.render()
            reward_total = 0
            action = self.chooseAction(observation)
            t = 0

            while True:
                t += 1
                new_observation, reward, done, info = self.env.step(action)
                self.env.render()

                if done:
                    self.thetas += self.alpha * (reward - self.fun_approx(observation, action))\
                        * np.array(self.delta(observation, action))
                    break

                next_action = self.chooseAction(new_observation)
                q = self.fun_approx(observation, action)
                q_ = self.fun_approx(new_observation, next_action)
                self.thetas += self.alpha * (reward + self.gamma * q_ - q)\
                    * np.array(self.delta(observation, action))

                observation = new_observation
                action = next_action
            print('Episode number : {} steps : {}'.format(i_episode, t))
        self.env.close()


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 1000
    Agent(env).run()