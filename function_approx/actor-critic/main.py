import gym
import random
from tile import *
import matplotlib.pyplot as plt
import numpy as np
from gym import wrappers
import random
import math


class Agent:
    def __init__(self, env, alpha_weight=0.004, alpha_theta= 0.0002, gamma=1, episodes=2000):
        
        self.gamma = gamma
        self.alpha_weight = alpha_weight
        self.alpha_theta = alpha_theta
        self.POSITION_UPPER_BOUND, self.VELOCITY_UPPER_BOUND = tuple(
            env.observation_space.high)
        self.POSITION_LOWER_BOUND, self.VELOCITY_LOWER_BOUND = tuple(
            env.observation_space.low)
        self.env = env
        self.tilings = 8
        self.maxtiles = 2048
        self.episodes = episodes
        self.weights = np.zeros(self.maxtiles)
        self.thetas = np.zeros(self.maxtiles)
        self.hash_table = IHT(self.maxtiles)

    def choose_action(self, observation):
       
        prob = [self.softmax(observation, action) for action in range(self.env.action_space.n)]
        return random.choices(range(self.env.action_space.n), prob)[0]

    def fun_approx(self, observation, action):
        temp = np.matmul(self.get_feature_theta(
        observation[0], observation[1], action), self.weights)
        return temp

    def get_feature_theta(self, position, velocity, action):

        if action != None:
            indices = tiles(self.hash_table, self.tilings,
                        [self.tilings * position / (self.POSITION_UPPER_BOUND - self.POSITION_LOWER_BOUND),
                         self.tilings * velocity / (self.VELOCITY_UPPER_BOUND - self.VELOCITY_LOWER_BOUND)],
                        [action])
        else:
            indices = tiles(self.hash_table, self.tilings,
                        [self.tilings * position / (self.POSITION_UPPER_BOUND - self.POSITION_LOWER_BOUND),
                         self.tilings * velocity / (self.VELOCITY_UPPER_BOUND - self.VELOCITY_LOWER_BOUND)])

        features = [0] * self.maxtiles
        for index in indices:
            features[index] = 1
        return features

    def delta(self, observation, action):
        return self.get_feature_theta(observation[0], observation[1], action)

    def softmax(self, observation, action):
        denominator = 0.0
        for i in range(self.env.action_space.n):
            denominator += math.exp(np.matmul(self.get_feature_theta(observation[0], observation[1], i), self.thetas))

        return math.exp(np.matmul(self.get_feature_theta(observation[0], observation[1], action), self.thetas))/denominator


    def delta_policy(self, observation, action):
        temp = np.zeros(self.maxtiles)
        for i in range(self.env.action_space.n):
            temp += np.array(self.get_feature_theta(observation[0], observation[1], i)) * self.softmax(observation, i)

        temp_arr = np.array(self.get_feature_theta(observation[0], observation[1], action)) - temp
        return temp_arr.tolist()

    def run(self):
        R = 0
        for i_episode in range(self.episodes):
            observation = self.env.reset()
            self.env.render()
            reward_total = 0
            action = self.choose_action(observation)
            t = 0
            

            while True:
                t += 1
                new_observation, reward, done, info = self.env.step(action)
                self.env.render()

                if done:
                    error = (reward - self.fun_approx(observation, None))
                    
                    self.weights += self.alpha_weight * error\
                        * np.array(self.delta(observation, None)) * (self.gamma ** t)

                    self.thetas += self.alpha_theta * error\
                        * np.array(self.delta_policy(observation, action)) * (self.gamma ** t)
                    break

                next_action = self.choose_action(new_observation)
            
                error = (reward + self.gamma*self.fun_approx(new_observation, None) - 
                    self.fun_approx(observation, None))

                

                self.weights += self.alpha_weight * error\
                    * np.array(self.delta(observation, None)) * (self.gamma ** t)

                self.thetas += self.alpha_theta * error\
                        * np.array(self.delta_policy(observation, action)) * (self.gamma ** t)

                observation = new_observation
                action = next_action
            print('Episode number : {} steps : {}'.format(i_episode, t))
        self.env.close()


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 20000
    Agent(env).run()