import gym
import numpy as np
from time import sleep
import gym_windy_gridworlds
import matplotlib.pyplot as plt

# grid height
GRID_HEIGHT = 7

# grid width
GRID_WIDTH = 10

# probability for exploration
EPSILON = 0.1

# Sarsa step size
ALPHA = 0.5

# possible actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]


def episode(q_value, env):
	# Keeping track of time steps for each episode
	time = 0
	DONE = False

	# Initialize state
	state = env.reset()

	# Choose an action based on epsilon-greedy algorithm
	if np.random.binomial(1, EPSILON):
		action = np.random.choice(ACTIONS)
	else:
		values = q_value[state[0], state[1], :]
		action = np.random.choice([action_ for action_, value_ in enumerate(values) if value_ == np.max(values)])

	while not DONE:
		############
		# CODE FOR RENDER 
		env.render()
		############

		next_state, REWARD, DONE, info = env.step(action)

		if np.random.binomial(1, EPSILON):
			next_action = np.random.choice(ACTIONS)
		else:
			values = q_value[next_state[0], next_state[1], :]
			next_action = np.random.choice([action_ for action_, value_ in enumerate(values) if value_ == np.max(values)])

		# Sarsa update
		q_value[state[0], state[1], action] += \
			ALPHA * (REWARD + q_value[next_state[0], next_state[1], next_action] -
				q_value[state[0], state[1], action])
		state = next_state
		action = next_action

		time +=1

		# Final render
	env.render()

	return time


def show_steps(steps):
	steps = np.add.accumulate(steps)
	plt.plot(steps, np.arange(1, len(steps) + 1))
	plt.xlabel('Time steps')
	plt.ylabel('Episodes')
	plt.show()


def init():
	q_value = np.zeros((GRID_HEIGHT, GRID_WIDTH, len(ACTIONS)))
	episode_limit = 200
	steps = []
	env = gym.make('WindyGridWorld-v0')
	ep = 0

	while ep < episode_limit:
		steps.append(episode(q_value, env))
		ep += 1
		
	show_steps(steps)


if __name__ == '__main__':
    init()