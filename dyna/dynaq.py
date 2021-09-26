from maze import BlockingMaze
import numpy as np
import matplotlib.pyplot as plt

class DynaAgent:
	def __init__(self, epsilon=0.3, lr=0.1, n_steps=5, episodes=1, with_model=True):
		self.maze = BlockingMaze()
		self.state = self.maze.state
		self.actions = self.maze.actions_index
		
		self.state_actions = []
		self.epsilon = epsilon
		self.lr = lr

		self.steps = n_steps
		self.episodes = episodes
		self.steps_per_episode = []

		self.Q_values = np.zeros((self.maze.rows, self.maze.columns, len(self.actions)))

		self.model = {}
		self.with_model = with_model


	def choose_action(self):
		if np.random.binomial(1, self.epsilon):
			action = np.random.choice(self.actions)
		else:
			values = self.Q_values[self.state[0], self.state[1], :]
			action = np.random.choice([action_ for action_, value_ in enumerate(values) if value_ == np.max(values)])

		return action

	def reset(self):
		self.maze = BlockingMaze()
		self.state = self.maze.state
		self.state_actions = []

	def play(self):
		self.steps_per_episode = []

		for ep in range(self.episodes):
			while not self.maze.end:

				action = self.choose_action()
				self.state_actions.append((self.state, action))

				next_state, reward = self.maze.step(action)
				next_state_values = self.Q_values[next_state[0], next_state[1], :]
				next_state_max_action = np.random.choice([action_ for action_, value_ in enumerate(next_state_values) if value_ == np.max(next_state_values)])
				
				self.Q_values[self.state[0], self.state[1], action] += \
						self.lr * (reward + self.Q_values[next_state[0], next_state[1], next_state_max_action] -
							self.Q_values[self.state[0], self.state[1], action])


				if self.state not in self.model.keys():
					self.model[self.state] = {}

				self.model[self.state][action] = (reward, next_state)
				self.state = next_state

				if self.with_model:
					for _ in range(self.steps):
						# Random previously observed state
						random_state_index = np.random.choice(range(len(self.model.keys())))
						_state = list(self.model)[random_state_index]

						# Random action previously taken in S
						random_action_index = np.random.choice(range(len(self.model[_state].keys())))
						_action = list(self.model[_state])[random_action_index]

						_reward, _next_state = self.model[_state][_action]

						_next_state_values = self.Q_values[_next_state[0], _next_state[1], :]
						_next_state_max_action = np.random.choice([action_ for action_, value_ in enumerate(_next_state_values) if value_ == np.max(_next_state_values)])

						self.Q_values[_state[0], _state[1], _action] += \
							self.lr * (_reward + self.Q_values[_next_state[0], _next_state[1], _next_state_max_action] - \
							self.Q_values[_state[0], _state[1], _action])


			
			if ep % 10 == 0:
				print('episode : ', ep)
			self.steps_per_episode.append(len(self.state_actions))
			self.reset()



if __name__ == "__main__":
    N_EPISODES = 50
    # comparison
    agent = DynaAgent(n_steps=0, episodes=N_EPISODES)
    agent.play()

    steps_episode_0 = agent.steps_per_episode

    agent = DynaAgent(n_steps=5, episodes=N_EPISODES)
    agent.play()

    steps_episode_5 = agent.steps_per_episode

    agent = DynaAgent(n_steps=50, episodes=N_EPISODES)
    agent.play()

    steps_episode_50 = agent.steps_per_episode

    agent = DynaAgent(n_steps=100, episodes=N_EPISODES)
    agent.play()

    steps_episode_100 = agent.steps_per_episode

    plt.figure(figsize=[10, 6])

    plt.ylim(0, 2000)
    plt.plot(range(N_EPISODES), steps_episode_0, label="step=0")
    plt.plot(range(N_EPISODES), steps_episode_5, label="step=5")
    plt.plot(range(N_EPISODES), steps_episode_50, label="step=50")
    plt.plot(range(N_EPISODES), steps_episode_100, label="step=100")

    plt.legend()

    plt.show()