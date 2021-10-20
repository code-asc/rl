from maze import BlockingMaze
import numpy as np

class DynaAgentPlus:
	def __init__(self, epsilon=0.3, lr=0.9, n_steps=5, episodes=1, kappa=1e-4, with_model=True, enable_change_env=False, enable_after=0):
		self.maze = BlockingMaze()
		self.state = self.maze.state
		self.actions = self.maze.actions_index
		
		self.state_actions = []
		self.epsilon = epsilon
		self.lr = lr
		self.time = 0
		self.kappa = kappa

		self.steps = n_steps
		self.episodes = episodes
		self.steps_per_episode = []

		self.Q_values = np.zeros((self.maze.rows, self.maze.columns, len(self.actions)))

		self.model = {}
		self.with_model = with_model
		self.enable_after = enable_after
		self.enable_change_env = enable_change_env


	def choose_action(self):
		if np.random.binomial(1, self.epsilon):
			action = np.random.choice(self.actions)
		else:
			values = self.Q_values[self.state[0], self.state[1], :]
			action = np.random.choice([action_ for action_, value_ in enumerate(values) if value_ == np.max(values)])

		return action

	def updateModel(self, state, next_state, action, reward):
		if state not in self.model.keys():
			self.model[state] = {}

		for a in self.actions:
			if a != action:
				self.model[state][a] = (0, state, 1)

		self.model[state][action] = (reward, next_state, self.time)


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


				self.updateModel(self.state, next_state, action, reward)
				self.state = next_state
				self.time += 1
				if self.with_model:
					for _ in range(self.steps):
						# Random previously observed state
						random_state_index = np.random.choice(range(len(self.model.keys())))
						_state = list(self.model)[random_state_index]

						# Random action previously taken in S
						random_action_index = np.random.choice(range(len(self.model[_state].keys())))
						_action = list(self.model[_state])[random_action_index]

						_reward, _next_state, _time = self.model[_state][_action]

						_reward += self.kappa * np.sqrt(self.time - _time)

						_next_state_values = self.Q_values[_next_state[0], _next_state[1], :]
						_next_state_max_action = np.random.choice([action_ for action_, value_ in enumerate(_next_state_values) if value_ == np.max(_next_state_values)])

						self.Q_values[_state[0], _state[1], _action] += \
							self.lr * (_reward + self.Q_values[_next_state[0], _next_state[1], _next_state_max_action] - \
							self.Q_values[_state[0], _state[1], _action])


			
			if ep % 10 == 0:
				print('dynaq+ episode : ', ep)
			self.steps_per_episode.append(len(self.state_actions))
			self.reset()

			if self.enable_change_env:
				if ep == self.enable_after:
					self.maze.change_env()


