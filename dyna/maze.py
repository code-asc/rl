import numpy as np



class BlockingMaze:
	def __init__(self):
		self.rows = 6
		self.columns = 9
		self.start = (2,0)
		self.end = False
		self.goal = (0,8)
		self.blocks = [(1, 2), (2, 2), (3, 2), (0, 7), (1, 7), (2, 7), (4, 5)]
		self.state = self.start
		self.actions = ['left', 'up', 'right', 'down']
		self.actions_index = [0,1,2,3]

		self.blocking_maze = np.zeros((self.rows, self.columns))
		for block in self.blocks:
			self.blocking_maze[block] = -1


	def step(self, action):
		row, col = self.state
		reward = 0

		action = self.actions[action]

		if action == 'left':
			col -= 1

		elif action == 'right':
			col += 1

		elif action == 'up':
			row -= 1

		else:
			row += 1


		if (row >= 0 and row <= self.rows-1) and (col >= 0 and col <= self.columns-1):
			if (row,col) not in self.blocks:
				self.state = (row, col)

		if self.state == self.goal:
			reward = 1 
			self.end = True

		return self.state, reward



	def render(self):
		self.blocking_maze[self.state] = 1
		for i in range(self.rows):
			print('-------------------------------------')
			out = '| '
			for j in range(self.columns):
				if self.blocking_maze[i, j] == 1:
					token = '*'

				if self.blocking_maze[i, j] == -1:
					token = 'z'

				if self.blocking_maze[i, j] == 0:
					token = '0'

				out += token + ' | '

			print(out)

		print('-------------------------------------')

