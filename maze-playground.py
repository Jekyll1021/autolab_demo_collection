import numpy as np
"""created a maze environment for agents coded in agents.py"""

N = (1, 0)
S = (-1, 0)
E = (0, 1)
W = (0, -1)

allActions = [N, E, S, W]

maze = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1],
			  	 [1, 0, 0, 1, 0, 0, 0, 0, 1],
			  	 [1, 0, 0, 1, 0, 0, 1, 0, 1],
			  	 [1, 0, 0, 1, 0, 0, 1, 0, 1],
			  	 [1, 0, 0, 1, 0, 1, 1, 0, 1],
			  	 [1, 0, 0, 0, 0, 0, 1, 0, 1],
			  	 [1, 1, 1, 1, 1, 1, 1, 0, 1],
			  	 [1, 0, 0, 0, 0, 0, 0, 0, 1],
			  	 [1, 1, 1, 1, 1, 1, 1, 1, 1]])
"""0 = blank and 1 = wall."""

class Maze:
	def __init__(self, Maze = maze, Actions = allActions):
		self.maze = Maze
		self.actions = Actions

	def legal_moves(self, pos):
		legal = []
		for action in allActions:
			if ((pos[0] + action[0]) >= 0) and ((pos[0] + action[0])) < self.maze.shape[0]):
				if ((pos[1] + action[1]) >= 0) and ((pos[1] + action[1])) < self.maze.shape[1]):
					if maze[(pos[0] + action[0])][(pos[1] + action[1])] != 1:
						legal.append(action)
		return legal