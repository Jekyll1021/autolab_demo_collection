import numpy as np
import random
from maze_playground import *

default_maze = Maze()
N = (1, 0)
S = (-1, 0)
E = (0, 1)
W = (0, -1)
dis_factor = 0.9
dic = {0: {N: 0, S: 0, E: 0, W: 0}, 1: {N: 0, S: 0, E: 0, W: 0}} # shared q value table for L_agent
init_pos = (4, 1)
maze_to_7_1 = {} # shared q value table for To_7_1_agent

maze_collection = {}

class Agent:
	"""a template of MDP agent"""
	def __init__(self, init_state, maze = default_maze):
		self.state = init_state
		self.maze = maze
		if self.maze.maze[init_state[0]][init_state[1]] == 1:
			raise ValueError("initial state in walls")

	def get_action(self):
		return self.maze.legal_moves(self.state)

	def take_action(self, action):
		if action in self.maze.legal_moves(self.state):
			self.state = (self.state[0] + action[0], self.state[1] + action[1])
		else:
			raise ValueError("moving in walls")

	def get_reward(self, action):
		pass

	def execute(self):
		pass

class L_agent(Agent):
	"""an agent that learns a sequence of action that first go down and then right"""
	def __init__(self, init_state= (4, 1), q_dic= dic):
		Agent.__init__(self, init_state)
		self.counter = 0
		self.q_dic = q_dic

	def get_action(self):
		val = -1000
		action = None
		for key in self.q_dic[self.counter].keys():
			if self.q_dic[self.counter][key] >= val:
				action = key
				val = self.q_dic[self.counter][key]
		if val == 0:
			action = random.choice(list(self.q_dic[self.counter].keys()))
		return action

	def get_reward(self, action):
		if self.counter == 0 and action == (-1, 0):
			return 1
		elif self.counter == 1 and action == (0, 1):
			return 1
		else: return 0

	def take_action(self, action):
		reward = self.get_reward(action)
		self.q_dic[self.counter][action] = reward
		Agent.take_action(self, action)
		if reward == 1:
			self.counter += 1
			if self.counter != 2:
				self.q_dic[self.counter][action] += dis_factor * max(self.q_dic[self.counter].values())

	def learn(self, i):
		for num in range(0, i):
			while self.counter != 2:
				action = self.get_action()
				self.take_action(action)
			self.reset()

	def reset(self):
		self.counter = 0
		self.state = init_pos

	def execute(self):
		while self.counter != 2:
			action = self.get_action()
			self.take_action(action)
		return self.state


class To7_1_Agent(Agent):
	"""an agent that learns the sequence that go to maze position 7,1"""
	"""this agent learns this sequence in form of solving an MDP problem."""
	def __init__(self, init_state= (4, 1), q_dic= maze_to_7_1):
		Agent.__init__(self, init_state)
		self.q_dic = maze_to_7_1

	def get_action(self):
		val = -1000
		action = None
		for key in Agent.get_action(self):
			temp_score = val
			if self.state not in self.q_dic.keys():
				temp_score = 0
			elif key in self.q_dic[self.state].keys():
				temp_score = self.q_dic[self.state][key]
			else:
				temp_score = 0
			if temp_score >= val:
				action = key
				val = temp_score
		if val == 0:
			action = random.choice(Agent.get_action(self))
		return action

	def get_reward(self, action):
		next_pos = (self.state[0] + action[0], self.state[1] + action[1])
		if next_pos == (7, 1):
			return 1
		else: return 0

	def take_action(self, action):
		reward = self.get_reward(action)
		next_pos = (self.state[0] + action[0], self.state[1] + action[1])
		if next_pos in self.q_dic.keys():
				self.q_dic[self.state] = {}
				self.q_dic[self.state][action] = reward + dis_factor * max(self.q_dic[next_pos].values())
		else:
			if self.state in self.q_dic.keys():
				self.q_dic[self.state][action] = reward
			else:
				self.q_dic[self.state] = {}
				self.q_dic[self.state][action] = reward
		Agent.take_action(self, action)

	def learn(self, i):
		for num in range(0, i):
			while self.state != (7, 1):
				action = self.get_action()
				self.take_action(action)
			self.reset()

	def reset(self):
		self.state = (1, 1)

	def execute(self):
		while self.state != (7, 1):
			action = self.get_action()
			self.take_action(action)
		return self.state

class Combined_Agent(Agent):
	"""an agent that first goes L than go to 7, 1."""
	"""this agent assumes action takes sequences."""
	def __init__(self, init_state= (4, 1)):
		Agent.__init__(self, init_state)
		self.counter = 0
		self.pre_L_agent = L_agent()
		self.pre_71_agent = To7_1_Agent()

	def pre_train(self):
		self.pre_L_agent.learn(10)
		self.pre_71_agent.learn(100)
	
	def take_action(self):
		if self.counter == 0:
			execute_L_agent = L_agent()
			new_pos = execute_L_agent.execute()
			self.state = new_pos
			if new_pos != (2, 2):
				raise ValueError("the previous action has not been completed correctly!")
			self.counter += 1
		elif self.counter == 1:
			execute_71_agent = To7_1_Agent()
			new_pos = execute_71_agent.execute()
			self.state = new_pos
			if new_pos != (7, 1):
				raise ValueError("the previous action has not been completed correctly!")
			self.counter += 1

########################################################
# experiments on arbitrary consecutive sequences below #
########################################################

class moveAgent(Agent):
	"""an agent that learns the sequence that go to a given maze position"""
	"""this agent learns this sequence in form of solving an MDP problem."""
	def __init__(self, goal, init_state= (1, 1), maze=default_maze):
		Agent.__init__(self, init_state, maze)
		self.goal = goal
		if self.maze.maze[goal[0]][goal[1]] == 1:
			raise ValueError("goal state in walls")
		if goal not in maze_collection.keys():
			maze_collection[goal] = {}
		self.q_dic = maze_collection[goal]
		self.maze = maze

	def get_action(self):
		val = -1000
		action = None
		for key in Agent.get_action(self):
			temp_score = val
			if self.state not in self.q_dic.keys():
				temp_score = 0
			elif key in self.q_dic[self.state].keys():
				temp_score = self.q_dic[self.state][key]
			else:
				temp_score = 0
			if temp_score >= val:
				action = key
				val = temp_score
		if val == 0:
			action = random.choice(Agent.get_action(self))
		return action

	def get_reward(self, action):
		next_pos = (self.state[0] + action[0], self.state[1] + action[1])
		if next_pos == self.goal:
			return 1
		else: return 0

	def take_action(self, action):
		reward = self.get_reward(action)
		next_pos = (self.state[0] + action[0], self.state[1] + action[1])
		if next_pos in self.q_dic.keys():
				self.q_dic[self.state] = {}
				self.q_dic[self.state][action] = reward + dis_factor * max(self.q_dic[next_pos].values())
		else:
			if self.state in self.q_dic.keys():
				self.q_dic[self.state][action] = reward
			else:
				self.q_dic[self.state] = {}
				self.q_dic[self.state][action] = reward
		Agent.take_action(self, action)

	def learn(self, i):
		for num in range(0, i):
			while self.state != self.goal:
				action = self.get_action()
				self.take_action(action)
			self.reset()

	def reset(self):
		self.state = (1, 1)

	def execute(self, curr_pos):
		self.state = curr_pos
		while self.state != self.goal:
			action = self.get_action()
			self.take_action(action)
		return self.state

class sequenceAgent(Agent):
	"""an agent that takes in a series of agents and execute their specified actions in order"""
	def __init__(self, agents_list, init_state= (1, 1), maze=default_maze):
		Agent.__init__(self, init_state, maze)
		self.agents = agents_list
		for agent in self.agents:
			if not np.array_equal(agent.maze.maze, self.maze.maze):
				raise ValueError("cannot execute in current environment!")
		self.counter = 0
	
	def pre_train(self):
		for agent in self.agents:
			trainable = getattr(agent, "learn", None)
			if callable(trainable):
				agent.learn(100)
	
	def take_action(self):
		for agent in self.agents:
			new_pos = agent.execute(self.state)
			self.state = new_pos
			if new_pos != agent.goal:
				raise ValueError("the previous action has not been completed correctly!")
			self.counter += 1
		if self.counter != len(self.agents):
			raise ValueError("not all action has been completed correctly!")


