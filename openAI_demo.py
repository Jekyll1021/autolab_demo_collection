import gym
from gym import wrappers
import random
env = gym.make('CartPole-v0')
# env = wrappers.Monitor(env, '/tmp/cartpole-experiment-1', force=True)
# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample())
q_dict_colle = {}

class Agent:
	"""a template of MDP agent"""
	def __init__(self, init_state, environment = env):
		self.state = init_state
		self.environment = env
		self.time = 0
		self.next_state = None
		
	def get_action(self):
		return self.environment.action_space.sample()

	def take_action(self, action):
		self.next_state = self.environment.step(action)
		self.state = self.next_state[0]
		self.time += 1
		return self.next_state

	def get_reward(self):
		if self.next_state:
			return self.next_state[1]
		else:
			return 0

	def execute(self):
		pass

	def learn(self, i):
		pass

class OcsiAgent(Agent):
	"""an agent that alternatively execute left and right action."""
	def __init__(self, init_state, goal, environment = env):
		Agent.__init__(self, init_state, environment)
		self.goal = goal
		self.start = 0
		if self.state[2] >= 0:
			self.start = 1

	def get_action(self):
		return (self.start + self.time % 2) % 2

	def execute(self):
		# self.next_state = None
		# self.state = self.environment.reset()
		for i in range(0, self.goal):
			self.environment.render()
			self.take_action(self.get_action())
		return self.state

class BalanceAgent(Agent):
	def __init__(self, init_state, goal, environment = env):
		Agent.__init__(self, init_state, environment)
		self.goal = goal
		self.start = 0
		if self.state[2] >= 0:
			self.start = 1

	def get_action(self):
		action = 0
		if self.state[2] >= 0:
			action = 1
		return action

	def execute(self):
		# self.environment = wrappers.Monitor(self.environment, '/tmp/cartpole-experiment-1', force=True)
		# self.next_state = None
		# self.state = self.environment.reset()
		for i in range(0, self.goal):
			self.environment.render()
			self.take_action(self.get_action())
		return self.state

class SequenceAgent(Agent):
	"""an agent that takes in a series of agents and execute their specified actions in order"""
	def __init__(self, agents_list, init_state, environment = env):
		Agent.__init__(self, init_state, environment)
		self.agents = agents_list
		self.goal = 0
		for agent in self.agents:
			if self.environment != agent.environment:
				raise ValueError("cannot execute in current environment!")
			self.goal += agent.goal


	def execute(self):
		# self.environment = wrappers.Monitor(self.environment, '/tmp/cartpole-experiment-1', force=True)
		# self.next_state = None
		# self.state = self.environment.reset()
		for agent in self.agents:
			agent.state = self.state
			self.state = agent.execute()
			self.time += agent.goal


# class GoodAgent(Agent):
# 	"""an agent that tries to balance the pole"""
# 	def __init__(self, init_state, goal = 200, environment = env):
# 		Agent.__init__(self, init_state, environment)
# 		self.name = "GoodAgent"
# 		if self.name not in q_dict_colle.keys():
# 			q_dict_colle[self.name] = {}
# 		self.q_dic = q_dict_colle[self.name]
# 		self.goal = goal

# 	def get_action(self, randomness = 0.5):
# 		val = -1000
# 		action = None
# 		state = tuple(self.state)
# 		for s in self.q_dic.keys():
# 			if abs(s[0] - self.state[0]) < 0.01 \
# 			and abs(s[1] - self.state[1]) < 0.01:
# 				state = s
# 				break
# 		for key in [0, 1]:
# 			temp_score = val
# 			if state not in self.q_dic.keys():
# 				temp_score = 0
# 			elif key in self.q_dic[state].keys():
# 				temp_score = self.q_dic[state][key]
# 			else:
# 				temp_score = 0
# 			if temp_score >= val:
# 				action = key
# 				val = temp_score
# 		if val == 0:
# 			action = self.environment.action_space.sample()
# 		random_or_not = random.uniform(0, 1)
# 		if random_or_not < randomness:
# 			action = self.environment.action_space.sample()
# 		return action

# 	def take_action(self, action):
# 		state = tuple(self.state)
# 		for s in self.q_dic.keys():
# 			if abs(s[0] - self.state[0]) < 0.01 \
# 			and abs(s[1] - self.state[1]) < 0.01:
# 				state = s
# 				break
# 		next_state = Agent.take_action(self, action)
# 		if state not in self.q_dic.keys():
# 			self.q_dic[state] = {}
# 		self.q_dic[state][action] = next_state[1]
# 		next = tuple(next_state[0])
# 		for s in self.q_dic.keys():
# 			if abs(s[0] - self.state[0]) < 0.01 \
# 			and abs(s[1] - self.state[1]) < 0.01:
# 				next = s
# 				break
# 		if next in self.q_dic.keys():
# 			self.q_dic[state][action] = self.q_dic[state][action] + 0.9 * max(self.q_dic[next].values())
# 		return next_state

# 	def learn(self, i):
# 		for trails in range(0, i):
# 			while not self.next_state or self.next_state[2] != True:
# 				self.take_action(self.get_action())
# 			self.next_state = None
# 			self.state = self.environment.reset()

# 	def execute(self):
# 		self.environment = wrappers.Monitor(self.environment, '/tmp/cartpole-experiment-1', force=True)
# 		self.next_state = None
# 		self.state = self.environment.reset()
# 		for i in range(0, self.goal):
# 			self.environment.render()
# 			self.take_action(self.get_action(0.0))
	


