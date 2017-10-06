import logging
import math
import gym
from gym import spaces
# from gym.utils import seeding
import numpy as np

import pygame
import numpy as np
import pickle

import Box2D  
from Box2D.b2 import (world, polygonShape, circleShape, staticBody, dynamicBody, kinematicBody)

PPM = 20.0  # pixels per meter
TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480

# GOAL_POS = (5, 2)
# GOAL_ANGLE = 0
INTERVAL = 0.1


class TestEnv(gym.Env):
	def __init__(self):
		self.action_space = spaces.Box(low=-10, high=10, shape=(4, ))
		self.observation_space = spaces.Box(low=-25, high=25, shape=(9, ))
		self.done = False
		self.goal_pos = (np.random.sample() * 20, np.random.sample() * 10)
		self.goal_angle = np.random.sample() * 3
		self.timesteps = 1000
		self.origin_dist_pos = 0
		self.origin_dist_angle = 0
		self._createWorld()

	def compute_rewards(self):
		pos_r = (self.origin_dist_pos - math.sqrt((self.box.position[0] - self.goal_pos[0])**2 + (self.box.position[1] - self.goal_pos[1])**2))/(self.origin_dist_pos + 1e-2)
		angle_r = (abs(self.box.angle - self.goal_angle) - self.origin_dist_angle)/(self.origin_dist_angle + 1e-2)
		return 0.1 * pos_r + 0.01 * angle_r

	def _createWorld(self):
		# --- pybox2d world setup ---
		# Create the world
		self.world = world(gravity=(0, 0), doSleep=True)

		self.rod = self.world.CreateDynamicBody(position=(np.random.sample() * 20, np.random.sample() * 10), allowSleep=False, userData='rod')
		rodfix = self.rod.CreatePolygonFixture(density=1, box=(0.25, 1), friction=0.0)

		# a target box to move around
		self.box = self.world.CreateDynamicBody(position=(np.random.sample() * 20, np.random.sample() * 10), allowSleep=False, userData='target')
		boxfix = self.box.CreatePolygonFixture(density=1, box=(1,1), friction=0.5)
		self.origin_dist_pos = math.sqrt((self.box.position[0] - self.goal_pos[0])**2 + (self.box.position[1] - self.goal_pos[1])**2)
		self.origin_dist_angle = abs(self.box.angle - self.goal_angle)

		colors = {
		    'target': (200, 200, 200, 255),
		    'rod': (100, 100, 100, 100),
		}

	def step(self, action):
		"""assuming action to be a tuple, (vector x, vector y, point x, point y)"""
		obs = None
		reward = 0
		if not self.done:

			# vector, point = (action[0], action[1]), (action[2], action[3])
			# print(vector)

			f = self.rod.GetWorldVector(localVector=(float(action[0]), float(action[1])))
			p = self.rod.GetWorldPoint(localPoint=(float(action[2]), float(action[3])))
			self.rod.ApplyForce(f, p, True)

			obs = np.array([self.box.position[0], self.box.position[1], self.box.angle, self.rod.position[0], self.rod.position[1], self.rod.angle, self.goal_pos[0], self.goal_pos[1], self.goal_angle])

			self.world.Step(TIME_STEP, 10, 10)
			self.timesteps -= 1
		
			self.done = bool(
				# self.screen.get_rect().contains(self.box.get_rect()) or \
				# self.screen.get_rect().contains(self.rod.get_rect()) or \
				self.timesteps == 0 or
				(abs(self.box.position[0] - self.goal_pos[0]) <= INTERVAL and \
				abs(self.box.position[1] - self.goal_pos[1]) <= INTERVAL and \
				abs(self.box.angle - self.goal_angle) <= INTERVAL))
			
			reward = self.compute_rewards()
			if abs(self.box.position[0] - self.goal_pos[0]) <= INTERVAL and \
				abs(self.box.position[1] - self.goal_pos[1]) <= INTERVAL and abs(self.box.angle - self.goal_angle) <= INTERVAL:
				reward += 1000
			
		return obs, reward, self.done, {}

	def reset(self):
		self.__init__()
		return np.array([self.box.position[0], self.box.position[1], self.box.angle, self.rod.position[0], self.rod.position[1], self.rod.angle, self.goal_pos[0], self.goal_pos[1], self.goal_angle])

	def close(self):
		return
