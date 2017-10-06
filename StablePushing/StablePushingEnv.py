import logging
import math
import gym
from gym import spaces
# from gym.utils import seeding
import numpy as np

import pygame
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)
import numpy as np
import pickle

import Box2D  
from Box2D.b2 import (world, polygonShape, circleShape, staticBody, dynamicBody, kinematicBody)

PPM = 20.0  # pixels per meter
TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS
SCREEN_WIDTH, SCREEN_HEIGHT = 640, 480

GOAL_POS = (9, 15)
GOAL_ANGLE = 0
INTERVAL = 0.1
ORIGIN_DIST_POS = 0
ORIGIN_DIST_ANGLE = 0

def compute_rewards(current_pos, current_angle):
	pos_r = (ORIGIN_DIST_POS - math.sqrt((current_pos[0] - GOAL_POS[0])**2 + (current_pos[1] - GOAL_POS[1])**2))/(ORIGIN_DIST_POS + 1e-2)
	angle_r = (abs(current_angle - GOAL_ANGLE) - ORIGIN_DIST_ANGLE)/(ORIGIN_DIST_ANGLE + 1e-2)
	return 10 * pos_r + angle_r

class StablePushingEnv(gym.Env):
	def __init__(self):
		self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
		pygame.display.set_caption('Rod and box pygame example')
		self.action_space = spaces.Box(low=-10, high=10, shape=(4, ))
		self.observation_space = spaces.Box(low=0, high=255, shape=(SCREEN_WIDTH * SCREEN_HEIGHT * 3, ))
		self.clock = pygame.time.Clock()
		pygame.display.iconify()
		self.done = False
		self.timesteps = 1000
		self.screen.fill((246, 241, 234, 0))
		self._createWorld()
		pygame.display.flip()

	def _createWorld(self):
		# --- pybox2d world setup ---
		# Create the world
		self.world = world(gravity=(0, 0), doSleep=True)

		self.rod = self.world.CreateDynamicBody(position=(10.5, 15), allowSleep=False, userData='rod')
		rodfix = self.rod.CreatePolygonFixture(density=1, box=(0.25, 1), friction=0.0)

		# a target box to move around
		self.box = self.world.CreateDynamicBody(position=(10, 15), allowSleep=False, userData='target')
		boxfix = self.box.CreatePolygonFixture(density=1, box=(1,1), friction=0.5)
		ORIGIN_DIST_POS = math.sqrt((self.box.position[0] - GOAL_POS[0])**2 + (self.box.position[1] - GOAL_POS[1])**2)
		ORIGIN_DIST_ANGLE = abs(self.box.angle - GOAL_ANGLE)

		colors = {
		    'target': (200, 200, 200, 255),
		    'rod': (100, 100, 100, 100),
		}

		def my_draw_polygon(polygon, body, fixture):
			vertices = [(body.transform * v) * PPM for v in polygon.vertices]
			vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
			k = body.userData

			if k == None:
			    k = 'default'

			pygame.draw.polygon(self.screen, colors[k], vertices, 0)

			pygame.draw.polygon(self.screen, (0,0,0,0), vertices, 5)


		polygonShape.draw = my_draw_polygon

		for body in self.world.bodies:
				for fixture in body.fixtures:
					fixture.shape.draw(body, fixture)

	def step(self, action):
		"""assuming action to be a tuple, (vector x, vector y, point x, point y)"""
		obs = None
		reward = 0
		if not self.done:

			self.screen.fill((246, 241, 234, 0))

			# vector, point = (action[0], action[1]), (action[2], action[3])
			# print(vector)

			f = self.rod.GetWorldVector(localVector=(float(action[0]), float(action[1])))
			p = self.rod.GetWorldPoint(localPoint=(float(action[2]), float(action[3])))
			self.rod.ApplyForce(f, p, True)

			for body in self.world.bodies:
				for fixture in body.fixtures:
					fixture.shape.draw(body, fixture)
			obs = np.array(pygame.surfarray.array3d(self.screen)).reshape(SCREEN_WIDTH * SCREEN_HEIGHT * 3, )

			self.world.Step(TIME_STEP, 10, 10)
			self.timesteps -= 1
			pygame.display.flip()
			self.clock.tick(TARGET_FPS)
		
			self.done = bool(
				# self.screen.get_rect().contains(self.box.get_rect()) or \
				# self.screen.get_rect().contains(self.rod.get_rect()) or \
				self.timesteps == 0 or
				(abs(self.box.position[0] - GOAL_POS[0]) <= INTERVAL and \
				abs(self.box.position[1] - GOAL_POS[1]) <= INTERVAL and \
				abs(self.box.angle - GOAL_ANGLE) <= INTERVAL))
			reward = compute_rewards(self.box.position, self.box.angle)
			if self.done:
				pygame.quit()

		return obs, reward, self.done, {}

	def reset(self):
		self.__init__()
		return np.array(pygame.surfarray.array3d(self.screen)).reshape(SCREEN_WIDTH * SCREEN_HEIGHT * 3, )

	def close(self):
		pygame.quit()
