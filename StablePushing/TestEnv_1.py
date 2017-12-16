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
TIME_STEP = 1.0 / TARGET_FPS * 10
SCREEN_WIDTH, SCREEN_HEIGHT = 240, 180

# GOAL_POS = (5, 2)
# GOAL_ANGLE = 0
INTERVAL = 0.1


class TestEnv1(gym.Env):
	def __init__(self):
		# self.screen = pygame.display.set_mode((240, 180), 0, 32)
		self.action_space = spaces.Box(low=-4, high=4, shape=(3, ))
		# self.observation_space = spaces.Box(low=-4, high=15, shape=(6, ))
		self.observation_space = spaces.Box(low=-4, high=15, shape=(4, ))
		self.done = False
		self.goal_pos = (2.5, 2.5)
		self.goal_angle = 0.5
		self.timesteps = 1000
		self.origin_dist_pos = 0
		self.origin_dist_angle = 0
		# self.screen.fill((0, 0, 0, 0))
		# self.screen.set_at((int(self.goal_pos[0] * PPM), int(self.goal_pos[1] * PPM)), (255, 255, 0, 0))
		self._createWorld()

	def compute_rewards(self):
		pos_r = (self.origin_dist_pos - math.sqrt((self.box.position[0] - self.goal_pos[0])**2 + (self.box.position[1] - self.goal_pos[1])**2))/(self.origin_dist_pos + 1e-8)
		angle_r = (self.origin_dist_angle - abs(self.box.angle - self.goal_angle))/(self.origin_dist_angle + 1e-8)
		return pos_r + 0.01 * angle_r

	def _createWorld(self):
		# --- pybox2d world setup ---
		# Create the world
		self.world = world(gravity=(0, 0), doSleep=True)

		# self.rod = self.world.CreateDynamicBody(position=(9.5, 8), allowSleep=False, userData='rod')
		# rodfix = self.rod.CreatePolygonFixture(density=1, box=(0.25, 1), friction=0.0)
		# self.rod.linearDamping = 0.3
		# self.rod.angularDamping = 0.1

		# a target box to move around
		self.box = self.world.CreateDynamicBody(position=(9, 7), allowSleep=False, userData='target')
		boxfix = self.box.CreatePolygonFixture(density=1, vertices=[(-0.5,-0.25), (0.5,-0.25), (0,0.5)], friction=0.5)
		# self.box.linearDamping = 0.3
		# self.box.angularDamping = 0.1

		self.origin_dist_pos = math.sqrt((self.box.position[0] - self.goal_pos[0])**2 + (self.box.position[1] - self.goal_pos[1])**2)
		self.origin_dist_angle = abs(self.box.angle - self.goal_angle)

		# colors = {
		#     'target': (200, 200, 200, 255),
		#     # 'rod': (100, 100, 100, 100),
		# }

		# def my_draw_polygon(polygon, body, fixture):
		# 	vertices = [(body.transform * v) * PPM for v in polygon.vertices]
		# 	vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
		# 	k = body.userData

		# 	if k == None:
		# 	    k = 'default'

		# 	pygame.draw.polygon(self.screen, colors[k], vertices, 0)

		# 	# pygame.draw.polygon(self.screen, (,0,0,0), vertices, 5)


		# polygonShape.draw = my_draw_polygon

		# for body in self.world.bodies:
		# 		for fixture in body.fixtures:
		# 			fixture.shape.draw(body, fixture)

	def step(self, action):
		"""assuming action to be a tuple, (vector x, vector y, point x, point y)"""
		obs = None
		reward = 0
		if not self.done:

			
			# self.screen.fill((0, 0, 0, 0))
			# self.screen.set_at((int(self.goal_pos[0] * PPM), int(self.goal_pos[1] * PPM)), (255, 255, 0, 0))

			rad = float(action[0])
			# f = self.rod.GetWorldVector(localVector=(math.cos(rad), math.sin(rad)))
			# p = self.rod.GetWorldPoint(localPoint=(float(action[1])/16, float(action[2])/4))
			# self.rod.ApplyForce(f, p, True)
			f = self.box.GetWorldVector(localVector=(math.cos(rad), math.sin(rad)))
			p = self.box.GetWorldPoint(localPoint=(float(action[1])/4, float(action[2])/4))
			self.box.ApplyForce(f, p, True)

			self.world.Step(TIME_STEP, 10, 10)
			self.timesteps -= 1
			# obs = np.array([self.box.position[0], self.box.position[1], self.box.angle, self.rod.position[0], self.rod.position[1], self.rod.angle])
			obs = np.array([1, self.box.position[0], self.box.position[1], self.box.angle])
			self.box.linearVelocity[0] = 0.0
			self.box.linearVelocity[1] = 0.0
			self.box.angularVelocity = 0.0
			# self.rod.linearVelocity[0] = 0.0
			# self.rod.linearVelocity[1] = 0.0
			# self.rod.angularVelocity = 0.0


			# for body in self.world.bodies:
			# 	for fixture in body.fixtures:
			# 		fixture.shape.draw(body, fixture)
		
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
			
		return obs, reward, self.done, {} #np.array(pygame.surfarray.array3d(self.screen)).reshape(240 * 180 * 3, )

	def reset(self):
		self.__init__()
		# return np.array([self.box.position[0], self.box.position[1], self.box.angle, self.rod.position[0], self.rod.position[1], self.rod.angle])
		return np.array([1, self.box.position[0], self.box.position[1], self.box.angle])

	def close(self):
		# pygame.display.quit()
		# pygame.quit()
		return

	# def get_image(self):
	# 	return np.array(pygame.surfarray.array3d(self.screen)).reshape(240 * 180 * 3, )

	# def save_image(self, path):
	# 	pygame.image.save(self.screen, path)
