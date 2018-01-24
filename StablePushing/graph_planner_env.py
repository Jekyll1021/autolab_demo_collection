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
TIME_STEP = 1
SCREEN_WIDTH, SCREEN_HEIGHT = 240, 180
INTERVAL = 0.1

def compute_centroid(vertices):
	"""
	helper function:

	input:
	vertices: a list of vertices of a polygon 
	under the assumption that all vertices are ordered either clockwise/counterclockwise

	output: 
	centroid: position of (x, y) tuple of the polygon relative to the local origin of polygon. 
	"""
	c_x = 0
	c_y = 0
	area = 0
	n = len(vertices)
	for i in range(n):
		curr = vertices[(i - n) % n]
		next = vertices[(i + 1 - n) % n]
		diff = (curr[0] * next[1] - curr[1] * next[0])
		c_x += (curr[0] + next[0]) * diff
		c_y += (curr[1] + next[1]) * diff
		area += diff
	area = area / 2
	c_x = c_x / (6 * area)
	c_y = c_y / (6 * area)
	return c_x, c_y

def get_trans_force(centroid, vertex1, vertex2):
	"""
	helper funtion:

	input:
	centroid: centroid of the polygon
	vertex1, vertex2: two points that defines an edge of the polygon

	output:
	action(object): magnitude 1 action with
	direction as perpendicular to the edge, 
	point of contact as projection of centroid
	"""
	k = ((vertex2[1]-vertex1[1])*(centroid[0]-vertex1[0])-(vertex2[0]-vertex1[0])*(centroid[1]-vertex1[1])) \
		/((vertex2[1]-vertex1[1])**2+(vertex2[0]-vertex1[0])**2)
	p = (centroid[0] - k * (vertex2[1] - vertex1[1]), centroid[1] + k * (vertex2[0] - vertex1[0]))
	v = (centroid[0] - p[0], centroid[1] - p[1])
	return Action(v, p)


def normalize(vector):
	"""
	helper function: 

	input:
	vector: (x, y) force vector

	output:
	vector: (x, y) force vector with normalized magnitude 1
	"""
	mag = math.sqrt(vector[0] ** 2 + vector[1] ** 2)
	return vector[0] / mag, vector[1] / mag

def get_orientation_force(centroid, vertex):
	"""
	helper function:

	input:
	centroid: (x, y) position of COM of polygon
	vertex: (x, y) position of point of contact

	output:
	two action objects: magnitude 1 action with
	two directions as perpendicular to line connecting centroid and the vertex, 
	point of contact as the vertex

	Will be further explained in a figure.
	"""
	return [Action((centroid[1] - vertex[1], vertex[0] - centroid[0]), vertex), \
			Action((vertex[1] - centroid[1], centroid[0] - vertex[0]), vertex)]

class Action:
	def __init__(self, vector, point):
		"""
		action that consists of:
		vector: (x, y) force vector
		point: (x, y) point of contact

		all relative to the local origin of polygon.
		"""
		self.vector = normalize(vector)
		self.point = point

class PolygonEnv:
	def __init__(self, original_pos, vertices, goal_pos, goal_angle):
		"""
		original_pos: define the original position by (x, y) tuple
		vertices: define the polygon as an ordered list of its vertices (x, y) tuple with the original_pos as (0, 0). 
			under assumptions:
			1. vertices are ordered as either clockwise/counterclockwise
			2. the polygon itself is a convex set
		goal_pos: define the goal position by (x, y) tuple
		goal_angle: define the goal angle by a float
		"""
		# self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
		self.done = False
		self.original_pos = original_pos
		self.goal_pos = goal_pos
		self.goal_angle = goal_angle
		self.world = world(gravity=(0, 0), doSleep=True)

		# a target box to move around
		self.box = self.world.CreateDynamicBody(position=original_pos, allowSleep=False, userData='target')
		boxfix = self.box.CreatePolygonFixture(density=1, vertices=vertices, friction=0.5)

		# figure out center of mass
		self.centroid = compute_centroid(vertices)

		self.actions = []

		# figure out actions that relate to translations
		n = len(vertices)
		for i in range(n):
			curr = vertices[(i - n) % n]
			next = vertices[(i + 1 - n) % n]
			self.actions.append(get_trans_force(self.centroid, curr, next))

		# figure out actions that relate to orientations
		for i in range(n):
			curr = vertices[i]
			self.actions.extend(get_orientation_force(self.centroid, curr))

	def step(self, previous_pos, previous_angle, action):
		self.box.position = previous_pos
		self.box.angle = previous_angle
		f = self.box.GetWorldVector(localVector=(action.vector))
		p = self.box.GetWorldPoint(localPoint=(action.point))
		self.box.ApplyForce(f, p, True)
		self.world.Step(TIME_STEP, 10, 10)
		self.box.linearVelocity[0] = 0.0
		self.box.linearVelocity[1] = 0.0
		self.box.angularVelocity = 0.0
		return self.box.position, self.box.angle

	def animate(self, list_actions):
		self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
		pygame.display.set_caption('example')
		pygame.display.iconify()
		self.screen.fill((0, 0, 0, 0))
		pygame.display.flip()
		self.box.position = self.original_pos
		self.box.angle = 0.0

		def my_draw_polygon(polygon, body, fixture):
			vertices = [(body.transform * v) * PPM for v in polygon.vertices]
			vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
			k = body.userData

			if k == None:
			    k = 'default'

			pygame.draw.polygon(self.screen, (200, 200, 200, 255), vertices, 0)


		polygonShape.draw = my_draw_polygon

		for body in self.world.bodies:
				for fixture in body.fixtures:
					fixture.shape.draw(body, fixture)
		i = 0
		pygame.image.save(self.screen, "anim"+str(i)+".png")
		while list_actions != []:
			self.screen.fill((0, 0, 0, 0))
			action = list_actions.pop()
			vector = action.vector
			point = action.point
			f = self.box.GetWorldVector(localVector=vector)
			p = self.box.GetWorldPoint(localPoint=point)
			self.box.ApplyForce(f, p, True)
			self.world.Step(TIME_STEP, 10, 10)
			self.box.linearVelocity[0] = 0.0
			self.box.linearVelocity[1] = 0.0
			self.box.angularVelocity = 0.0

			for body in self.world.bodies:
				for fixture in body.fixtures:
					fixture.shape.draw(body, fixture)

			pygame.display.flip()
			i += 1
			pygame.image.save(self.screen, "anim"+str(i)+".png")

		pygame.display.quit()
		pygame.quit()


