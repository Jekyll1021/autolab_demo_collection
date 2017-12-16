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

