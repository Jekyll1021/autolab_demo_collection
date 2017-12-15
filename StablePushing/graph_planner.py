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
	action(object): action that with magnitude 1 and 
	"""
class Action:
	def __init__(self, vector, point):
		"""
		action that consists of:
		vector: (x, y) force vector
		point: (x, y) point of contact

		all relative to the local origin of polygon.
		"""
		self.vector = vector
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
		self.centroid = compute_centroid(vertices)

