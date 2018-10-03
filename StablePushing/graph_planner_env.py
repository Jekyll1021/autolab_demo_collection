import logging
import math
# import gym
# from gym import spaces
# from gym.utils import seeding
import numpy as np

import pygame
import numpy as np
import pickle

import Box2D  
from Box2D.b2 import (world, polygonShape, circleShape, staticBody, dynamicBody, kinematicBody)

from graph_planner import *
import copy

PPM = 20.0  # pixels per meter
TIME_STEP = 2/3
SCREEN_WIDTH, SCREEN_HEIGHT = 240, 180
INTERVAL = 0.1

DIS_CONST = math.sin(math.pi/4)

Discrete_Points = [(1, 0), (DIS_CONST, DIS_CONST), (0, 1), (-DIS_CONST, DIS_CONST), \
					(-1, 0), (-DIS_CONST, -DIS_CONST), (0, -1), (DIS_CONST, -DIS_CONST)]

def euclidean_dist(pos1, pos2):
	return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

def line_by_two_pts(p1, p2):
	"""either of the two: y = kx + b; x = b"""
	if (p2[0] != p1[0]): # y = kx + b
		slope = (p2[1] - p1[1])/(p2[0] - p1[0])
		return (slope, p1[1] - slope*p1[0])
	else:
		return (None, p1[0])

def intersect_two_lines(l1, l2):
	"""
	each of the line is a tuple of (slope, intercept)/(k, b).
	two cases to handle: (k, b) -> y = kx+b;
						 (None, b) -> x = b
	"""
	if (l1[0] != None and l2[0] != None and l1[0] != l2[0]):
		x = (l2[1] - l1[1])/(l1[0] - l2[0])
		y = l1[0] * x + l1[1]
		return (x, y)
	elif l1[0] == None and l2[0] != None:
		return (l1[1], l2[0]*l1[1] + l2[1])
	elif l2[0] == None and l1[0] != None:
		return (l2[1], l1[0]*l2[1] + l1[1])
	else:
		return None 

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

def get_orientation_force(centroid, vertex, use_param=True):
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
	if not use_param:
		return [Action((centroid[1] - vertex[1], vertex[0] - centroid[0]), vertex), \
				Action((vertex[1] - centroid[1], centroid[0] - vertex[0]), vertex)]
	else:
		return [Action((centroid[1] - vertex[1], vertex[0] - centroid[0]), (vertex[0]/10*9, vertex[1]/10*9)), \
				Action((vertex[1] - centroid[1], centroid[0] - vertex[0]), (vertex[0]/10*9, vertex[1]/10*9))]

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

	def __eq__(self, other):
		"""
		check if vector == vector, point == point
		"""
		return self.vector == other.vector and self.point == self.point

class PolygonEnv:
	def __init__(self, original_pos, vertices, goal_pos, goal_angle, use_param=True, use_discrete=False):
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
		self.vertices = vertices
		self.use_param = use_param

		# figure out center of mass
		self.centroid = compute_centroid(vertices)

		# centeralize polygon by centroid
		self.original_pos = (original_pos[0] + self.centroid[0], original_pos[1] + self.centroid[1])
		self.goal_pos = (goal_pos[0] + self.centroid[0], goal_pos[1] + self.centroid[1])
		self.vertices = [(v[0] - self.centroid[0], v[1] - self.centroid[1]) for v in vertices]
		self.centroid = (0, 0)
		self.bounding_circle_radius = max([euclidean_dist(v, self.centroid) for v in self.vertices])

		# a target box to move around
		self.box = self.world.CreateDynamicBody(position=self.original_pos, allowSleep=False, userData='target')
		boxfix = self.box.CreatePolygonFixture(density=1, vertices=self.vertices, friction=0.5)

		self.actions = []

		n = len(self.vertices)

		if not use_discrete:
			# figure out actions that relate to orientations
			for i in range(n):
				curr = self.vertices[i]
				self.actions.extend(get_orientation_force(self.centroid, curr, use_param))

			# figure out actions that relate to translations
			for i in range(n):
				curr = self.vertices[(i - n) % n]
				next = self.vertices[(i + 1 - n) % n]
				self.actions.append(get_trans_force(self.centroid, curr, next))

		else:
			discrete_pts_lst = [(p[0]*self.bounding_circle_radius, p[1]*self.bounding_circle_radius) for p in Discrete_Points]
			for i in range(len(discrete_pts_lst)):
				for j in range(len(discrete_pts_lst)):
					if i != j:
						self.actions.append(self.bounding_param_to_actions(discrete_pts_lst[i], discrete_pts_lst[j]))

		if use_param:
			self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
			pygame.display.set_caption('example')
			pygame.display.iconify()

	def _vertexIterator(self):
		N = len(self.vertices)
		for i in range(N):
			j = (i + 1) % N

			xi = self.vertices[i][0]
			xj = self.vertices[j][0]
			yi = self.vertices[i][1]
			yj = self.vertices[j][1]

			yield {'cur':(xi,yi), \
					'next':(xj, yj), \
					'det': (xi*yj - xj*yi) }

	def edgePointContact(self, p1, distance=0.01):
		contacts = [p1 for v in self._vertexIterator() if self.pointToLineDistance(p1, v['cur'], v['next']) < distance] 
		return (len(contacts) > 0)

	def pointToLineDistance(self, p1, e1, e2):
		numerator = np.abs((e2[1] - e1[1])*p1[0] - (e2[0] - e1[0])*p1[1] + e2[0]*e1[1] - e1[0]*e2[1])
		normalization =  np.sqrt((e2[1] - e1[1])**2 + (e2[0] - e1[0])**2)
		return numerator/normalization

	def closestPointOnEdge(self, p1):
		"""TODO: debug, possible error in sorted"""
		close = sorted([(p1, self.pointToLineDistance(p1, v['cur'], v['next']), v['cur'], v['next']) for v in self._vertexIterator()], key=lambda x: x[1])[0]
		ratio = (euclidean_dist(close[0], (0, 0)) - self.pointToLineDistance(close[0], close[2], close[3])) / euclidean_dist(close[0], (0, 0))
		return (close[0][0] * ratio, close[0][1] * ratio)

	def parametrize_by_bounding_circle(self, action):
		"""parametrize as p1 to p2"""
		a = (action.vector[0]**2 + action.vector[1]**2)
		b = (2 * action.point[0] * action.vector[0] + 2 * action.point[1] * action.vector[1])
		c = (action.point[0] ** 2 + action.point[1] ** 2 - self.bounding_circle_radius ** 2)
		if (b**2 - 4 * a * c) < 0:
			print("unable to parametrize by bounding circle: line of force does not touch bounding circle")
			return None
		else:
			t1 = (-b + math.sqrt(b**2 - 4 * a * c))/(2*a)
			t2 = (-b - math.sqrt(b**2 - 4 * a * c))/(2*a)
			p1 = (action.point[0] + t2 * action.vector[0], action.point[1] + t2 * action.vector[1])
			p2 = (action.point[0] + t1 * action.vector[0], action.point[1] + t1 * action.vector[1])
			return [p1[0], p1[1], p2[0], p2[1]]

	def bounding_param_to_actions(self, p1, p2):
		line_force = line_by_two_pts(p1, p2)
		potential_intercepts = [intersect_two_lines(line_force, line_by_two_pts(v['cur'], v['next'])) for v in self._vertexIterator()]
		potential_intercepts = [p for p in potential_intercepts if p is not None]
		intercept = sorted(potential_intercepts, key = lambda p: euclidean_dist(p1, p))[0]
		return Action((p2[0]-p1[0], p2[1]-p1[1]), intercept)

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

	def get_goal_image(self):
		self.screen.fill((0, 0, 0, 0))

		def my_draw_polygon(polygon, body, fixture):
			vertices = [(body.transform * v) * PPM for v in polygon.vertices]
			vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
			k = body.userData

			if k == None:
			    k = 'default'

			pygame.draw.polygon(self.screen, (200, 200, 200, 255), vertices, 0)


		polygonShape.draw = my_draw_polygon

		self.box.position = self.goal_pos
		self.box.angle = self.goal_angle

		for body in self.world.bodies:
				for fixture in body.fixtures:
					fixture.shape.draw(body, fixture)

		img = np.array(pygame.surfarray.array3d(self.screen)).reshape(240 * 180 * 3, )
		pygame.image.save(self.screen, "goal.png")


		return img

	def get_start_image(self):
		self.screen.fill((0, 0, 0, 0))

		def my_draw_polygon(polygon, body, fixture):
			vertices = [(body.transform * v) * PPM for v in polygon.vertices]
			vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
			k = body.userData

			if k == None:
			    k = 'default'

			pygame.draw.polygon(self.screen, (200, 200, 200, 255), vertices, 0)


		polygonShape.draw = my_draw_polygon

		self.box.position = self.original_pos
		self.box.angle = 0

		for body in self.world.bodies:
				for fixture in body.fixtures:
					fixture.shape.draw(body, fixture)

		img = np.array(pygame.surfarray.array3d(self.screen)).reshape(240 * 180 * 3, )
		pygame.image.save(self.screen, "start.png")


		return img


	def step_with_image(self, previous_pos, previous_angle, action, path="", save_image=False):
		self.screen.fill((0, 0, 0, 0))

		def my_draw_polygon(polygon, body, fixture):
			vertices = [(body.transform * v) * PPM for v in polygon.vertices]
			vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
			k = body.userData

			if k == None:
			    k = 'default'

			pygame.draw.polygon(self.screen, (200, 200, 200, 255), vertices, 0)


		polygonShape.draw = my_draw_polygon

		self.box.position = previous_pos
		self.box.angle = previous_angle

		for body in self.world.bodies:
				for fixture in body.fixtures:
					fixture.shape.draw(body, fixture)

		prev_ob = np.array(pygame.surfarray.array3d(self.screen)).reshape(240 * 180 * 3, )

		if save_image:
			pygame.image.save(self.screen, path+"prev.png")

		self.screen.fill((0, 0, 0, 0))

		f = self.box.GetWorldVector(localVector=(action.vector))
		p = self.box.GetWorldPoint(localPoint=(action.point))
		self.box.ApplyForce(f, p, True)
		self.world.Step(TIME_STEP, 10, 10)
		self.box.linearVelocity[0] = 0.0
		self.box.linearVelocity[1] = 0.0
		self.box.angularVelocity = 0.0

		for body in self.world.bodies:
				for fixture in body.fixtures:
					fixture.shape.draw(body, fixture)

		next_ob = np.array(pygame.surfarray.array3d(self.screen)).reshape(240 * 180 * 3, )

		if save_image:
			pygame.image.save(self.screen, path+"next.png")

		return self.box.position, self.box.angle, prev_ob, next_ob
		
	def animate(self, list_actions, save_image=True, max_acts=3):
		"""taking in a list of actions in reverse order, generate animation when rolling out data and pack action/state/goal"""
		self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
		pygame.display.set_caption('example')
		pygame.display.iconify()
		self.screen.fill((0, 0, 0, 0))
		pygame.display.flip()
		self.box.position = self.original_pos
		self.box.angle = 0.0
		
		obs = []
		acts = []
		goal_obs = []
		bounding_circle_acts = []
		bounding_circle_normalized_acts = []

		def my_draw_polygon(polygon, body, fixture):
			vertices = [(body.transform * v) * PPM for v in polygon.vertices]
			vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
			k = body.userData

			if k == None:
			    k = 'default'

			pygame.draw.polygon(self.screen, (200, 200, 200, 255), vertices, 0)


		polygonShape.draw = my_draw_polygon

		self.box.position = self.goal_pos
		self.box.angle = self.goal_angle

		for body in self.world.bodies:
				for fixture in body.fixtures:
					fixture.shape.draw(body, fixture)

		goal_ob = np.array(pygame.surfarray.array3d(self.screen)).reshape(240 * 180 * 3, )
		if save_image:
			pygame.image.save(self.screen, "goal.png")
		goal_obs.append(goal_ob)

		self.screen.fill((0, 0, 0, 0))

		self.box.position = self.original_pos
		self.box.angle = 0.0

		for body in self.world.bodies:
				for fixture in body.fixtures:
					fixture.shape.draw(body, fixture)
		
		ind = 0

		set_actions = set([(a.vector[0], a.vector[1], a.point[0], a.point[1]) for a in list_actions])
		p_act = []
		b_p_act = []
		bn_p_act = []
		obs.append(np.array(pygame.surfarray.array3d(self.screen)).reshape(240 * 180 * 3, ))
		a = None
		for i in range(max_acts):
			# a = None
			if len(set_actions) > 0:
				a = set_actions.pop()
				act = Action((a[0], a[1]), (a[2], a[3]))
				b_p_act.append(np.array(self.parametrize_by_bounding_circle(act)))
				bn_p_act.append(np.array([p/self.bounding_circle_radius for p in self.parametrize_by_bounding_circle(act)]))
			p_act.append(np.array(a))
		acts.append(p_act)
		bounding_circle_acts.append(b_p_act)
		bounding_circle_normalized_acts.append(bn_p_act)

		if save_image:
			pygame.image.save(self.screen, "anim"+str(ind)+".png")
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
			ind += 1
			if save_image:
				pygame.image.save(self.screen, "anim"+str(ind)+".png")

			set_actions = set([(a.vector[0], a.vector[1], a.point[0], a.point[1]) for a in list_actions])
			p_act = []
			b_p_act = []
			bn_p_act = []
			goal_obs.append(goal_ob)
			obs.append(np.array(pygame.surfarray.array3d(self.screen)).reshape(240 * 180 * 3, ))
			a = None
			for i in range(max_acts):
				if len(set_actions) > 0:
					a = set_actions.pop()
				p_act.append(np.array(a))
				if a is None:
					b_p_act.append(None)
					bn_p_act.append(None)
				else:
					act = Action((a[0], a[1]), (a[2], a[3]))
					b_p_act.append(np.array(self.parametrize_by_bounding_circle(act)))
					bn_p_act.append(np.array([p/self.bounding_circle_radius for p in self.parametrize_by_bounding_circle(act)]))
			acts.append(p_act)
			bounding_circle_acts.append(b_p_act)
			bounding_circle_normalized_acts.append(bn_p_act)

		pygame.display.quit()
		pygame.quit()

		return obs, goal_obs, acts, bounding_circle_acts, bounding_circle_normalized_acts

	def rollout_predict_model_policy(self, model, save_image_path, save_image=True, horizon=50):
		"""rollout for double bn fitting"""

		discrete_pts_lst = [(p[0]*self.bounding_circle_radius, p[1]*self.bounding_circle_radius) for p in Discrete_Points]

		if not self.use_param:
			self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
			pygame.display.set_caption('example')
			pygame.display.iconify()

		self.screen.fill((0, 0, 0, 0))
		pygame.display.flip()

		def my_draw_polygon(polygon, body, fixture):
			vertices = [(body.transform * v) * PPM for v in polygon.vertices]
			vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
			k = body.userData

			if k == None:
			    k = 'default'

			pygame.draw.polygon(self.screen, (200, 200, 200, 255), vertices, 0)

		polygonShape.draw = my_draw_polygon

		self.box.position = self.goal_pos
		self.box.angle = self.goal_angle

		for body in self.world.bodies:
				for fixture in body.fixtures:
					fixture.shape.draw(body, fixture)

		goal_ob = np.array(pygame.surfarray.array3d(self.screen)).reshape(240 * 180 * 3, )

		if save_image:
			pygame.image.save(self.screen, save_image_path+"goal.png")

		self.screen.fill((0, 0, 0, 0))

		self.box.position = self.original_pos
		self.box.angle = 0.0

		for body in self.world.bodies:
				for fixture in body.fixtures:
					fixture.shape.draw(body, fixture)

		ob = np.array(pygame.surfarray.array3d(self.screen)).reshape(240 * 180 * 3, )

		if save_image:
			pygame.image.save(self.screen, save_image_path+"anim0.png")

		closest = 0
		closest_dist = euclidean_dist(self.box.position, self.goal_pos) + 0.1*abs(self.box.angle - self.goal_angle)

		for i in range(horizon):
			self.screen.fill((0, 0, 0, 0))
			raw_p1, raw_p2 = model.predict(ob, goal_ob)
			p1 = discrete_pts_lst[int(round(raw_p1)) % 8]
			p2 = discrete_pts_lst[int(round(raw_p2)) % 8]

			if p1 == p2:
				continue

			action = self.bounding_param_to_actions(p1, p2)
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

			ob = np.array(pygame.surfarray.array3d(self.screen)).reshape(240 * 180 * 3, )

			if save_image:
				pygame.image.save(self.screen, save_image_path+"anim"+str(i+1)+".png")

			if (euclidean_dist(self.box.position, self.goal_pos) < INTERVAL and abs(self.box.angle - self.goal_angle) < INTERVAL):
				return i+1

			if euclidean_dist(self.box.position, self.goal_pos) + 0.1*abs(self.box.angle - self.goal_angle) < closest_dist:
				closest_dist = euclidean_dist(self.box.position, self.goal_pos) + 0.1*abs(self.box.angle - self.goal_angle)
				closest = i+1

		pygame.display.quit()
		pygame.quit()

		return closest

	def rollout_classification_model_policy(self, model, save_image_path, save_image=True, horizon=50):
		"""rollout for double bn fitting"""

		discrete_pts_lst = [(p[0]*self.bounding_circle_radius, p[1]*self.bounding_circle_radius) for p in Discrete_Points]

		if not self.use_param:
			self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
			pygame.display.set_caption('example')
			pygame.display.iconify()

		self.screen.fill((0, 0, 0, 0))
		pygame.display.flip()

		def my_draw_polygon(polygon, body, fixture):
			vertices = [(body.transform * v) * PPM for v in polygon.vertices]
			vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
			k = body.userData

			if k == None:
			    k = 'default'

			pygame.draw.polygon(self.screen, (200, 200, 200, 255), vertices, 0)

		polygonShape.draw = my_draw_polygon

		self.box.position = self.goal_pos
		self.box.angle = self.goal_angle

		for body in self.world.bodies:
				for fixture in body.fixtures:
					fixture.shape.draw(body, fixture)

		goal_ob = np.array(pygame.surfarray.array3d(self.screen)).reshape(240 * 180 * 3, )

		if save_image:
			pygame.image.save(self.screen, save_image_path+"goal.png")

		self.screen.fill((0, 0, 0, 0))

		self.box.position = self.original_pos
		self.box.angle = 0.0

		for body in self.world.bodies:
				for fixture in body.fixtures:
					fixture.shape.draw(body, fixture)

		ob = np.array(pygame.surfarray.array3d(self.screen)).reshape(240 * 180 * 3, )

		if save_image:
			pygame.image.save(self.screen, save_image_path+"anim0.png")

		closest = 0
		closest_dist = euclidean_dist(self.box.position, self.goal_pos)
		closest_angle = abs(self.box.angle - self.goal_angle)

		for i in range(horizon):
			self.screen.fill((0, 0, 0, 0))
			"""TODO: fill out action selection"""

			action = self.actions[0]
			prob = 0

			for a in self.actions:
				curr_prob = model.predict(ob, goal_ob, [x / self.bounding_circle_radius for x in self.parametrize_by_bounding_circle(a)])[1]
				if curr_prob > prob:
					action = a
					prob = curr_prob

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

			ob = np.array(pygame.surfarray.array3d(self.screen)).reshape(240 * 180 * 3, )

			if save_image:
				pygame.image.save(self.screen, save_image_path+"anim"+str(i+1)+".png")

			if (euclidean_dist(self.box.position, self.goal_pos) < INTERVAL and abs(self.box.angle - self.goal_angle) < INTERVAL):
				return i+1

			if euclidean_dist(self.box.position, self.goal_pos) + 0.1*abs(self.box.angle - self.goal_angle) < closest_dist:
				closest_dist = euclidean_dist(self.box.position, self.goal_pos)
				closest_angle = abs(self.box.angle - self.goal_angle)
				closest = i+1

		pygame.display.quit()
		pygame.quit()

		return closest_dist / euclidean_dist(self.box.position, self.goal_pos), closest_angle / abs(self.box.angle - self.goal_angle), closest

	def rollout_classification_abs_rot_model_policy(self, model, save_image_path, save_image=True, horizon=50):
		"""rollout for double bn fitting"""

		action_space_param = []
		for p1 in Discrete_Points:
			for p2 in Discrete_Points:
				if p1 != p2:
					action_space_param.append([p1[0], p1[1], p2[0], p2[1]])

		if not self.use_param:
			self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
			pygame.display.set_caption('example')
			pygame.display.iconify()

		self.screen.fill((0, 0, 0, 0))
		pygame.display.flip()

		def my_draw_polygon(polygon, body, fixture):
			vertices = [(body.transform * v) * PPM for v in polygon.vertices]
			vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
			k = body.userData

			if k == None:
			    k = 'default'

			pygame.draw.polygon(self.screen, (200, 200, 200, 255), vertices, 0)

		polygonShape.draw = my_draw_polygon

		self.box.position = self.goal_pos
		self.box.angle = self.goal_angle

		for body in self.world.bodies:
				for fixture in body.fixtures:
					fixture.shape.draw(body, fixture)

		goal_ob = np.array(pygame.surfarray.array3d(self.screen)).reshape(240 * 180 * 3, )

		if save_image:
			pygame.image.save(self.screen, save_image_path+"goal.png")

		self.screen.fill((0, 0, 0, 0))

		self.box.position = self.original_pos
		self.box.angle = 0.0

		for body in self.world.bodies:
				for fixture in body.fixtures:
					fixture.shape.draw(body, fixture)

		ob = np.array(pygame.surfarray.array3d(self.screen)).reshape(240 * 180 * 3, )

		if save_image:
			pygame.image.save(self.screen, save_image_path+"anim0.png")

		closest = 0
		closest_dist = euclidean_dist(self.box.position, self.goal_pos)
		closest_angle = abs(self.box.angle - self.goal_angle)

		for i in range(horizon):
			self.screen.fill((0, 0, 0, 0))
			"""TODO: fill out action selection"""

			action = self.actions[0]
			prob = 0

			for ap in action_space_param:
				curr_prob = model.predict(ob, goal_ob, ap)[1]
				if curr_prob > prob:
					action_p = ap
					prob = curr_prob

			param_a = copy.deepcopy(action_p)
			param_a[0] = action_p[0] * math.cos(-polygon_env.box.angle) - action_p[1] * math.sin(-polygon_env.box.angle)
			param_a[1] = action_p[0] * math.sin(-polygon_env.box.angle) + action_p[1] * math.cos(-polygon_env.box.angle)
			param_a[2] = action_p[2] * math.cos(-polygon_env.box.angle) - action_p[3] * math.sin(-polygon_env.box.angle)
			param_a[3] = action_p[2] * math.sin(-polygon_env.box.angle) + action_p[3] * math.cos(-polygon_env.box.angle)
			a = polygon_env.bounding_param_to_actions((param_a[0]*polygon_env.bounding_circle_radius, param_a[1]*polygon_env.bounding_circle_radius), \
			(param_a[2]*polygon_env.bounding_circle_radius, param_a[3]*polygon_env.bounding_circle_radius))

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

			ob = np.array(pygame.surfarray.array3d(self.screen)).reshape(240 * 180 * 3, )

			if save_image:
				pygame.image.save(self.screen, save_image_path+"anim"+str(i+1)+".png")

			if (euclidean_dist(self.box.position, self.goal_pos) < INTERVAL and abs(self.box.angle - self.goal_angle) < INTERVAL):
				return i+1

			if euclidean_dist(self.box.position, self.goal_pos) + 0.1*abs(self.box.angle - self.goal_angle) < closest_dist:
				closest_dist = euclidean_dist(self.box.position, self.goal_pos)
				closest_angle = abs(self.box.angle - self.goal_angle)
				closest = i+1

		pygame.display.quit()
		pygame.quit()

		return closest_dist / euclidean_dist(self.box.position, self.goal_pos), closest_angle / abs(self.box.angle - self.goal_angle), closest

	def rollout_classification_rot_model_policy(self, model, save_image_path, save_image=True, horizon=3):
		"""rollout for double bn fitting"""

		action = [Discrete_Points[0][0], Discrete_Points[0][1], Discrete_Points[2][0], Discrete_Points[2][1]]

		if not self.use_param:
			self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
			pygame.display.set_caption('example')
			pygame.display.iconify()

		self.screen.fill((0, 0, 0, 0))
		pygame.display.flip()

		def my_draw_polygon(polygon, body, fixture):
			vertices = [(body.transform * v) * PPM for v in polygon.vertices]
			vertices = [(v[0], SCREEN_HEIGHT - v[1]) for v in vertices]
			k = body.userData

			if k == None:
			    k = 'default'

			pygame.draw.polygon(self.screen, (200, 200, 200, 255), vertices, 0)

		polygonShape.draw = my_draw_polygon

		self.box.position = self.goal_pos
		self.box.angle = self.goal_angle

		for body in self.world.bodies:
				for fixture in body.fixtures:
					fixture.shape.draw(body, fixture)

		goal_ob = np.array(pygame.surfarray.array3d(self.screen)).reshape(240 * 180 * 3, )

		if save_image:
			pygame.image.save(self.screen, save_image_path+"goal.png")

		self.screen.fill((0, 0, 0, 0))

		self.box.position = self.original_pos
		self.box.angle = 0.0

		for body in self.world.bodies:
				for fixture in body.fixtures:
					fixture.shape.draw(body, fixture)

		ob = np.array(pygame.surfarray.array3d(self.screen)).reshape(240 * 180 * 3, )

		if save_image:
			pygame.image.save(self.screen, save_image_path+"anim0.png")

		closest = 0
		closest_dist = euclidean_dist(self.box.position, self.goal_pos)
		closest_angle = abs(self.box.angle - self.goal_angle) % (math.pi/2)

		for i in range(horizon):
			self.screen.fill((0, 0, 0, 0))
			"""TODO: fill out action selection"""

			rot_prob = model.predict(ob, goal_ob)[1]
			if rot_prob > 0.5:
				action_p = action
			else:
				for body in self.world.bodies:
					for fixture in body.fixtures:
						fixture.shape.draw(body, fixture)
				# pygame.display.flip()
				pygame.image.save(self.screen, save_image_path+"not_rotating"+str(i)+".png")
				return euclidean_dist(self.box.position, self.goal_pos), (abs(self.box.angle + math.pi*2 - self.goal_angle) % (math.pi/2)), closest

			param_a = copy.deepcopy(action_p)
			param_a[0] = action_p[0] * math.cos(-self.box.angle) - action_p[1] * math.sin(-self.box.angle)
			param_a[1] = action_p[0] * math.sin(-self.box.angle) + action_p[1] * math.cos(-self.box.angle)
			param_a[2] = action_p[2] * math.cos(-self.box.angle) - action_p[3] * math.sin(-self.box.angle)
			param_a[3] = action_p[2] * math.sin(-self.box.angle) + action_p[3] * math.cos(-self.box.angle)
			a = self.bounding_param_to_actions((param_a[0]*self.bounding_circle_radius, param_a[1]*self.bounding_circle_radius), \
			(param_a[2]*self.bounding_circle_radius, param_a[3]*self.bounding_circle_radius))

			f = self.box.GetWorldVector(localVector=a.vector)
			p = self.box.GetWorldPoint(localPoint=a.point)
			self.box.ApplyForce(f, p, True)
			self.world.Step(TIME_STEP, 10, 10)
			self.box.linearVelocity[0] = 0.0
			self.box.linearVelocity[1] = 0.0
			self.box.angularVelocity = 0.0

			for body in self.world.bodies:
				for fixture in body.fixtures:
					fixture.shape.draw(body, fixture)

			# pygame.display.flip()

			ob = np.array(pygame.surfarray.array3d(self.screen)).reshape(240 * 180 * 3, )

			if save_image:
				pygame.image.save(self.screen, save_image_path+"anim"+str(i+1)+".png")

			if (euclidean_dist(self.box.position, self.goal_pos) < INTERVAL and abs(self.box.angle - self.goal_angle) < INTERVAL):
				return i+1

			if euclidean_dist(self.box.position, self.goal_pos) + 0.1*abs(self.box.angle - self.goal_angle) < closest_dist:
				closest_dist = euclidean_dist(self.box.position, self.goal_pos)
				closest_angle = abs(self.box.angle - self.goal_angle)
				closest = i+1

		pygame.display.quit()
		pygame.quit()

		return euclidean_dist(self.box.position, self.goal_pos), abs(self.box.angle + math.pi*2 - self.goal_angle)% (math.pi/2), closest


# class TestModel:
# 	def __init__(self):
# 		pass

# 	def predict(self, state, goal_state):
# 		return (0, 4)
