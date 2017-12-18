import collections
import sys

from graph_planner_env import *

class Vertex:
	def __init__(self, pos, angle, action, timestamp, parent):
		self.pos = pos
		self.angle = angle
		self.action = action
		self.parent = parent
		self.timestamp = timestamp

def euclidean_dist(pos1, pos2):
	return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

def modified_bfs_planner(polygon_env, horizon=100):
	queue = collections.deque()
	for a in polygon_env.actions:
		queue.append(Vertex(polygon_env.original_pos, 0, a, 0, None))
	ending_vertices = []
	while queue:
		vertex = queue.popleft()
		curr_pos, curr_angle = polygon_env.step(vertex.pos, vertex.angle, vertex.action)
		print(vertex.pos, curr_pos)
		if (euclidean_dist(curr_pos, polygon_env.goal_pos) < INTERVAL and abs(curr_angle - polygon_env.goal_angle) < INTERVAL):
			ending_vertices.append(vertex)
			break
		if vertex.timestamp == horizon:
			ending_vertices.append(vertex)
		if (vertex.timestamp > 0 and vertex.timestamp < horizon and \
			(euclidean_dist(curr_pos, polygon_env.goal_pos) < euclidean_dist(vertex.pos, polygon_env.goal_pos) or \
			(euclidean_dist(curr_pos, polygon_env.goal_pos) < euclidean_dist(polygon_env.original_pos, polygon_env.goal_pos)))) or vertex.timestamp == 0:
			for act in polygon_env.actions:
				queue.append(Vertex(curr_pos, curr_angle, act, vertex.timestamp+1, vertex))
	ending_vertex = None
	min_dist = sys.maxsize
	for v in ending_vertices:
		if euclidean_dist(v.pos, polygon_env.goal_pos) < min_dist:
			ending_vertex = v
			min_dist = euclidean_dist(v.pos, polygon_env.goal_pos)
	return ending_vertex
