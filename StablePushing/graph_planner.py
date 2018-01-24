import collections
import sys

from graph_planner_env import *

class Vertex:
	def __init__(self, pos, angle, action, timestamp, parent):
		self.pos = (pos[0], pos[1])
		self.angle = angle
		self.action = action
		self.parent = parent
		self.timestamp = timestamp

def euclidean_dist(pos1, pos2):
	return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

def modified_bfs_planner(polygon_env, horizon=1000):
	queue = collections.deque()
	for a in polygon_env.actions:
		queue.append(Vertex(polygon_env.original_pos, 0, a, 0, None))
	ending_vertex = None
	seen = set()
	while queue:
		vertex = queue.popleft()
		curr_pos, curr_angle = polygon_env.step(vertex.pos, vertex.angle, vertex.action)
		if ending_vertex == None or \
		10*euclidean_dist(curr_pos, polygon_env.goal_pos) + abs(curr_angle - polygon_env.goal_angle) < \
		10*euclidean_dist(ending_vertex.pos, polygon_env.goal_pos) + abs(ending_vertex.angle - polygon_env.goal_angle):
			ending_vertex = vertex
		print(euclidean_dist(curr_pos, polygon_env.goal_pos))
		if (euclidean_dist(curr_pos, polygon_env.goal_pos) < INTERVAL and abs(curr_angle - polygon_env.goal_angle) < INTERVAL):
			break
		if (vertex.timestamp > 0 and vertex.timestamp < horizon and \
			(euclidean_dist(curr_pos, polygon_env.goal_pos) < euclidean_dist(vertex.pos, polygon_env.goal_pos) +0.05 and \
			euclidean_dist(curr_pos, polygon_env.goal_pos) < euclidean_dist(polygon_env.original_pos, polygon_env.goal_pos) +0.05 and \
			abs(curr_angle - polygon_env.goal_angle) < abs(vertex.angle - polygon_env.goal_angle) + 0.05 and \
			abs(curr_angle - polygon_env.goal_angle) < abs(polygon_env.goal_angle) + 0.05) and \
			((round(curr_pos[0], 2), round(curr_pos[1], 2), round(curr_angle, 2)) not in seen)) or vertex.timestamp == 0:
			seen.add((round(curr_pos[0], 2), round(curr_pos[1], 2), round(curr_angle, 2)))
			for act in polygon_env.actions:
				queue.append(Vertex(curr_pos, curr_angle, act, vertex.timestamp+1, vertex))
	return ending_vertex

def get_steps(ending_vertex):
	"""backtrack steps from ending vertex"""
	final_acts = []
	curr_vertex = ending_vertex
	while curr_vertex != None:
		final_acts.append(curr_vertex.action)
		curr_vertex = curr_vertex.parent
	return final_acts

def main():
	env = PolygonEnv((8,6), [(-1,0),(-1/2, 1),(1/2,1),(1,0),(0,-1)], (3, 3), 2)
	end = modified_bfs_planner(env)
	act_list = get_steps(end)
	env.animate(act_list)
