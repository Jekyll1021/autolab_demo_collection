import collections
import sys
import random
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import math

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

def modified_bfs_planner(polygon_env, horizon=25):
	queue = collections.deque()
	for a in polygon_env.actions:
		curr_pos, curr_angle = polygon_env.step(polygon_env.original_pos, 0, a)
		if euclidean_dist(curr_pos, polygon_env.goal_pos) < euclidean_dist(polygon_env.original_pos, polygon_env.goal_pos) + 0.05 and \
		abs(curr_angle - polygon_env.goal_angle) < abs(polygon_env.goal_angle) + 0.05: 
			queue.append(Vertex(polygon_env.original_pos, 0, a, 0, None))
	ending_vertex = None
	seen = set()
	while queue:
		vertex = queue.popleft()
		# print(vertex.timestamp)
		curr_pos, curr_angle = polygon_env.step(vertex.pos, vertex.angle, vertex.action)
		if ending_vertex == None or \
		10*euclidean_dist(curr_pos, polygon_env.goal_pos) + abs(curr_angle - polygon_env.goal_angle) < \
		10*euclidean_dist(ending_vertex.pos, polygon_env.goal_pos) + abs(ending_vertex.angle - polygon_env.goal_angle):
			ending_vertex = vertex
		if (euclidean_dist(curr_pos, polygon_env.goal_pos) < INTERVAL and abs(curr_angle - polygon_env.goal_angle) < INTERVAL):
			# print('wha')
			break
		if (vertex.timestamp > 0 and vertex.timestamp < horizon and \
			(euclidean_dist(curr_pos, polygon_env.goal_pos) < euclidean_dist(vertex.pos, polygon_env.goal_pos) + 0.05 and \
			euclidean_dist(curr_pos, polygon_env.goal_pos) < euclidean_dist(polygon_env.original_pos, polygon_env.goal_pos) + 0.05 and \
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

def random_matching(polygon_env, horizon=30):
	"""get random matching result scatter plot for one polygon"""
	curr_pos = polygon_env.original_pos
	curr_angle = 0
	for i in range(horizon):
		a = Action((random.random()*2-1, random.random()*2-1), (random.random() * 2 -1, random.random() * 2 -1))
		curr_pos, curr_angle = polygon_env.step(curr_pos, curr_angle, a)
	match_env = PolygonEnv(polygon_env.original_pos, polygon_env.vertices, curr_pos, curr_angle - round(curr_angle / math.pi) * math.pi)
	end = modified_bfs_planner(match_env)
	end_pos, end_angle = match_env.step(end.pos, end.angle, end.action)
	# print((euclidean_dist(end_pos, match_env.goal_pos) < INTERVAL and abs(end_angle - match_env.goal_angle) < INTERVAL))
	return (curr_pos, curr_angle, (euclidean_dist(end_pos, match_env.goal_pos) < INTERVAL and abs(end_angle - match_env.goal_angle) < INTERVAL))

def generate_data(polygon_vertices_list, batch=1000):
	obs = []
	goal_obs = []
	acts = []
	bounding_circle_acts = []
	bounding_circle_normalized_acts = []
	while len(obs) < batch:
		print(len(obs))
		curr_original_pos = (random.random()*8 + 1, random.random()*4 + 1)
		curr_goal_pos = (random.random()*8 + 1, random.random()*4 + 1)
		curr_goal_angle = random.random()*4 % math.pi
		curr_polygon_vertices = random.choice(polygon_vertices_list)
		env = PolygonEnv(curr_original_pos, curr_polygon_vertices, curr_goal_pos, curr_goal_angle)
		end = modified_bfs_planner(env)
		act_list = get_steps(end)
		curr_obs, curr_g_obs, curr_acts, curr_b_acts, curr_bn_acts = env.animate(act_list)
		obs.extend(curr_obs)
		goal_obs.extend(curr_g_obs)
		acts.extend(curr_acts)
		bounding_circle_acts.extend(curr_b_acts)
		bounding_circle_normalized_acts.extend(curr_bn_acts)
	return np.array(obs), np.array(goal_obs), np.array(acts), np.array(bounding_circle_acts), np.array(bounding_circle_normalized_acts)

def generate_symmetric_data(batch=1000):
	obs = []
	goal_obs = []
	acts = []
	bounding_circle_acts = []
	bounding_circle_normalized_acts = []
	while len(obs) < batch:
		print(len(obs))
		n = random.randint(3, 6)
		curr_original_pos = (random.random()*8 + 1, random.random()*4 + 1)
		curr_goal_pos = (random.random()*8 + 1, random.random()*4 + 1)
		curr_goal_angle = random.random()*4 % (2*math.pi/n)
		curr_polygon_vertices = get_regular_polygon_vertices(n)
		env = PolygonEnv(curr_original_pos, curr_polygon_vertices, curr_goal_pos, curr_goal_angle)
		end = modified_bfs_planner(env)
		act_list = get_steps(end)
		curr_obs, curr_g_obs, curr_acts, curr_b_acts, curr_bn_acts = env.animate(act_list)
		obs.extend(curr_obs)
		goal_obs.extend(curr_g_obs)
		acts.extend(curr_acts)
		bounding_circle_acts.extend(curr_b_acts)
		bounding_circle_normalized_acts.extend(curr_bn_acts)
	return np.array(obs), np.array(goal_obs), np.array(acts), np.array(bounding_circle_acts), np.array(bounding_circle_normalized_acts)


def get_regular_polygon_vertices(n):
	lst = []
	r = (random.random() + 1)/2
	for i in range(0, n):
		lst.append((r * math.cos(2*math.pi*i/n), r * math.sin(2*math.pi*i/n)))
	return lst

def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--start', type=int, default=0)
	parser.add_argument('--end', type=int, default=100)
	args = parser.parse_args()
	# polygon_vertices_list = [[(-1,0),(-1/2, 1),(1/2,1),(1,0),(0,-1)], [(-1,-1),(-1,1),(1,1),(1,-1)], [(-1,1),(1,1),(1,-1)]]
	for i in range(args.start, args.end):
		print(i)
		obs, goal_obs, acts, b_acts, bn_acts = generate_symmetric_data()
		np.save("sym_data/state_data_"+ str(i)+".npy", obs)
		np.save("sym_data/goal_state_data_"+ str(i)+".npy", goal_obs)
		np.save("sym_data/actions_data_"+ str(i)+".npy", acts)
		np.save("sym_data/bounding_circle_actions_data_"+ str(i)+".npy", b_acts)
		np.save("sym_data/bounding_circle_normalized_actions_data_"+ str(i)+".npy", bn_acts)
	# env = PolygonEnv((8,6), [(-1,0),(-1/2, 1),(1/2,1),(1,0),(0,-1)], (3, 3), 2)
	# end = modified_bfs_planner(env)
	# act_list = get_steps(end)
	# env.animate(act_list)

# def plot_reachability(polygon_env):
# 	xs_r = []
# 	ys_r = []
# 	zs_r = []
# 	xs_notr = []
# 	ys_notr = []
# 	zs_notr = []

# 	for i in range(100):
# 		print(i)
# 		r = random_matching(polygon_env)
# 		if r[2]:
# 			xs_r.append(r[0][0])
# 			ys_r.append(r[0][1])
# 			zs_r.append(r[1])
# 		else:
# 			xs_notr.append(r[0][0])
# 			ys_notr.append(r[0][1])
# 			zs_notr.append(r[1])
# 	# return xs_r, ys_r, zs_r, xs_notr, ys_notr, zs_notr

# 	fig = plt.figure()
# 	ax = fig.add_subplot(111, projection='3d')
# 	ax.scatter(xs_r, ys_r, zs_r, c='b')
# 	ax.scatter(xs_notr, ys_notr, zs_notr, c='r')

# 	ax.set_xlabel('x coor')
# 	ax.set_ylabel('y coor')
# 	ax.set_zlabel('theta')
# 	fig.savefig('plot.png')
if __name__ == "__main__":
    main()
