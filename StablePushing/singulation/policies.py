import math
import numpy as np
from helper import *

####################
# Helper functions #
####################

def find_best_remove_object(env):
	# select object to push
	push_obj = -1
	max_dist_sum = 0
	for obj in range(len(env.objs)):
		dist_sum = 0
		for i in range(len(env.objs) - 1):
			for j in range(i + 1, len(env.objs)):
				if i != obj and j != obj:
					dist_sum += euclidean_dist(env.objs[i].original_pos, env.objs[j].original_pos)
		if dist_sum > max_dist_sum:
			push_obj = obj
			max_dist_sum = dist_sum

	return push_obj

def find_closest_pair(env, removal_lst):
	# list can be empty
	min_dist = 100
	pair = []
	for i in range(len(env.objs) - 1):
		for j in range(i + 1, len(env.objs)):
			if (not i in removal_lst) and (not j in removal_lst):
				if euclidean_dist(env.objs[i].original_pos, env.objs[j].original_pos) < min_dist:
					min_dist = euclidean_dist(env.objs[i].original_pos, env.objs[j].original_pos)
					pair = [i, j]
	return pair

def find_closest_ranking_to_object(env, obj):
	dic = {}
	for i in range(len(env.objs)):
		if i != obj:
			dic[i] = euclidean_dist(env.objs[i].original_pos, env.objs[obj].original_pos)
	return [k for k in sorted(dic, key=dic.get)]

def find_clusters(env, cluster_num):
	if cluster_num == 1:
		return [list(range(len(env.objs)))]

	first_obj = 0
	for obj in range(len(env.objs)):
		if env.objs[obj].original_pos[0] < env.objs[first_obj].original_pos[0]:
			first_obj = obj

	cluster_center = [first_obj]
	cluster_lst = []

	for i in range(cluster_num - 1):
		max_sum_dist = 0
		center_item = None
		for obj in range(len(env.objs)):
			if obj not in cluster_center:
				sum_dist = sum([euclidean_dist(env.objs[obj].original_pos, env.objs[c].original_pos) for c in cluster_center])
				if sum_dist > max_sum_dist:
					center_item = obj
					max_sum_dist = sum_dist
		if center_item is not None:
			cluster_center.append(center_item)

	for c in cluster_center:
		cluster_lst.append([c])

	for obj in range(len(env.objs)):
		if obj not in cluster_center:
			center = None
			min_dist = 1e2
			for c in cluster_center:
				if euclidean_dist(env.objs[obj].original_pos, env.objs[c].original_pos) < min_dist:
					center = c
					min_dist = euclidean_dist(env.objs[obj].original_pos, env.objs[c].original_pos)
			if center is not None:
				for lst in cluster_lst:
					if lst[0] == c:
						lst.append(obj)
	return cluster_lst

############
# Policies #
############

def proposed0(env):
	push_obj = find_best_remove_object(env)
	dist_lst = find_closest_ranking_to_object(env, push_obj)
	seg = np.array(env.objs[dist_lst[0]].original_pos) - np.array(env.objs[dist_lst[1]].original_pos)
	vector1 = (1, -(seg[0] / (seg[1]+1e-8)))
	vector2 = (-1, (seg[0] / (seg[1]+1e-8)))
	vector1 = normalize(vector1)
	vector2 = normalize(vector2)
	max_away = normalize(findMaxAwayVector([env.objs[push_obj].original_pos - env.objs[dist_lst[0]].original_pos, env.objs[push_obj].original_pos - env.objs[dist_lst[1]].original_pos]))

	# print(vector1, vector2, max_away, euclidean_dist(vector1, max_away), euclidean_dist(vector2, max_away))
	if euclidean_dist(vector1, max_away) < euclidean_dist(vector2, max_away):
		# print(1)
		pts = parametrize_by_bounding_circle(env.objs[push_obj].original_pos, vector1, env.objs[push_obj].original_pos, env.objs[push_obj].bounding_circle_radius+0.1)
	else:
		# print(2)
		pts = parametrize_by_bounding_circle(env.objs[push_obj].original_pos, vector2, env.objs[push_obj].original_pos, env.objs[push_obj].bounding_circle_radius+0.1)
	
	return pts

def proposed1(env):
	push_obj = find_best_remove_object(env)
	vertices = np.array(env.objs[push_obj].vertices) + np.array(env.objs[push_obj].original_pos)
	min_contact_range = 1e2
	push_vector = None
	for j in range(16):
		vector = (math.cos(2*j*3.14 / 16), math.sin(2*j*3.14 / 16))
		pts = parametrize_by_bounding_circle(env.objs[push_obj].original_pos, vector, env.objs[push_obj].original_pos, env.objs[push_obj].bounding_circle_radius+0.1)
		contact_range, range_pt_lst = find_max_contact_range(vertices, pts[0], pts[1])
		if contact_range != 0 and contact_range < min_contact_range:
			min_contact_range = contact_range
			push_vector = vector
	if not push_vector is None:
		dist_lst = find_closest_ranking_to_object(env, push_obj)
		vector1 = normalize(push_vector)
		vector2 = normalize((-push_vector[0], -push_vector[1]))
		max_away = normalize(findMaxAwayVector([env.objs[push_obj].original_pos - env.objs[dist_lst[i]].original_pos for i in range(len(dist_lst))]))
		if euclidean_dist(vector1, max_away) < euclidean_dist(vector2, max_away):
			pts = parametrize_by_bounding_circle(env.objs[push_obj].original_pos, vector1, env.objs[push_obj].original_pos, env.objs[push_obj].bounding_circle_radius+0.1)
		else:
			pts = parametrize_by_bounding_circle(env.objs[push_obj].original_pos, vector2, env.objs[push_obj].original_pos, env.objs[push_obj].bounding_circle_radius+0.1)
		return pts
	return None
	
def proposed2(env):
	push_obj = find_best_remove_object(env)
	max_dist_sum = 0
	push_pts = None
	for j in range(16):
		vector = (math.cos(2*j*3.14 / 16), math.sin(2*j*3.14 / 16))
		pts = parametrize_by_bounding_circle(env.objs[push_obj].original_pos, vector, env.objs[push_obj].original_pos, env.objs[push_obj].bounding_circle_radius+0.1)
		min_dist_l = 1e2
		min_dist_r = 1e2
		for k in range(len(env.objs)):
			if k != push_obj and scalarProject(pts[0], pts[1], env.objs[k].original_pos) > 0:
				side_com = side_of_point_on_line(pts[0], pts[1], env.objs[k].original_pos)
				if side_com < 0 and pointToLineDistance(pts[0], pts[1], env.objs[k].original_pos) - env.objs[k].bounding_circle_radius < min_dist_l:
					min_dist_l = pointToLineDistance(pts[0], pts[1], env.objs[k].original_pos) - env.objs[k].bounding_circle_radius
				if side_com > 0 and pointToLineDistance(pts[0], pts[1], env.objs[k].original_pos) - env.objs[k].bounding_circle_radius < min_dist_r:
					min_dist_r = pointToLineDistance(pts[0], pts[1], env.objs[k].original_pos) - env.objs[k].bounding_circle_radius
		if min_dist_l + min_dist_r > max_dist_sum:
			max_dist_sum = min_dist_l + min_dist_r
			push_pts = pts
	return push_pts

def boundaryShear(env):
	max_free_space = 0
	obj = None
	p = None
	v = None
	candidate_pair = find_closest_pair(env, [])
	linking_line = np.array(env.objs[candidate_pair[0]].original_pos) - np.array(env.objs[candidate_pair[1]].original_pos)
	vector_lst = [(1, -(linking_line[0] / (linking_line[1]+1e-8))), (-1, (linking_line[0] / (linking_line[1]+1e-8)))]
	for candidate in candidate_pair:
		for vt in vector_lst:
			free_space = find_free_space(np.array(env.objs[candidate].original_pos), vt, env.objs)
			if free_space > max_free_space:
				max_free_space = free_space
				p = np.array(env.objs[candidate].original_pos)
				v = vt
				obj = candidate
	if (not p is None) and (not v is None):
		pts = parametrize_by_bounding_circle(p, v, env.objs[obj].original_pos, env.objs[obj].bounding_circle_radius+0.1)
		return pts
	return None

def clusterDiffusion(env):
	cluster_num = (len(env.objs)-1) // 3 + 1
	cluster_lst = find_clusters(env, cluster_num)
	cluster = cluster_lst[np.argmax([len(lst) for lst in cluster_lst])]
	vertices_lst = []
	for obj in cluster:
		vertices_lst.extend((np.array(env.objs[obj].vertices)+np.array(env.objs[obj].original_pos)).tolist())
	cluster_center = compute_centroid(create_convex_hull(np.array(vertices_lst)))

	push_pts = None

	min_dist = 1e2

	for obj in cluster:
		vector = normalize(env.objs[obj].original_pos - np.array(cluster_center))
		max_away = normalize(findMaxAwayVector([env.objs[obj].original_pos - env.objs[o].original_pos for o in cluster if o != obj]))
		if euclidean_dist(vector, max_away) < min_dist:
			min_dist = euclidean_dist(vector, max_away)
			push_pts = parametrize_by_bounding_circle(env.objs[obj].original_pos, vector, env.objs[obj].original_pos, env.objs[obj].bounding_circle_radius+0.1)
	
	return push_pts

def maximumClearanceRatio(env):
	p_lst = [np.array(obj.original_pos) for obj in env.objs]
	v_lst = [(math.cos(i * 3.14* 2 / 16), math.sin(i * 3.14* 2 / 16)) for i in range(16)]
	max_free_space = 0
	p = None
	v = None
	candidate = None
	for obj_ind in range(len(env.objs)):
		for vt in v_lst:
			free_space = find_free_space(np.array(env.objs[obj_ind].original_pos), vt, env.objs) / find_free_space(np.array(env.objs[obj_ind].original_pos), (-vt[0], -vt[1]), env.objs)
			if free_space > max_free_space:
				max_free_space = free_space
				p = np.array(env.objs[obj_ind].original_pos)
				v = vt
				candidate = obj_ind
	if (not p is None) and (not v is None):
		pts = parametrize_by_bounding_circle(p, v, env.objs[candidate].original_pos, env.objs[candidate].bounding_circle_radius+0.1)
	return pts




