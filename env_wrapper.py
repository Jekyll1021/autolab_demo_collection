#!/usr/bin/env python
import rospy, pickle, time
from utils.robot import *
from utils.constants import *

from geometry_msgs.msg import Pose
import numpy as np
import PyKDL
import multiprocessing
import tfx
import datetime

import cv_bridge
import cv2
from sensor_msgs.msg import Image, CameraInfo

class Environment:
	def __init__(self, env_time, rewarding_rules = None):
		"""initialize an environment with 
			1. robot right arm
			2. robot left arm(so far)

			also with the following parameters:
			1. env_time: a goal time interval that this environment exists
			2. time_counter: to keep on track with current time.
		"""
		self.psmR = robot("PSM1")
		self.psmL = robot("PSM2")
		self.env_time = env_time
		self.time_counter = 0
		self.rewarding_rules = rewarding_rules
		self.state = self.reset()
		# TODO: implement self.action_space and self.observation_space that defined as
		# shape of action_space and shape of observation_space


	def reset(self):
		"""get back to default position and return the default position as observation"""
		rot = tfx.tb_angles(INIT_POS[2][0], INIT_POS[2][1], INIT_POS[2][2])
		self.psmR.move_cartesian_frame(tfx.pose(INIT_POS[1], rot))
		return [INIT_POS[1], INIT_POS[2]]
		# TODO: reset apply to the platform and camera also

	def step(self, arm, action):
		"""it now takes in a list/tuple of [position, rotation], execute the action(move arm to indicated pos, rot)
		and returns:"""
		"""
		observation: object --- an environment-specific object representing your observation of the environment; 
			in this env new list of pos, rot
		reward: float --- amount of reward achieved by the previous action. 
			The scale varies between environments, and can also be customized using BLANK method.
		done: boolean --- whether it's time to reset the environment again. 
			Most tasks are divided up into well-defined episodes, and done being True indicates the episode has terminated. 
		info: dict --- value TBD, naive thoughts are to include info for next step
		"""
		# TODO: make more complicating actions
		pos, rot = action
		t1, t2, t3 = pos
		# observation = action
		# TODO: check if out of range
		# Take action
		obs_pos = arm.get_current_cartesian_position().position
		obs_rot = tfx.tb_angles(arm.get_current_cartesian_position().rotation)
		for i in range(0, 3):
			pos[i] += obs_pos[i]
		r1 = obs_rot.yaw_rad
		r2 = obs_rot.pitch_rad
		r3 = obs_rot.roll_rad
		new_rot = tfx.tb_angles(r1 + rot[0], r2 + rot[1], r3 + rot[2], rad = True)
		observation = [pos, new_rot]
		self.psmR.move_cartesian_frame(tfx.pose(pos, new_rot))
		self.time_counter += 1
		# Get reward
		reward = self.reward(self.state, action)
		# Determine done
		done = self.time_counter >= self.env_time
		# TODO: implement info
		info = {}
		return [observation, reward, done, info]

	def reward(self, state = None, action = None):
		"""we define rewarding rules as a customized function that maps (state, action) pair to float."""
		if self.rewarding_rules:
			return self.rewarding_rules(state, action)
		else:
			return 0