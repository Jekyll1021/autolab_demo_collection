import numpy as np
import tensorflow as tf
import random
from planner_fitting_model import *

def train(first_time=True, iterations=100000, batchsize=512, path="/Zisu/data/"):
	sess = tf.Session()
	model = PlannerFittingModel(sess)
	sess.__enter__()
	tf.global_variables_initializer().run()
	
	for i in range(iterations):
		ind = random.randint(0, 49)
		state = np.load(path+"state_data_"+ str(ind)+".npy")
		goal_state = np.load(path+"goal_state_data_"+ str(ind)+".npy")
		a0 = np.load(path+"actions_data_"+ str(ind)+"_0.npy")
		a1 = np.load(path+"actions_data_"+ str(ind)+"_1.npy")
		a2 = np.load(path+"actions_data_"+ str(ind)+"_2.npy")

		ind_list = np.random.choice(len(state), batchsize, replace=False)

		min_loss = 1e5

		loss = model.fit(state[ind_list], goal_state[ind_list], a0[ind_list], a1[ind_list], a2[ind_list])
		if loss < min_loss:
			min_loss = loss
			model.save("/Zisu/model/model1.ckpt")

		if (i + 1) % 1000 == 0:
			eval_loss = model.evaluate(state, goal_state, a0, a1, a2)
			print("Iter "+str(i)+" loss: %.5f" % (eval_loss))

	sess.close()


