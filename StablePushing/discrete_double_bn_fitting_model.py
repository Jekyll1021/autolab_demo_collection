import numpy as np
import tensorflow as tf
import random

def weight_variable(shape, name=''):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name=name)

def bias_variable(shape, name=''):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, name=name)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x, name=''):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def max_pool_3x3(x, name=''):
	return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')

class DiscreteDoubleBnFittingModel:
	def __init__(self, learning_rate=1e-5, sess=tf.Session()):
		self.sess = sess
		self.learning_rate = learning_rate

		# placeholder definition
		self.state = tf.placeholder(tf.float32, [None, 240*180*3], name="curr_state")
		self.goal_state = tf.placeholder(tf.float32, [None, 240*180*3], name="goal_state")
		self.a0 = tf.placeholder(tf.float32, [None, 2], name="a0")
		self.a1 = tf.placeholder(tf.float32, [None, 2], name="a1")
		self.a2 = tf.placeholder(tf.float32, [None, 2], name="a2")

		image1 = tf.reshape(self.state, [-1, 240, 180, 3])
		image2 = tf.reshape(self.goal_state, [-1, 240, 180, 3])

		with tf.variable_scope("conv1") as scope:
		# Variables created here will be named "conv1/weights", "conv1/biases".
			W_conv1 = weight_variable([11, 11, 3, 64], name='weight')
			b_conv1 = bias_variable([64], name='bias')
			conv1_pool1 = max_pool_3x3(tf.nn.relu(conv2d(image1, W_conv1) + b_conv1))
			scope.reuse_variables()
			conv1_pool2 = max_pool_3x3(tf.nn.relu(conv2d(image2, W_conv1) + b_conv1))

		with tf.variable_scope("conv2") as scope:
			# Variables created here will be named "conv2/weights", "conv2/biases".
			W_conv2 = weight_variable([5, 5, 64, 64], name='weight')
			b_conv2 = bias_variable([64], name='bias')
			conv2_pool1 = max_pool_2x2(tf.nn.relu(conv2d(conv1_pool1, W_conv2) + b_conv2))
			scope.reuse_variables()
			conv2_pool2 = max_pool_2x2(tf.nn.relu(conv2d(conv1_pool2, W_conv2) + b_conv2))

		with tf.variable_scope("conv3") as scope:
			# Variables created here will be named "conv3/weights", "conv3/biases".
			W_conv3 = weight_variable([3, 3, 64, 64], name='weight')
			b_conv3 = bias_variable([64], name='bias')
			conv3_pool1 = max_pool_2x2(tf.nn.relu(conv2d(conv2_pool1, W_conv3) + b_conv3))
			scope.reuse_variables()
			conv3_pool2 = max_pool_2x2(tf.nn.relu(conv2d(conv2_pool2, W_conv3) + b_conv3))

			result1 = tf.reshape(conv3_pool1, [-1, 2400*8])
			result2 = tf.reshape(conv3_pool2, [-1, 2400*8])


		# with tf.variable_scope("image_filter") as scope:
		# 	result1 = image_filter(image1)
		# 	# print(result1.get_shape())
		# 	scope.reuse_variables()
		# 	result2 = image_filter(image2)
		# 	# print(result2.get_shape())
			image_features = tf.stack([result1, result2], 1)
			image_features = tf.reshape(image_features, [-1, 4800*8])
			# print(image_features.get_shape())

		with tf.variable_scope("fc1"):
			weights1 = tf.get_variable("weights", [4800*8, 4096], initializer=tf.random_normal_initializer())
			biases1 = tf.get_variable("biases", [4096], initializer=tf.constant_initializer(0.0))
			fc1 = tf.nn.relu(tf.matmul(image_features, weights1) + biases1)

		with tf.variable_scope("fc2"):
			weights2 = tf.get_variable("weights", [4096, 1024], initializer=tf.random_normal_initializer())
			biases2 = tf.get_variable("biases", [1024], initializer=tf.constant_initializer(0.0))
			fc2 = tf.nn.relu(tf.matmul(fc1, weights2) + biases2)

		with tf.variable_scope("fc3"):
			weights3 = tf.get_variable("weights", [1024, 2], initializer=tf.random_normal_initializer())
			biases3 = tf.get_variable("biases", [2], initializer=tf.constant_initializer(0.0))
			self.y = tf.matmul(fc2, weights3) + biases3

		y1, y2 = tf.split(self.y, [1, 1], 1)
		a01, a02 = tf.split(self.a0, [1, 1], 1)
		a11, a12 = tf.split(self.a1, [1, 1], 1)
		a21, a22 = tf.split(self.a2, [1, 1], 1)

		loss_a0 = tf.sqrt(tf.minimum(tf.minimum(tf.abs(y1-a01), tf.abs(y1+8-a01)), tf.abs(y1-8-a01)) ** 2 + tf.minimum(tf.minimum(tf.abs(y2-a02), tf.abs(y2+8-a02)), tf.abs(y2-8-a02)) ** 2)
		loss_a1 = tf.sqrt(tf.minimum(tf.minimum(tf.abs(y1-a11), tf.abs(y1+8-a11)), tf.abs(y1-8-a11)) ** 2 + tf.minimum(tf.minimum(tf.abs(y2-a12), tf.abs(y2+8-a12)), tf.abs(y2-8-a12)) ** 2)
		loss_a2 = tf.sqrt(tf.minimum(tf.minimum(tf.abs(y1-a21), tf.abs(y1+8-a21)), tf.abs(y1-8-a21)) ** 2 + tf.minimum(tf.minimum(tf.abs(y2-a22), tf.abs(y2+8-a22)), tf.abs(y2-8-a22)) ** 2)

		# loss_a0 = tf.reduce_sum((self.y - self.a0)**2, 1)
		# # print(loss_a0.get_shape())
		# loss_a1 = tf.reduce_sum((self.y - self.a1)**2, 1)
		# loss_a2 = tf.reduce_sum((self.y - self.a2)**2, 1)

		self.loss_by_logits = tf.minimum(tf.minimum(loss_a0, loss_a1), loss_a2)
		# print(self.loss_by_logits.get_shape())

		self.loss = tf.reduce_mean(self.loss_by_logits)
		# self.loss = tf.reduce_mean(loss_a0)

		self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

		grads_and_vars = self.optimizer.compute_gradients(self.loss, tf.trainable_variables())

		# for g, v in grads_and_vars:
		# 	if g is not None:
		# 		print("****************this is variable*************")
		# 		print(v.get_shape())
		# 		print(v)
		# 		print("****************this is gradient*************")
		# 		print(g.get_shape())
		# 		print(g)

		
		self.update_op = self.optimizer.apply_gradients(grads_and_vars)

		self.saver = tf.train.Saver()

	def fit(self, state, goal_state, a0, a1, a2):
		_, loss = self.sess.run([self.update_op, self.loss], feed_dict={self.state:state, self.goal_state:goal_state, self.a0:a0, self.a1:a1, self.a2:a2})
		return loss

	def evaluate(self, state, goal_state, a0, a1, a2):
		return self.sess.run(self.loss, feed_dict={self.state:state, self.goal_state:goal_state, self.a0:a0, self.a1:a1, self.a2:a2})

	def predict(self, state, goal_state):
		return self.sess.run(self.y, feed_dict={self.state:[state], self.goal_state:[goal_state]})[0]

	def save(self, path):
		self.saver.save(self.sess, path)

	def load(self, path):
		self.saver.restore(self.sess, path)

def train(iterations=1000000, batchsize=128, path="param_data/"):
	tf_config = tf.ConfigProto() # fill in w/ custom config 
	tf_config.gpu_options.allow_growth = True
	sess = tf.Session(config=tf_config)
	model = DiscreteDoubleBnFittingModel(sess=sess)
	model.sess.__enter__()
	# tf.global_variables_initializer().run()
	model.load("model/discrete_double_bn_model1.ckpt")
	min_loss = 1e2
	for i in range(iterations):
		ind = random.randint(0, 199)
		
		state = np.load(path+"state_data_"+ str(ind)+".npy")
		ind_list = np.random.choice(len(state), batchsize, replace=False)
		state = state[ind_list]
		goal_state = np.load(path+"goal_state_data_"+ str(ind)+".npy")[ind_list]
		a0 = np.load(path+"0discrete_double_bn_actions_data_"+ str(ind)+".npy")[ind_list]
		a1 = np.load(path+"1discrete_double_bn_actions_data_"+ str(ind)+".npy")[ind_list]
		a2 = np.load(path+"2discrete_double_bn_actions_data_"+ str(ind)+".npy")[ind_list]

		loss = model.fit(state, goal_state, a0, a1, a2)
		# print(loss)
		if loss < min_loss:
			min_loss = loss
			
			model.save("model/discrete_double_bn_model1.ckpt")

		if (i + 1) % 100 == 0:
			print("Iter "+str(i)+" loss: %.5f" % (loss))

	sess.close()

if __name__ == "__main__":
	train()
