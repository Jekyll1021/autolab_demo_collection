import numpy as np
import tensorflow as tf
from TestEnv import *

def weight_variable(shape, name=''):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name=''):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x, name=''):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

class ImageFittingModel:
	def __init__(self, iterations=1000, learning_rate=5e-4, sess=tf.Session()):
		self.sess = sess
		self.iterations = iterations
		self.learning_rate = learning_rate

		self.x = tf.placeholder(tf.float32, [None, 240*180*3], name="x")
		self.y_ = tf.placeholder(tf.float32, [None, 4], name="y_")

		# x_image = tf.reshape(self.x, [-1,240,180,3])

		# W_conv1 = weight_variable([21, 21, 3, 32], name='weight1')
		# b_conv1 = bias_variable([32], name='bias1')
		# h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
		# h_pool1 = max_pool_2x2(h_conv1)

		# W_conv2 = weight_variable([11, 11, 32, 64], name='weight2')
		# b_conv2 = bias_variable([64], name='bias2')
		# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
		# h_pool2 = max_pool_2x2(h_conv2)

		# h_pool2_flat = tf.reshape(h_pool2, [-1, 172800])
		W_fc1 = weight_variable([240*180*3, 64], name='weight3')
		b_fc1 = bias_variable([64], name='bias3')
		h_fc1 = tf.nn.relu(tf.matmul(self.x, W_fc1) + b_fc1)

		# h_fc1 = tf.nn.dropout(h_fc1, 0.5)

		W_fc2 = weight_variable([64, 64], name='weight4')
		b_fc2 = bias_variable([64], name='bias4')
		h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

		W_fc3 = weight_variable([64, 4], name='weight5')
		b_fc3 = bias_variable([4], name='bias5')
		self.y = tf.matmul(h_fc2, W_fc3) + b_fc3

		self.loss = tf.reduce_mean((self.y_ - self.y)**2)
		self.update_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

		self.saver = tf.train.Saver()

	def fit(self, images, actions):
		_, loss = self.sess.run([self.update_op, self.loss], feed_dict={self.x:images, self.y_:actions})
		return loss

	def predict(self, image):
		return self.sess.run(self.y, feed_dict={self.x:[image]})[0]

	def save(self, path):
		self.saver.save(self.sess, path)

	def load(self, path):
		self.saver.restore(self.sess, path)

def train(iterations=1000, learning_rate=5e-4):
	sess = tf.Session()
	model = ImageFittingModel(iterations, learning_rate, sess)
	sess.__enter__()
	# tf.global_variables_initializer().run()
	model.load("model/fc.ckpt")

	images = np.load("image_data_0.npy")
	actions = np.load("action_data_0.npy")
	min_loss = 0.00537804
	for i in range(model.iterations):
		ind = np.random.choice(1000, 1000, replace=False)
		images = images[ind]
		actions = actions[ind]
		loss = []
		for j in range(5):
			loss.append(model.fit(images[j*200:(j+1)*200], actions[j*200:(j+1)*200]))
		print("iter " +str(i)+": "+ str(np.mean(loss))+ "; min loss: "+ str(min_loss))
		if np.mean(loss) < min_loss:
			model.save("model/fc.ckpt")
			min_loss = np.mean(loss)

def evaluate():
	sess = tf.Session()
	model = ImageFittingModel(sess=sess)
	sess.__enter__()
	# tf.global_variables_initializer().run()
	model.load("model/fc.ckpt")
	env = TestEnv()
	obs = env.get_image()
	total_rew = 0
	while not env.done:
		if env.timesteps % 100 == 0:
			env.save_image("image" + str(1000 - env.timesteps) +".png")
		act = model.predict(obs)
		print(act)
		state, rew, done, obs = env.step(act)
		total_rew += rew
	env.reset()
	return np.mean(total_rew)




if __name__ == "__main__":
	# train()
	print(evaluate())
