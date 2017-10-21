import numpy as np
import tensorflow as tf
from TestEnv import *

def main():
	x = tf.placeholder(shape=[None, 240 * 180 * 3], name="image", dtype=tf.float32)
	y = tf.placeholder(shape=[None, 4], name="act", dtype=tf.float32)

	W_fc0 = tf.Variable(tf.truncated_normal([240 * 180 * 3, 2048], stddev=0.1), name='weight1')
	b_fc0 = tf.Variable(tf.truncated_normal([2048], stddev=0.1), name='bias1')
	h_fc0 = tf.nn.relu(tf.matmul(x, W_fc0) + b_fc0)

	W_fc1 = tf.Variable(tf.truncated_normal([2048, 1024], stddev=0.1), name='weight3')
	b_fc1 = tf.Variable(tf.truncated_normal([1024], stddev=0.1), name='bias3')
	h_fc1 = tf.nn.relu(tf.matmul(h_fc0, W_fc1) + b_fc1)

	W_fc2 = tf.Variable(tf.truncated_normal([1024, 4], stddev=0.1), name='weight4')
	b_fc2 = tf.Variable(tf.truncated_normal([4], stddev=0.1), name='bias4')
	y_ = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

	loss = tf.reduce_sum(tf.losses.mean_squared_error(y, y_))
	update_op = tf.train.AdamOptimizer().minimize(loss)

	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		image_data = np.load("image_data_0.npy")
		action_data = np.load("action_data_0.npy")

		for i in range(50):
			ind = np.random.choice(1000, 100, replace=False)
			images = image_data[ind]
			print(images.shape)
			actions = action_data[ind]
			_ = sess.run(update_op, feed_dict={x:images, y:actions})
			# if i % 5 == 0:
			# 	loss = sess.run(loss, feed_dict={x:images, y:actions})
			# 	print(loss)

		env = TestEnv()
		obs = env.get_image()
		rews = []
		total_rew = 0
		while not env.done:
			if env.timesteps % 100 == 1:
				env.save_image("image" + str(1000 - env.timesteps) +".png")
			act = sess.run(y_, feed_dict={x:[obs]})[0]
			print(act)
			state, rew, done, obs = env.step(act)
			total_rew += rew
			rews.append(total_rew)
		env.reset()
	return np.mean(rews)

if __name__ == "__main__":
	print(main())