import numpy as np
import tensorflow as tf


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
	def __init__(self, iterations, learning_rate, sess):
		self.sess = sess
		self.iterations = iterations
		self.learning_rate = learning_rate

		self.x = tf.placeholder(tf.float32, [None, 240*180*3], name="x")
		self.y_ = tf.placeholder(tf.float32, [None, 4], name="y_")

		x_image = tf.reshape(self.x, [-1,240,180,3])

		W_conv1 = weight_variable([11, 11, 3, 32], name='weight1')
		b_conv1 = bias_variable([32], name='bias1')
		h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
		h_pool1 = max_pool_2x2(h_conv1)

		W_conv2 = weight_variable([5, 5, 32, 64], name='weight2')
		b_conv2 = bias_variable([64], name='bias2')
		h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
		h_pool2 = max_pool_2x2(h_conv2)

		h_pool2_flat = tf.reshape(h_pool2, [-1, 86400 * 2])
		W_fc1 = weight_variable([86400 * 2, 1024], name='weight3')
		b_fc1 = bias_variable([1024], name='bias3')
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

		h_fc1 = tf.nn.dropout(h_fc1, 0.5)

		W_fc2 = weight_variable([1024, 512], name='weight4')
		b_fc2 = bias_variable([512], name='bias4')
		h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

		W_fc3 = weight_variable([512, 4], name='weight5')
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

def train(iterations=1000, learning_rate=0.005):
	sess = tf.Session()
	model = ImageFittingModel(iterations, learning_rate, sess)
	sess.__enter__()
	# tf.global_variables_initializer().run()
	model.load("model/conv_model.ckpt")

	images = np.load("image_data_0.npy")
	actions = np.load("action_data_0.npy")
	for i in range(model.iterations):
		ind = np.random.choice(1000, 1000, replace=False)
		images = images[ind]
		actions = actions[ind]
		loss = []
		for j in range(5):
			loss.append(model.fit(images[j*200:(j+1)*200], actions[j*200:(j+1)*200]))
		print("iter " +str(i)+": "+ str(np.mean(loss)))
		if i % 10 == 0:
			model.save("model/conv_model.ckpt")




if __name__ == "__main__":
	train()
