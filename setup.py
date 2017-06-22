import tensorflow as tf

x_ = tf.placeholder(tf.float32, shape=[4,2], name='x-input')
y_ = tf.placeholder(tf.float32, shape=[4,2], name='y-input')

Theta1 = tf.Variable(tf.random_uniform([2,2], -1, 1), name='Theta1')
Theta2 = tf.Variable(tf.random_uniform([2,2], -1, 1), name='Theta2')

Bias1 = tf.Variable(tf.zeros([2]), name='Bias1')
Bias2 = tf.Variable(tf.zeros([2]), name='Bias2')

A2 = tf.sigmoid(tf.matmul(x_, Theta1) + Bias1)
Hypothesis = tf.sigmoid(tf.matmul(A2, Theta2) + Bias2)

cost = tf.reduce_mean(((y_ * tf.log(Hypothesis)) + ((1 - y_) * tf.log(1.0 - Hypothesis))) * -1)

global_step = tf.Variable(0, name='global_step', trainable=False)
train_step = tf.train.GradientDescentOptimizer(0.02).minimize(cost, global_step=global_step)

tf.summary.scalar('cost', cost)
tf.summary.histogram('Hypothesis', Hypothesis)
merged = tf.summary.merge_all()

XOR_X = [[0,0], [0,1], [1,0], [1,1]]
XOR_Y = [[0,1], [1,0], [1,0], [0,1]]

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

saver = tf.train.Saver()
train_writer = tf.summary.FileWriter('./train', sess.graph)

#for i in range(10000):
#	summary, acc = sess.run([merged, train_step], feed_dict={x_: XOR_X, y_: XOR_Y})
#	train_writer.add_summary(summary, i)
#	if (i % 1000 == 0):
#		print('Epoch ', i)

save_path = saver.save(sess, "xor.ckpt")
print("Model saved as %s" % save_path);
