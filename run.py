import tensorflow as tf

x_ = tf.placeholder(tf.float32, shape=[4,2], name='x-input')
y_ = tf.placeholder(tf.float32, shape=[4,2], name='y-input')

Theta1 = tf.Variable(tf.zeros([2,2]), name='Theta1')
Theta2 = tf.Variable(tf.zeros([2,2]), name='Theta2')

Bias1 = tf.Variable(tf.zeros([2]), name='Bias1')
Bias2 = tf.Variable(tf.zeros([2]), name='Bias2')

A2 = tf.sigmoid(tf.matmul(x_, Theta1) + Bias1)
Hypothesis = tf.sigmoid(tf.matmul(A2, Theta2) + Bias2)

XOR_X = [[0,0], [0,1], [1,0], [1,1]]
XOR_Y = [[0,1], [1,0], [1,0], [0,1]]

sess = tf.Session()

saver = tf.train.Saver()
saver.restore(sess, "xor.ckpt")

result = sess.run(Hypothesis, feed_dict={x_: XOR_X, y_: XOR_Y})

print(result)
