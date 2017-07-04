import tensorflow as tf
from tensorflow.contrib.keras.api.keras.layers import Dense
from tensorflow.contrib.keras.api.keras.losses import mean_squared_error

sess = tf.Session()

input_layer = tf.placeholder(tf.float32, shape=(None, 2), name='input')

hidden = Dense(4, activation='sigmoid', name='hidden')(input_layer)
prediction = Dense(2, activation='sigmoid', name='prediction')(hidden)

expected = tf.placeholder(tf.float32, shape=(None, 2), name='expected')
loss = tf.reduce_mean(mean_squared_error(expected, prediction))

global_step = tf.Variable(0, name='global_step', trainable=False)
# train_step = tf.train.AdagradOptimizer(0.2).minimize(loss, global_step=global_step)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss, global_step=global_step)

tf.summary.scalar('loss', loss)
tf.summary.histogram('Hidden', hidden)
tf.summary.histogram('Prediction', prediction)
merged = tf.summary.merge_all()

INPUT_DATA = [[0, 0], [0, 1], [1, 0], [1, 1]]
EXPECTED_DATA = [[1, 0], [0, 0], [0, 0], [1, 0]]

init = tf.global_variables_initializer()

sess.run(init)

saver = tf.train.Saver()
saver.restore(sess, "xor.ckpt")

train_writer = tf.summary.FileWriter('./train', sess.graph)

for i in range(100000):
    summary, acc, error, out, e = sess.run([merged, train_step, loss, prediction, expected],
                                           feed_dict={input_layer: INPUT_DATA, expected: EXPECTED_DATA})
    train_writer.add_summary(summary, i)
    if (i % 1000 == 0):
        print('Epoch ', i)
        print('Error ', error)
        save_path = saver.save(sess, "xor.ckpt")
        print("Prediction", out)
        print("Expected", e)
