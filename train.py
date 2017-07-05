import csv
import tensorflow as tf
import numpy as np
import os.path
from tensorflow.contrib.keras.api.keras.layers import Dense, GaussianNoise
from tensorflow.contrib.keras.api.keras.losses import mean_squared_error
from tensorflow.contrib.keras import backend as K

sess = tf.Session()
K.set_session(sess)
K.set_learning_phase(1)

with open("legs.csv", "r") as csvFile:
    x = csv.reader(csvFile)
    data = list(x)
data = np.delete(data, [0, 1], axis=1)

raw_legs = tf.convert_to_tensor(data, dtype=tf.float32, name='data')

to_float = [1/12, 1/31, 1/(24*60), 1/(24*60), 1/(0.5*60), 1/(0.5*60), 1/(0.5*60), 1/(0.5*60)]
from_float = [0.5*60]

legs = tf.multiply(raw_legs, to_float)

input_layer = tf.placeholder(tf.float32, shape=(None, 7), name='input')
noise = GaussianNoise(0.02, dtype=tf.float32)(input_layer)
hidden = Dense(14, activation='tanh', name='hidden')(noise)
prediction = Dense(1, activation='tanh', name='prediction')(hidden)

expected = tf.placeholder(tf.float32, shape=(None, 1), name='expected')
loss = tf.reduce_mean(mean_squared_error(expected, prediction))

global_step = tf.Variable(0, name='global_step', trainable=False)
train_step = tf.train.AdagradOptimizer(0.2).minimize(loss, global_step=global_step)
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss, global_step=global_step)


prediction_output = tf.multiply(prediction, from_float)
expected_output = tf.multiply(expected, from_float)

tf.summary.scalar('Loss', loss)
tf.summary.histogram('Hidden', hidden)
tf.summary.histogram('Prediction', prediction)
tf.summary.histogram('Expected', expected)
merged = tf.summary.merge_all()

init = tf.global_variables_initializer()

sess.run(init)

saver = tf.train.Saver()
if os.path.exists('xor.ckpt.meta'):
    print('loading from xor.ckpt')
    saver.restore(sess, "xor.ckpt")

train_writer = tf.summary.FileWriter('./train', sess.graph)

data = sess.run(legs)

train_data = data[:len(data) - 20, :]
test_data = data[-20:, :]

input_train_data = train_data[:, 0:7]
output_train_data = train_data[:, [7]]

input_test_data = test_data[:, 0:7]
output_test_data = test_data[:, [7]]

for i in range(1000000):
    sess.run(train_step, feed_dict={input_layer: input_train_data, expected: output_train_data})
    if i % 5000 == 0:
        step, summary, train_error, out, e = sess.run([global_step, merged, loss, prediction, expected],
                                                feed_dict={input_layer: input_train_data, expected: output_train_data})
        test_error, out, e = sess.run([loss, prediction, expected],
                                      feed_dict={input_layer: input_test_data, expected: output_test_data})
        train_writer.add_summary(summary, step)
        print('Epoch ', step)
        print('Train Error ', train_error)
        save_path = saver.save(sess, "xor.ckpt")
        print('Test Error ', test_error)
        foo1, foo2 = sess.run([prediction_output, expected_output],
                              feed_dict={input_layer: input_test_data, expected: output_test_data})
        foo3 = np.subtract(foo2, foo1)
        foo4 = np.multiply(foo3, from_float)
        print(foo4)
        # print("Prediction", out)
        # print("Expected", e)
