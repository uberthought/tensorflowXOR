import csv
import tensorflow as tf
import numpy as np
import os.path
import math

sess = tf.Session()

# load the raw data
with open("legs.csv", "r") as csvFile:
    x = csv.reader(csvFile)
    data = list(x)

# one hot encode departure and destination airports
dept = np.array(data)[:, [0]]
dest = np.array(data)[:, [1]]
all_airports = np.append(dept, dest)
airport_set = np.unique(all_airports)
dept_indexed = np.zeros(0)
dest_indexed = np.zeros(0)
for d in dept:
    i = np.where(airport_set == d)
    dept_indexed = np.append(dept_indexed, i)
for d in dest:
    i = np.where(airport_set == d)
    dest_indexed = np.append(dest_indexed, i)

dept_one_hot_calculation = tf.one_hot(dept_indexed, airport_set.size)
dest_one_hot_calculation = tf.one_hot(dest_indexed, airport_set.size)

dept_one_hot_result = sess.run(dept_one_hot_calculation)
dest_one_hot_result = sess.run(dest_one_hot_calculation)

data = np.concatenate((data, dept_one_hot_result), axis=1)
data = np.concatenate((data, dest_one_hot_result), axis=1)
data = np.delete(data, [0, 1], axis=1)

# convert data from strings to floats because the data was loaded as strings
raw_legs = tf.convert_to_tensor(data, dtype=tf.float32, name='data').eval(session=sess)

# on_times = raw_legs[:, 7]

# if IN greater than late_value, set to 1 otherwise set to -1 so we target flight that are greater than late_value
late_value = 30
in_times = raw_legs[:, 8]
for i in range(in_times.shape[0]):
    if in_times[i] >= late_value:
        in_times[i] = 1;
    else:
        in_times[i] = -1;
raw_legs[:, 8] = in_times

# delete the ON times
raw_legs = np.delete(raw_legs, [5, 6, 7], axis=1)


# build the vector to normalize the data to -1 to 1
to_float = np.ones(raw_legs[0].size)
to_float[0] = 1/12
to_float[1] = 1/31
to_float[2] = 1/24
to_float[3] = 1/60
to_float[4] = 1/np.max(data[:, 4].astype(np.float))
to_float[5] = 1

legs = tf.multiply(raw_legs, to_float)

# clamp the data to eliminate out of range values
clamped_legs = tf.clip_by_value(legs, -1, 1)
data = sess.run(clamped_legs)

# target column 5 which is the IN time in the cooked data
target_column = 5

# split the data into 80% training and 20% test data
test_data_size = int(data.shape[0] * 0.2)
train_data = data[:len(data) - test_data_size, :]
test_data = data[-test_data_size:, :]

input_train_data = np.delete(train_data, [target_column], axis=1)
output_train_data = train_data[:, [target_column]]

input_test_data = np.delete(test_data, [target_column], axis=1)
output_test_data = test_data[:, [target_column]]

# build the graph
input_layer = tf.placeholder(tf.float32, shape=(None, input_train_data[0].size), name='input')

stddev = tf.placeholder_with_default(0.0, [])
noise_generator = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=stddev, dtype=tf.float32)
noise = tf.add(input_layer, noise_generator)

# hidden_units1 = math.ceil(data[0].size / 8)
hidden_units1 = 16
keep_prob1 = tf.placeholder_with_default(1.0, [])
hidden1 = tf.layers.dense(inputs=noise, units=hidden_units1, activation=tf.nn.tanh)
dropout1 = tf.nn.dropout(hidden1, keep_prob1)

# hidden_units2 = math.ceil(data[0].size / 16)
hidden_units2 = 4
keep_prob2 = tf.placeholder_with_default(1.0, [])
hidden2 = tf.layers.dense(inputs=dropout1, units=hidden_units2, activation=tf.nn.tanh)
dropout2 = tf.nn.dropout(hidden2, keep_prob2)

prediction = tf.layers.dense(inputs=dropout2, units=1, activation=tf.nn.tanh)

expected = tf.placeholder(tf.float32, shape=(None, 1), name='expected')
train_loss = tf.reduce_mean(tf.losses.mean_squared_error(expected, prediction))
test_loss = tf.reduce_mean(tf.losses.mean_squared_error(expected, prediction))

global_step = tf.Variable(0, name='global_step', trainable=False)
learning_rate = tf.placeholder_with_default(0.2, [])
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(train_loss, global_step=global_step)
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss, global_step=global_step)

# send some stats to tensorboard
train_loss_summary = tf.summary.scalar('Train Loss', train_loss)
test_loss_summary = tf.summary.scalar('Test Loss', test_loss)
# tf.summary.histogram('Hidden1', hidden1)
prediction_summary = tf.summary.histogram('Prediction', prediction)
expected_summary = tf.summary.histogram('Expected', expected)
# merged = tf.summary.merge_all()

# initialize the graph
init = tf.global_variables_initializer()
sess.run(init)

# setup the saver so we can save the graph later
saver = tf.train.Saver()
if os.path.exists('xor.ckpt.meta'):
    print('loading from xor.ckpt')
    saver.restore(sess, "xor.ckpt")

# setup a writer to dump the stats for tensorgraph
train_writer = tf.summary.FileWriter('./train', sess.graph)

# dump the testing data into a variable for pca
# test_data_tensor = tf.Variable(tf.random_normal(test_data.shape), name='test_data')
# test_data_tensor_summary = tf.summary.tensor_summary('test_data_summary', test_data_tensor, summary_description="test_data_summary_description")

# main loop

for i in range(1000000):
    #single training run
    mask = np.random.choice([False, True], len(input_train_data), p=[0.80, 0.20])
    step, _ = sess.run([global_step, train_step],
                       feed_dict={input_layer: input_train_data[mask],
                                  expected: output_train_data[mask],
                                  keep_prob1: 0.75,
                                  keep_prob2: 1.0,
                                  stddev: 0.1,
                                  learning_rate: 0.5})

    # every x steps, dump some stats to see how well the net is doing
    if i % 200 == 0:
        train_error, train_summary = sess.run([train_loss, train_loss_summary],
                                              feed_dict={input_layer: input_train_data, expected: output_train_data})

        test_error, prediction_output, expected_output, test_summary, prediction_summary_output, expected_summary_output = sess.run([test_loss, prediction, expected, test_loss_summary, prediction_summary, expected_summary],
                                                                                feed_dict={input_layer: input_test_data,
                                                                                           expected: output_test_data})
        # foo3 = np.subtract(prediction_output, expected_output)
        # foo4 = np.concatenate((prediction_output, expected_output, foo3), axis=1)
        # foo5 = foo4[0:90] / to_float[target_column]
        # foo6 = foo3[0:90, 0] / to_float[target_column]
        # print(foo6.astype(int))
        print('Step ', step, ' Train Error ', train_error, ' Test Error ', test_error)

        success = 0
        false_positive = 0
        false_negative = 0
        uncertain = 0
        late_threshold = 0.2
        on_time_threshold = -0.2
        for j in range(expected_output.shape[0]):
            if (prediction_output[j] > late_threshold and expected_output[j] > 0.5) or (prediction_output[j] < on_time_threshold and expected_output[j] < -0.5):
                success = success + 1
            elif prediction_output[j] > late_threshold and expected_output[j] < -0.5:
                false_negative = false_negative + 1
            elif prediction_output[j] < on_time_threshold and expected_output[j] > 0.5:
                false_positive = false_positive + 1
            else:
                uncertain = uncertain + 1

        print ('Success ', success / expected_output.shape[0] * 100)
        print ('Failed To Predict Late ', false_positive / expected_output.shape[0] * 100)
        print ('Failed To Predict On Time ', false_negative / expected_output.shape[0] * 100)
        print ('Uncertain ', uncertain / expected_output.shape[0] * 100)

        # test_data_tensor_result = sess.run(test_data_tensor_summary, feed_dict={test_data_tensor: test_data})
        # train_writer.add_summary(test_data_tensor_result, step)

        train_writer.add_summary(train_summary, step)
        train_writer.add_summary(test_summary, step)
        train_writer.add_summary(prediction_summary_output, step)
        train_writer.add_summary(expected_summary_output, step)
        train_writer.flush()

        save_path = saver.save(sess, "xor.ckpt")
