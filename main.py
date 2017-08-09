import csv
import tensorflow as tf
import numpy as np
import os.path
import math

from pca import plot_pca
# from my_math import normalize
from train import train

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

data = np.delete(data, [0, 1], axis=1)

# convert data from strings to floats because the data was loaded as strings
legs = tf.convert_to_tensor(data, dtype=tf.float32, name='data').eval(session=sess)

depart_month = legs[:, [0]]
depart_day = legs[:, [1]]
depart_hour = legs[:, [2]]
depart_minute = legs[:, [3]]
duration = legs[:, [4]]
actual_out = legs[:, [5]]
actual_off = legs[:, [6]]
actual_on = legs[:, [7]]
actual_in = legs[:, [8]]

in_delta = actual_out - actual_in
taxi_out = actual_off - actual_out
taxi_in = actual_in - actual_on

bad_values_indicies = [i for (i, v) in enumerate(taxi_in) if v <= 0 or v > np.median(taxi_in) * 2]

depart_month = np.delete(depart_month, bad_values_indicies, axis=0)
depart_day = np.delete(depart_day, bad_values_indicies, axis=0)
depart_hour = np.delete(depart_hour, bad_values_indicies, axis=0)
depart_minute = np.delete(depart_minute, bad_values_indicies, axis=0)
duration = np.delete(duration, bad_values_indicies, axis=0)
actual_out = np.delete(actual_out, bad_values_indicies, axis=0)
actual_off = np.delete(actual_off, bad_values_indicies, axis=0)
actual_on = np.delete(actual_on, bad_values_indicies, axis=0)
actual_in = np.delete(actual_in, bad_values_indicies, axis=0)

in_delta = np.delete(in_delta, bad_values_indicies, axis=0)
taxi_out = np.delete(taxi_out, bad_values_indicies, axis=0)
taxi_in = np.delete(taxi_in, bad_values_indicies, axis=0)

dept_one_hot_result = np.delete(dept_one_hot_result, bad_values_indicies, axis=0)
dest_one_hot_result = np.delete(dest_one_hot_result, bad_values_indicies, axis=0)

# if IN greater than late_value, set to 1 otherwise set to -1 so we target flight that are greater than late_value
# late_value = 30
# in_times = raw_legs[:, 8]
# for i in range(in_times.shape[0]):
#     if in_times[i] >= late_value:
#         in_times[i] = 1;
#     else:
#         in_times[i] = -1;
# raw_legs[:, 8] = in_times

# delete the IN times
# raw_legs = np.delete(raw_legs, [8], axis=1)


# X = np.concatenate((depart_month, depart_day, depart_hour, depart_minute, duration, actual_out, actual_off, actual_on, actual_in), axis=1)
# X = np.concatenate((depart_month, depart_day, depart_hour, depart_minute, duration), axis=1)
# X = np.concatenate((actual_out, actual_off), axis=1)
X = np.concatenate((actual_on, actual_in), axis=1)
# X = np.concatenate((depart_month, depart_day, depart_hour, depart_minute, duration, actual_out, dept_one_hot_result), axis=1)
# X = dept_one_hot_result

# add the dept and dest back in
# X = np.concatenate((X, dept_one_hot_result), axis=1)
# X = np.concatenate((X, dest_one_hot_result), axis=1)

# min_in_delta = -30
# max_in_delta = 30

# Y = (in_delta - min_in_delta) / max_in_delta
# Y = sess.run(tf.clip_by_value(Y, 0, 1))

# cutoff = np.median(taxi_in)
# early = (np.ones(in_delta.shape) * [in_delta <= cutoff])[0]
# late = (np.ones(taxi_in.shape) * [taxi_in > cutoff])[0]


# Y = normalize(taxi_out)

# plot_pca(X, Y)

# early = (np.ones(in_delta.shape) * [in_delta < -5])[0]
# on_time = (np.ones(in_delta.shape) * np.logical_and([in_delta >= -5], [in_delta<=5]))[0]
# late = (np.ones(in_delta.shape) * [in_delta > 5])[0]
#
# Y = np.concatenate((early, on_time, late), axis=1)

# far_range_indicies = [i for (i, v) in enumerate(in_delta) if not -15 < v < 15]
# X = np.concatenate((X[far_range_indicies], X))
# Y = np.concatenate((Y[far_range_indicies], Y))
# X = np.concatenate((X[far_range_indicies], X))
# Y = np.concatenate((Y[far_range_indicies], Y))

# taxi_out_under = (np.ones(taxi_out.shape) * [taxi_out <= 14])[0]
# taxi_out_over = (np.ones(taxi_out.shape) * [taxi_out > 14])[0]
#
# Y = np.concatenate((taxi_out_under, taxi_out_over), axis=1)

# Y = sess.run(tf.clip_by_value(Y, 0, 1))


train(X, taxi_in, 0.001, 1.0)

