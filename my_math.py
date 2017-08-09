import tensorflow as tf
import numpy as np

def lrelu(x, alpha=0.1, max_value=None):
    return tf.maximum(alpha*x,x)

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def denormalize(v, x):
    return v * (np.max(x) - np.min(x)) + np.min(x)

