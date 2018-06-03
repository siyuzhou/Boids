import argparse

import numpy as np
import tensorflow as tf

import utils


def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()


def eval_input_fn(features, labels, batch_size):
    if labels is None:
        return tf.data.Dataset.from_tensor_slices(features).batch(batch_size)

    return tf.data.Dataset.from_tensor_slices((features, labels)).batch(batch_size)
