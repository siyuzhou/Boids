import argparse

import numpy as np
import tensorflow as tf

from gnn import utils
from gnn.decoder import decoder_fn


# def train_input_fn(features, labels, batch_size):
#     dataset = tf.data.Dataset.from_tensor_slices((features, labels))
#     dataset = dataset.shuffle(1000).repeat().batch(batch_size)
#     return dataset.make_one_shot_iterator().get_next()


# def eval_input_fn(features, labels, batch_size):
#     if labels is None:
#         return tf.data.Dataset.from_tensor_slices(features).batch(batch_size)

#     return tf.data.Dataset.from_tensor_slices((features, labels)).batch(batch_size)

def decoder_model_fn(features, labels, mode, params):
    pred = decoder_fn[params['decoder']](features, params['decoder_params'],
                                         training=(mode == tf.estimator.ModeKeys.TRAIN))
                                
    predictions = {'pred': pred}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.mean_squared_error(features['time_series'][:, 1:, :, :],
                                        pred[:, :-1, :, :])
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0004)
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # eval accuracy
