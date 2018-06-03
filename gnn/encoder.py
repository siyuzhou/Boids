import numpy as np
import tensorflow as tf


def mlp_encoder(feature, mode, params):
    # Tensor `feature` has shape [num_sims, time_steps, num_agents, ndims].
    feature = feature['time_series']
    time_steps, num_agents, ndims = feature.get_shape().as_list()[1:]
    # Input Layer
    # Transpose to [num_sims, num_agents, time_steps, ndims]
    feature = tf.transpose(feature, [0, 2, 1, 3])
    state = tf.reshape(feature, shape=[-1, num_agents, time_steps * ndims])
    # Node state encoder with MLP.
    for units in params['node_encoder']['hidden_units'][:-1]:
        state = tf.layers.dense(state, units, activation=tf.nn.relu)
        state = tf.layers.dropout(state, params['dropout'],
                                  training=(mode == tf.estimator.ModeKeys.TRAIN))
    out_units = params['node_encoder']['hidden_units'][-1]
    state = tf.layers.dense(state, out_units, activation=tf.nn.relu)
    # `state` encoded the node state, shape [num_sims, num_agents, out_units].
    if params['batch_norm']:
        state = tf.layers.batch_normalization(
            state, training=(mode == tf.estimator.ModeKeys.TRAIN))
    # Send encoded state to edges.
    # `edge_sources` and `edge_targets` in shape [num_edges, num_agents].
    edge_sources, edge_targets = np.where(np.ones((num_agents, num_agents)) - np.eye(num_agents))
    # One-hot representation of indices of edge sources and targets.
    edge_sources = tf.one_hot(edge_sources, num_agents)
    edge_targets = tf.one_hot(edge_targets, num_agents)

    msg_from_source = tf.transpose(tf.tensordot(state, edge_sources, axes=[[1], [1]]),
                                   perm=[0, 2, 1])
    msg_from_target = tf.transpose(tf.tensordot(state, edge_targets, axes=[[1], [1]]),
                                   perm=[0, 2, 1])
    ## msg_source and msg_target in shape [num_sims, num_edges, out_units]
    msg_edge = tf.concat([msg_from_source, msg_from_target], axis=-1)

    # Encode edge messages with MLP
    for units in params['edge_encoder']['hidden_units'][:-1]:
        msg_edge = tf.layers.dense(msg_edge, units, activation=tf.nn.relu)
        msg_edge = tf.layers.dropout(msg_edge, params['dropout'],
                                     training=(mode == tf.estimator.ModeKeys.TRAIN))
    out_units = params['node_encoder']['hidden_units'][-1]
    msg_edge = tf.layers.dense(msg_edge, out_units, activation=tf.nn.relu)

    if params['batch_norm']:
        msg_edge = tf.layers.batch_normalization(
            msg_edge, training=(mode == tf.estimator.ModeKeys.TRAIN))

    edge_type = tf.layers.dense(msg_edge, params['edge_types'])

    return edge_type


# Encoder function factory.
encoder_fn = {
    'mlp': mlp_encoder
}
