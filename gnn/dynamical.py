import numpy as np
import tensorflow as tf

from .modules import *
from .utils import fc_matrix


def node_to_edge(node_msg, edge_sources, edge_targets):
    """Propagate node states to edges."""
    with tf.name_scope("node_to_edge"):
        msg_from_source = tf.transpose(tf.tensordot(node_msg, edge_sources, axes=[[1], [1]]),
                                       perm=[0, 4, 1, 2, 3])
        msg_from_target = tf.transpose(tf.tensordot(node_msg, edge_targets, axes=[[1], [1]]),
                                       perm=[0, 4, 1, 2, 3])
        # msg_from_source and msg_from_target in shape [num_sims, num_edges, num_time_series, 1, out_units]
        edge_msg = tf.concat([msg_from_source, msg_from_target], axis=-1)

    return edge_msg


def edge_to_node(edge_msg, edge_targets):
    """Send edge messages to target nodes."""
    with tf.name_scope("edge_to_node"):
        node_msg = tf.transpose(tf.tensordot(edge_msg, edge_targets, axes=[[1], [0]]),
                                perm=[0, 4, 1, 2, 3])  # Shape [num_sims, num_agents, num_time_series, 1, out_units].

    return node_msg


def cnn_dynamical(time_series_stack, params, training=False):
    """Next step prediction using CNN and GNN."""
    # Tensor `time_series` has shape [num_sims, num_agents, num_time_series, time_steps, ndims].
    num_sims, num_agents, num_time_series, time_steps, ndims = time_series_stack.shape.as_list()
    n_conv_layers = len(params['cnn']['filters'])
    if time_steps is None:
        time_steps = 2*n_conv_layers+1

    if params['cnn']['filters']:
        # Input Layer
        # Reshape to [num_sims*num_agents*num_time_series, time_steps, ndims], since conv1d only accept
        # tensor with 3 dimensions.
        state = tf.reshape(time_series_stack, shape=[-1, time_steps, ndims])

        # Node state encoder with 1D convolution along timesteps and across ndims as channels.
        encoded_state = state
        for filters in params['cnn']['filters']:
            encoded_state = tf.layers.conv1d(encoded_state, filters, 3, activation=tf.nn.relu)
            # No pooling layer

        # encoded_state shape [num_sims, num_agents, num_time_series, 1, filters]
        encoded_state = tf.reshape(encoded_state,
                                   shape=[-1, num_agents, num_time_series, 1, filters])
    else:
        encoded_state = time_series_stack

    # Send encoded state to edges.
    # `edge_sources` and `edge_targets` in shape [num_edges, num_agents].
    edge_sources, edge_targets = np.where(fc_matrix(num_agents))
    # One-hot representation of indices of edge sources and targets.
    with tf.name_scope("one_hot"):
        edge_sources = tf.one_hot(edge_sources, num_agents)
        edge_targets = tf.one_hot(edge_targets, num_agents)

    # Form edges. Shape [num_sims, num_edges, num_time_series, 1, filters]
    edge_msg = node_to_edge(encoded_state, edge_sources, edge_targets)

    # Store skip.
    edge_msg_skip = edge_msg

    # Encode edge messages with MLP. Shape [num_sims, num_edges, num_time_series, 1, hidden_units]
    edge_msg = mlp_layers(edge_msg,
                          params['mlp']['hidden_units'],
                          params['mlp']['dropout'],
                          params['mlp']['batch_norm'],
                          training=training,
                          name='edge_encoding_MLP_1')

    # Compute edge influence to node. Shape [num_sims, num_agents, num_time_series, 1, hidden_units]
    edge_msg_aggr = edge_to_node(edge_msg, edge_targets)

    # Encode node messages with MLP
    node_msg = mlp_layers(edge_msg_aggr,
                          params['mlp']['hidden_units'],
                          params['mlp']['dropout'],
                          params['mlp']['batch_norm'],
                          training=training,
                          name='node_encoding_MLP_1')

    # The last state in each timeseries of the stack.
    prev_state = time_series_stack[:, :, :, -1:, :]

    node_state = tf.concat([prev_state, node_msg], axis=-1)

    # Decode next step. Shape [num_sims, num_agents, num_time_series, 1, hidden_units]
    node_state = mlp_layers(node_state,
                            params['mlp']['hidden_units'],
                            params['mlp']['dropout'],
                            params['mlp']['batch_norm'],
                            training=training,
                            name='node_decoding_MLP')

    next_state = tf.layers.dense(node_state, ndims, name='linear')

    return next_state


def dynamical_multisteps(features, params, pred_steps, training=False):
    # features shape [num_sims, time_steps, num_agents, ndims]
    num_sims, time_steps, num_agents, ndims = features.shape.as_list()
    # Transpose to [num_sims, num_agents, time_steps, ndims]
    time_series = tf.transpose(features, [0, 2, 1, 3])

    n_conv_layers = len(params['cnn']['filters'])

    time_series_stack = tf.stack([time_series[:, :, i:i+time_steps-2*n_conv_layers, :]
                                  for i in range(2*n_conv_layers+1)],
                                 axis=3)
    # Shape [num_sims, num_agents, time_steps-2*n_conv_layers, 2*n_conv_layers+1, ndims]

    with tf.variable_scope('prediction_one_step') as scope:
        pass

    def one_step(i, time_series_stack):
        with tf.name_scope(scope.original_name_scope):
            prev_step = time_series_stack[:, :, :, -1:, :]
            next_state = prev_step + cnn_dynamical(
                time_series_stack[:, :, :, i:, :], params, training=training)

            return i+1, tf.concat([time_series_stack, next_state], axis=3)

    i = 0
    _, time_series_stack = tf.while_loop(
        lambda i, _: i < pred_steps,
        one_step,
        [i, time_series_stack],
        shape_invariants=[tf.TensorShape(None),
                          tf.TensorShape([num_sims, num_agents, time_steps-2*n_conv_layers, None, ndims])]
    )

    # Transpose to [num_sims, num_timeseries, seg_time_steps, num_agents, ndims]
    # where num_timeseries = time_steps - 2*n_conv_layers, seg_time_steps = 2*n_conv_layers + pred_steps
    time_series_stack = tf.transpose(time_series_stack, [0, 2, 3, 1, 4])

    return time_series_stack[:, :, 2*n_conv_layers+1:, :, :]
