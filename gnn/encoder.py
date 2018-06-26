import numpy as np
import tensorflow as tf

from .modules import *


def mlp_encoder(features, classes, params, training=False):
    # Tensor `features` has shape [num_sims, time_steps, num_agents, ndims].
    time_steps, num_agents, ndims = features.shape.as_list()[1:]
    # Input Layer
    # Transpose to [num_sims, num_agents, time_steps, ndims]
    features = tf.transpose(features, [0, 2, 1, 3])
    state = tf.reshape(features, shape=[-1, num_agents, time_steps * ndims])
    # Node state encoder with MLP.
    state = mlp_layers(state,
                       params['hidden_units'],
                       params['dropout'],
                       params['batch_norm'],
                       training=training)

    # Send encoded state to edges.
    # `edge_sources` and `edge_targets` in shape [num_edges, num_agents].
    edge_sources, edge_targets = np.where(
        np.ones((num_agents, num_agents)) - np.eye(num_agents))
    # One-hot representation of indices of edge sources and targets.
    edge_sources = tf.one_hot(edge_sources, num_agents)
    edge_targets = tf.one_hot(edge_targets, num_agents)

    def node_to_edge(node_state):
        """Propagate node states to edges."""
        msg_from_source = tf.transpose(tf.tensordot(state, edge_sources, axes=[[1], [1]]),
                                       perm=[0, 2, 1])
        msg_from_target = tf.transpose(tf.tensordot(state, edge_targets, axes=[[1], [1]]),
                                       perm=[0, 2, 1])
        ## msg_from_source and msg_from_target in shape [num_sims, num_edges, out_units]
        msg_edge = tf.concat([msg_from_source, msg_from_target], axis=-1)

        # Encode edge messages with MLP
        msg_edge = mlp_layers(msg_edge,
                              params['hidden_units'],
                              params['dropout'],
                              params['batch_norm'],
                              training=training)
        return msg_edge

    msg_edge = node_to_edge(state)

    def edge_to_node(edge_msg):
        """Send edge messages to target nodes."""
        node_state = tf.transpose(tf.tensordot(edge_msg, edge_targets, axes=[[1], [0]]),
                                  perm=[0, 2, 1])  # Shape [num_sims, num_agents, out_units].
        # Encode node state with MLP
        node_state = mlp_layers(node_state,
                                params['hidden_units'],
                                params['dropout'],
                                params['batch_norm'],
                                training=training)
        return node_state

    state = edge_to_node(msg_edge)

    # Propagate node states to edges again.
    msg_edge = node_to_edge(state)

    edge_type = tf.layers.dense(msg_edge, classes)

    return edge_type


# Encoder function factory.
encoder_fn = {
    'mlp': mlp_encoder
}
