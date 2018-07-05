import numpy as np
import tensorflow as tf

from .modules import *
from .utils import fc_matrix


def mlp_decoder(features,  params, training=False):
    time_series, edge_type = features['time_series'], features['edge_type']
    # time_series shape [num_sims, time_steps, num_agents, ndims]
    # edge_type shape [num_sims, num_edges, num_edge_types]
    time_steps, num_agents, ndims = time_series.shape.as_list()[1:]
    num_types = edge_type.shape.as_list()[-1]

    # Send encoded state to edges.
    edge_sources, edge_targets = np.where(fc_matrix(num_agents))
    # One-hot representation of indices of edge sources and targets.
    # `edge_sources` and `edge_targets` in shape [num_edges, num_agents].
    edge_sources = tf.one_hot(edge_sources, num_agents)
    edge_targets = tf.one_hot(edge_targets, num_agents)

    msg_from_source = tf.transpose(tf.tensordot(time_series, edge_sources, axes=[[2], [1]]),
                                   perm=[0, 1, 3, 2])
    msg_from_target = tf.transpose(tf.tensordot(time_series, edge_targets, axes=[[2], [1]]),
                                   perm=[0, 1, 3, 2])
    # msg_from_source and msg_from_target in shape [num_sims, time_steps, num_edges, ndims]
    msg_edge = tf.concat([msg_from_source, msg_from_target], axis=-1)

    encoded_msg_by_type = []
    # Encode edge message by types and concatenate them.
    for _ in range(num_types):
        # mlp_encoder for one edge type.
        encoded_msg = mlp_layers(msg_edge,
                                 params['hidden_units'],
                                 params['dropout'],
                                 params['batch_norm'],
                                 training=training)

        encoded_msg_by_type.append(encoded_msg)

    encoded_msg_by_type = tf.stack(encoded_msg_by_type, axis=3)

    # shape [num_sims, time_steps, num_edges, num_types, out_units]

    # Expand edge_type shape to [num_sims, 1, num_edges, num_edge_types, 1]
    edge_type = tf.expand_dims(edge_type, 1)
    edge_type = tf.expand_dims(edge_type, 4)

    # Sum of the edge encoding from all possible types.
    encoded_msg_sum = tf.reduce_sum(tf.multiply(encoded_msg_by_type,
                                                edge_type),
                                    axis=3)
    # shape [num_sims, time_steps, num_edges, out_units]

    # Aggregate msg from all edges to target node.
    msg_aggregated = tf.transpose(tf.tensordot(encoded_msg_sum, edge_targets, axes=[[2], [0]]),
                                  perm=[0, 1, 3, 2])
    # shape [num_sims, time_steps, num_agents, out_units]

    # Skip connection.
    msg_node = tf.concat([time_series, msg_aggregated], axis=-1)
    # shape [num_sims, time_steps, num_edges, 2*ndims + out_units]

    # MLP encoder
    msg_node_encoded = mlp_layers(msg_node,
                                  params['hidden_units'],
                                  params['dropout'],
                                  batch_norm=False,
                                  training=training)

    pred_displacement = tf.layers.dense(msg_node_encoded, ndims)

    return time_series + pred_displacement


decoder_fn = {
    'mlp': mlp_decoder
}
