import os

import numpy as np

from gnn import utils


def load_data(data_path, transpose=None, edge=True):
    # Training data.
    train_loc = np.load(os.path.join(data_path, 'train_position.npy'))
    train_vel = np.load(os.path.join(data_path, 'train_velocity.npy'))

    if transpose:
        train_loc = np.transpose(train_loc, transpose)
        train_vel = np.transpose(train_vel, transpose)

    train_data = np.concatenate([train_loc, train_vel], axis=-1).astype(np.float32)

    # Validation data.
    valid_loc = np.load(os.path.join(data_path, 'valid_position.npy'))
    valid_vel = np.load(os.path.join(data_path, 'valid_velocity.npy'))

    if transpose:
        valid_loc = np.transpose(valid_loc, transpose)
        valid_vel = np.transpose(valid_vel, transpose)

    valid_data = np.concatenate([valid_loc, valid_vel], axis=-1).astype(np.float32)

    # Test data.
    test_loc = np.load(os.path.join(data_path, 'test_position.npy'))
    test_vel = np.load(os.path.join(data_path, 'test_velocity.npy'))

    if transpose:
        test_loc = np.transpose(test_loc, transpose)
        test_vel = np.transpose(test_vel, transpose)

    test_data = np.concatenate([test_loc, test_vel], axis=-1).astype(np.float32)

    if edge:
        # Edge data.
        train_edge = np.load(os.path.join(data_path, 'train_edge.npy')).astype(np.int)

        instances, n_agents, _ = train_edge.shape
        train_edge = np.stack([train_edge[i][np.where(utils.fc_matrix(n_agents))]
                               for i in range(instances)], 0)

        valid_edge = np.load(os.path.join(data_path, 'valid_edge.npy')).astype(np.int)

        instances, n_agents, _ = valid_edge.shape
        valid_edge = np.stack([valid_edge[i][np.where(utils.fc_matrix(n_agents))]
                               for i in range(instances)], 0)

        test_edge = np.load(os.path.join(data_path, 'test_edge.npy')).astype(np.int)

        instances, n_agents, _ = test_edge.shape
        test_edge = np.stack([test_edge[i][np.where(utils.fc_matrix(n_agents))]
                              for i in range(instances)], 0)

        return train_data, train_edge, valid_data, valid_edge, test_data, test_edge

    return train_data, valid_data, test_data
