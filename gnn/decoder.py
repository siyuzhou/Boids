import tensorflow as tf
import numpy as np


def mlp_decoder(feature, params):
    time_series, edge_type = feature['time_series'], feature['edge_type']
