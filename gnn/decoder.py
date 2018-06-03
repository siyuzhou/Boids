import numpy as np
import tensorflow as tf


def mlp_decoder(feature, params):
    time_series, edge_type = feature['time_series'], feature['edge_type']
