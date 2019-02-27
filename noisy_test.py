import os
import argparse
import json

import tensorflow as tf
import numpy as np

import gnn
from gnn.data import load_data
from gnn.utils import gumbel_softmax


def model_fn(features, labels, mode, params):
    pred_stack = gnn.dynamical.dynamical_multisteps(features,
                                                    params,
                                                    params['pred_steps'],
                                                    training=True)

    predictions = {'next_steps': pred_stack}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


def main():
    with open(ARGS.config) as f:
        model_params = json.load(f)

    model_params['pred_steps'] = 1

    cnn_multistep_regressor = tf.estimator.Estimator(
        model_fn=model_fn,
        params=model_params,
        model_dir=ARGS.log_dir)

    # Load test data
    if model_params.get('edge_types', 0) > 1:
        test_data, test_edge = load_data(ARGS.data_dir, ARGS.data_transpose, edge=True,
                                         prefix='test')
        test_edge = gnn.utils.one_hot(test_edge, model_params['edge_types'], np.float32)

        features = {'time_series': test_data, 'edge_type': test_edge}
    else:
        test_data = load_data(ARGS.data_dir, ARGS.data_transpose, edge=False,
                              prefix='test')
        features = {'time_series': test_data}

    # Reserve segment for only 1 step prediction
    seg_len = 2 * len(model_params['cnn']['filters']) + 1
    test_data_start = test_data[:, :seg_len, :, :]

    all_pred_record = []
    for i in range(ARGS.repeat):
        pred_record = []
        # Initialize time series
        features['time_series'] = test_data_start
        for _ in range(ARGS.pred_steps):
            # Predict the next step.
            predict_input_fn = tf.estimator.inputs.numpy_input_fn(
                x=features,
                batch_size=ARGS.batch_size,
                shuffle=False
            )
            prediction = cnn_multistep_regressor.predict(input_fn=predict_input_fn)
            prediction = np.array([pred['next_steps'] for pred in prediction])
            # Shape [num_sims, 1, 1, nagents, ndims]
            prediction = np.squeeze(prediction, axis=1)  # Shape [num_sims, 1, nagents, ndims]

            pred_record.append(prediction)
            features['time_series'] = np.concatenate(
                [features['time_series'][:, 1:, :, :], prediction], axis=1)

        pred_record = np.concatenate(pred_record, axis=1)
        all_pred_record.append(pred_record)
        print(f'Prediction repeatition {i} done.')

    all_pred_record = np.stack(all_pred_record, axis=0)
    np.save(os.path.join(ARGS.log_dir, 'prediction_{}.npy'.format(
        ARGS.pred_steps)), all_pred_record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str,
                        help='data directory')
    parser.add_argument('--data-transpose', type=int, nargs=4, default=None,
                        help='axes for data transposition')
    parser.add_argument('--config', type=str,
                        help='model config file')
    parser.add_argument('--log-dir', type=str,
                        help='log directory')
    parser.add_argument('--pred-steps', type=int, default=1,
                        help='number of steps the estimator predicts for time series')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='turn on logging info')
    parser.add_argument('--repeat', type=int,
                        help='number of repeatition for prediction')

    ARGS = parser.parse_args()

    if ARGS.verbose:
        tf.logging.set_verbosity(tf.logging.INFO)

    main()
