import os
import argparse
import json

import numpy as np
import tensorflow as tf

import gnn
from gnn.data import load_data


def decoder_model_fn(features, labels, mode, params):
    pred = gnn.decoder.decoder_fn[params['decoder']](
        features,
        params['decoder_params'],
        params['pred_steps'],
        training=(mode == tf.estimator.ModeKeys.TRAIN)
    )

    predictions = {'state_next_steps': pred}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    expected_time_series = gnn.utils.stack_time_series(features['time_series'],
                                                       params['pred_steps'])

    loss = tf.losses.mean_squared_error(
        expected_time_series, pred[:, :-params['pred_steps'], :, :])

    if mode == tf.estimator.ModeKeys.TRAIN:
        learning_rate = tf.train.exponential_decay(
            learning_rate=params['learning_rate'],
            global_step=tf.train.get_global_step(),
            decay_steps=100,
            decay_rate=0.95,
            staircase=True,
            name='learning_rate'
        )
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Use the loss between adjacent steps in original time_series as baseline
    time_series_loss_baseline = tf.metrics.mean_squared_error(features['time_series'][:, 1:, :, :],
                                                              features['time_series'][:, :-1, :, :])
    eval_metric_ops = {'time_series_loss_baseline': time_series_loss_baseline}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main():
    with open(ARGS.config) as f:
        model_params = json.load(f)

    print('Loading data...')
    train_data, train_edge, test_data, test_edge = load_data(
        ARGS.data_dir, ARGS.data_transpose)

    # Convert int labels to one hot vectors.
    train_edge = gnn.utils.one_hot(train_edge, model_params['edge_types'], np.float32)
    test_edge = gnn.utils.one_hot(test_edge, model_params['edge_types'], np.float32)

    model_params['pred_steps'] = ARGS.pred_steps

    mlp_decoder_regressor = tf.estimator.Estimator(
        model_fn=decoder_model_fn,
        params=model_params,
        model_dir=ARGS.log_dir)

    if not ARGS.no_train:
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'time_series': train_data,
               'edge_type': train_edge},
            batch_size=ARGS.batch_size,
            num_epochs=None,
            shuffle=True
        )

        mlp_decoder_regressor.train(input_fn=train_input_fn,
                                    steps=ARGS.train_steps)

    # Evaluation
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'time_series': test_data,
           'edge_type': test_edge},
        batch_size=ARGS.batch_size,
        num_epochs=1,
        shuffle=False
    )
    eval_results = mlp_decoder_regressor.evaluate(input_fn=eval_input_fn)
    # print("Validation set:", eval_results)

    # Prediction
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'time_series': test_data[:10],
           'edge_type': test_edge[:10]},
        batch_size=ARGS.batch_size,
        shuffle=False
    )

    prediction = mlp_decoder_regressor.predict(input_fn=predict_input_fn)
    prediction = np.array([pred['state_next_steps'] for pred in prediction])
    np.save(os.path.join(ARGS.log_dir, 'prediction.npy'), prediction)


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
    parser.add_argument('--train-steps', type=int, default=1000,
                        help='number of training steps')
    parser.add_argument('--pred-steps', type=int, default=1,
                        help='number of steps the estimator predicts for time series')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--no-train', action='store_true', default=False,
                        help='skip training and use for evaluation only')
    ARGS = parser.parse_args()

    tf.logging.set_verbosity(tf.logging.INFO)

    main()
