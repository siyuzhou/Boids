import os
import argparse
import json

import tensorflow as tf
import numpy as np

import gnn
from gnn.modules import mlp_layers
from gnn.data import load_data


def lstm(time_series, params, pred_steps, training=False):
    # timeseries shape [num_sims, time_steps, num_agents, ndims]
    num_sims, time_steps, num_agents, ndims = time_series.shape.as_list()

    time_series = tf.reshape(time_series, [-1, num_agents * ndims])
    time_series_stack = tf.TensorArray(tf.float32, pred_steps)
    # Shape [pred_steps, num_sims * time_steps, num_agents * ndims]
    # print('\n\n time_series_stack shape {} \n'.format(time_series_stack.shape))

    with tf.variable_scope('prediction_one_step') as scope:
        lstm_cell = tf.nn.rnn_cell.LSTMCell(params['units'])
        init_state = lstm_cell.zero_state(tf.shape(time_series)[0], tf.float32)

    def one_step(i, prev_state, rnn_state, time_series_stack):
        with tf.name_scope(scope.original_name_scope):
            output, rnn_state = lstm_cell(prev_state, rnn_state)
            pred = tf.layers.dense(output, num_agents * ndims, name='linear')
            next_state = prev_state + pred

            time_series_stack = time_series_stack.write(i, next_state)

            return i+1, next_state, rnn_state, time_series_stack

    i = 0
    _, _, _, time_series_stack = tf.while_loop(
        lambda i, p, t, s: i < pred_steps,
        one_step,
        [i, time_series, init_state, time_series_stack]
    )

    time_series_stack = tf.transpose(time_series_stack.stack(), [1, 0, 2])
    time_series_stack = tf.reshape(time_series_stack,
                                   [-1, time_steps, pred_steps, num_agents, ndims])

    return time_series_stack


def model_fn(features, labels, mode, params):
    pred_stack = lstm(features,
                      params,
                      params['pred_steps'],
                      training=(mode == tf.estimator.ModeKeys.TRAIN))

    predictions = {'next_steps': pred_stack}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    expected_time_series = gnn.utils.stack_time_series(features[:, 1:, :, :],
                                                       params['pred_steps'])

    loss = tf.losses.mean_squared_error(expected_time_series,
                                        pred_stack[:, :-params['pred_steps'], :, :, :])

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
    time_series_loss_baseline = tf.metrics.mean_squared_error(features[:, 1:, :, :],
                                                              features[:, :-1, :, :])

    eval_metric_ops = {'time_series_loss_baseline': time_series_loss_baseline}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main():
    with open(ARGS.config) as f:
        model_params = json.load(f)

    model_params['pred_steps'] = ARGS.pred_steps

    lstm_multistep_regressor = tf.estimator.Estimator(
        model_fn=model_fn,
        params=model_params,
        model_dir=ARGS.log_dir)

    if ARGS.train:
        train_data = load_data(ARGS.data_dir, ARGS.data_transpose, edge=False,
                               prefix='train')

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=train_data,
            batch_size=ARGS.batch_size,
            num_epochs=None,
            shuffle=True
        )

        lstm_multistep_regressor.train(input_fn=train_input_fn,
                                       steps=ARGS.train_steps)

    # Evaluation
    if ARGS.eval:
        valid_data = load_data(ARGS.data_dir, ARGS.data_transpose, edge=False,
                               prefix='valid')

        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=valid_data,
            batch_size=ARGS.batch_size,
            num_epochs=1,
            shuffle=False
        )
        eval_results = lstm_multistep_regressor.evaluate(input_fn=eval_input_fn)

        if not ARGS.verbose:
            print('Evaluation results: {}'.format(eval_results))

    # Prediction
    if ARGS.test:
        test_data = load_data(ARGS.data_dir, ARGS.data_transpose, edge=False,
                              prefix='test')

        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=test_data,
            batch_size=ARGS.batch_size,
            shuffle=False
        )

        prediction = lstm_multistep_regressor.predict(input_fn=predict_input_fn)
        prediction = np.array([pred['next_steps'] for pred in prediction])
        np.save(os.path.join(ARGS.log_dir, 'prediction_{}.npy'.format(
            ARGS.pred_steps)), prediction)


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
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='turn on logging info')
    parser.add_argument('--train', action='store_true', default=False,
                        help='turn on training')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='turn on evaluation')
    parser.add_argument('--test', action='store_true', default=False,
                        help='turn on test')
    ARGS = parser.parse_args()

    if ARGS.verbose:
        tf.logging.set_verbosity(tf.logging.INFO)

    main()
