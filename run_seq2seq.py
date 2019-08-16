import os
import argparse
import json

import tensorflow as tf
import numpy as np

import gnn
from gnn.modules import mlp_layers
from gnn.data import load_data


def seq2seq(time_segs, params, pred_steps, training=False):
    # timeseries shape [num_sims, time_seq_len, num_agents, ndims]
    num_segs, time_seg_len, num_agents, ndims = time_segs.shape.as_list()

    time_segs = tf.reshape(time_segs, [-1, time_seg_len, num_agents * ndims])
    # Shape [pred_steps, time_seg_len, num_agents * ndims]
    pred_seqs = tf.TensorArray(tf.float32, pred_steps)

    with tf.variable_scope('RNN_Encoder'):
        encoder_lstm_cell = tf.nn.rnn_cell.LSTMCell(
            params['units'], dtype=tf.float32, name='encoder_lstm')
        init_state = encoder_lstm_cell.zero_state(tf.shape(time_segs)[0], dtype=tf.float32)
        _, hidden_state = tf.nn.dynamic_rnn(encoder_lstm_cell, time_segs[:, :-1, :],
                                            initial_state=init_state,
                                            dtype=tf.float32)

    with tf.variable_scope('prediction_one_step') as decoder_scope:
        decoder_lstm_cell = tf.nn.rnn_cell.LSTMCell(
            params['units'], name='decoder_lstm', dtype=tf.float32)

    prev_state = time_segs[:, -1, :]

    def one_step(i, prev_state, rnn_state, time_series_stack):
        with tf.name_scope(decoder_scope.original_name_scope):
            output, rnn_state = decoder_lstm_cell(prev_state, rnn_state)
            pred = tf.layers.dense(output, num_agents * ndims, name='linear')
            next_state = prev_state + pred

            time_series_stack = time_series_stack.write(i, next_state)

            return i+1, next_state, rnn_state, time_series_stack

    i = 0
    _, _, _, pred_seqs = tf.while_loop(
        lambda i, p, t, s: i < pred_steps,
        one_step,
        [i, prev_state, hidden_state, pred_seqs]
    )

    pred_seqs = tf.transpose(pred_seqs.stack(), [1, 0, 2])
    pred_seqs = tf.reshape(pred_seqs,
                           [-1, pred_steps, num_agents, ndims])

    return pred_seqs


def model_fn(features, labels, mode, params):
    pred_seqs = seq2seq(features['time_series'],
                        params,
                        params['pred_steps'],
                        training=(mode == tf.estimator.ModeKeys.TRAIN))

    predictions = {'next_steps': pred_seqs}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.mean_squared_error(labels, pred_seqs)

    if mode == tf.estimator.ModeKeys.TRAIN:
        learning_rate = tf.train.exponential_decay(
            learning_rate=params['learning_rate'],
            global_step=tf.train.get_global_step(),
            decay_steps=1000,
            decay_rate=0.99,
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
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def input_fn(features, seg_len, pred_steps, batch_size, mode='train'):
    time_series = features['time_series']
    num_sims, time_steps, num_agents, ndims = time_series.shape
    # Shape [num_sims, time_steps, num_agents, ndims]
    time_series_stack = gnn.utils.stack_time_series(time_series[:, :-pred_steps, :, :],
                                                    seg_len)
    # Shape [num_sims, time_steps-seg_len-pred_steps+1, seg_len, num_agents, ndims]
    expected_time_series_stack = gnn.utils.stack_time_series(time_series[:, seg_len:, :, :],
                                                             pred_steps)
    # Shape [num_sims, time_steps-seg_len-pred_steps+1, pred_steps, num_agents, ndims]
    assert time_series_stack.shape[:2] == expected_time_series_stack.shape[:2]

    time_segs = time_series_stack.reshape([-1, seg_len, num_agents, ndims])
    expected_time_segs = expected_time_series_stack.reshape([-1, pred_steps, num_agents, ndims])

    processed_features = {'time_series': time_segs}

    labels = expected_time_segs

    if mode == 'train':
        return tf.estimator.inputs.numpy_input_fn(
            x=processed_features,
            y=labels,
            batch_size=batch_size,
            num_epochs=None,
            shuffle=True
        )
    elif mode == 'eval':
        return tf.estimator.inputs.numpy_input_fn(
            x=processed_features,
            y=labels,
            batch_size=batch_size,
            shuffle=False
        )
    elif mode == 'test':
        return tf.estimator.inputs.numpy_input_fn(
            x=processed_features,
            batch_size=batch_size,
            shuffle=False
        )


def main():
    with open(ARGS.config) as f:
        model_params = json.load(f)

    seg_len = model_params['seg_len']
    model_params['pred_steps'] = ARGS.pred_steps

    seq2seq_regressor = tf.estimator.Estimator(
        model_fn=model_fn,
        params=model_params,
        model_dir=ARGS.log_dir)

    if ARGS.train:
        train_data = load_data(ARGS.data_dir, ARGS.data_transpose, edge=False,
                               prefix='train')

        features = {'time_series': train_data}

        train_input_fn = input_fn(features, seg_len, ARGS.pred_steps, ARGS.batch_size, 'train')

        seq2seq_regressor.train(input_fn=train_input_fn,
                                steps=ARGS.train_steps)

    # Evaluation
    if ARGS.eval:
        valid_data = load_data(ARGS.data_dir, ARGS.data_transpose, edge=False,
                               prefix='valid')
        features = {'time_series': valid_data}

        eval_input_fn = input_fn(features, seg_len, ARGS.pred_steps, ARGS.batch_size, 'eval')

        eval_results = seq2seq_regressor.evaluate(input_fn=eval_input_fn)

        if not ARGS.verbose:
            print('Evaluation results: {}'.format(eval_results))

    # Prediction
    if ARGS.test:
        test_data = load_data(ARGS.data_dir, ARGS.data_transpose, edge=False,
                              prefix='test')
        features = {'time_series': test_data}

        predict_input_fn = input_fn(features, seg_len, ARGS.pred_steps, ARGS.batch_size, 'test')

        prediction = seq2seq_regressor.predict(input_fn=predict_input_fn)
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

    ARGS.data_dir = os.path.expanduser(ARGS.data_dir)
    ARGS.config = os.path.expanduser(ARGS.config)
    ARGS.log_dir = os.path.expanduser(ARGS.log_dir)

    if ARGS.verbose:
        tf.logging.set_verbosity(tf.logging.INFO)

    main()
