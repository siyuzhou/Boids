import os
import argparse
import json

import tensorflow as tf
import numpy as np

import gnn
from gnn.data import load_data
from gnn.utils import gumbel_softmax


def model_fn(features, labels, mode, params):
    time_series, edge_type = features, labels

    # For training and evaluatin, the last 'pred_steps' cannot be used as starting points,
    # because there would be no further ground truth to compare with.
    known_time_series = time_series[:, :-params['pred_steps'], :, :]

    # Infer edge_type with encoder.
    edge_type_logits = gnn.encoder.encoder_fn[params['encoder']](
        known_time_series,
        params['edge_types'],
        params['encoder_params'],
        training=(mode == tf.estimator.ModeKeys.TRAIN))

    if params['temperature'] == 0:  # Turn off gumbel, use softmax instead
        edge_type_prob = tf.nn.softmax(edge_type_logits)
    else:
        edge_type_prob = gumbel_softmax(edge_type_logits, params['temperature'])
    # Predict state of next steps with decoder
    # using time_series and edge_type_logits
    state_next_step = gnn.decoder.decoder_fn[params['decoder']](
        {'time_series': known_time_series,
         'edge_type': edge_type_prob},
        params['decoder_params'],
        params['pred_steps'],
        training=(mode == tf.estimator.ModeKeys.TRAIN))

    predictions = {'state_next_step': state_next_step,
                   'edge_type_prob': tf.nn.softmax(edge_type_logits),
                   'edge_type': tf.argmax(input=edge_type_logits, axis=-1)}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    expected_time_series = gnn.utils.stack_time_series(time_series[:, 1:, :, :],
                                                       params['pred_steps'])

    trajectory_loss = tf.losses.mean_squared_error(expected_time_series,
                                                   predictions['state_next_step'])

    edge_type_loss = tf.losses.sparse_softmax_cross_entropy(labels=edge_type,
                                                            logits=edge_type_logits)

    edge_kl_loss = tf.losses.softmax_cross_entropy(
        onehot_labels=predictions['edge_type_prob'],
        logits=edge_type_logits
    )

    loss = trajectory_loss  # / params['variance'] - edge_kl_loss
    if params['supervised']:
        loss += edge_type_loss

    if mode == tf.estimator.ModeKeys.TRAIN:
        learning_rate = tf.train.exponential_decay(
            learning_rate=params['learning_rate'],
            global_step=tf.train.get_global_step(),
            decay_steps=100,
            decay_rate=0.95,
            staircase=True
        )
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Use the loss between adjacent steps in original time_series as baseline
    trajectory_loss_eval = tf.metrics.mean_squared_error(expected_time_series,
                                                         predictions['state_next_step'])

    time_series_loss_baseline = tf.metrics.mean_squared_error(time_series[:, 1:, :, :],
                                                              time_series[:, :-1, :, :])

    edge_type_accuracy = tf.metrics.accuracy(
        labels=edge_type, predictions=predictions['edge_type'])

    eval_metric_ops = {'trajectory_loss': trajectory_loss_eval,
                       'time_series_loss_baseline': time_series_loss_baseline,
                       'edge_type_accuracy': edge_type_accuracy}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                      eval_metric_ops=eval_metric_ops)


def main():
    with open(ARGS.config) as f:
        model_params = json.load(f)

    model_params['pred_steps'] = ARGS.pred_steps
    model_params['supervised'] = ARGS.supervised

    print('Loading data...')
    train_data, train_edge, test_data, test_edge = load_data(
        ARGS.data_dir, ARGS.data_transpose)

    mlp_gnn_regressor = tf.estimator.Estimator(
        model_fn=model_fn,
        params=model_params,
        model_dir=ARGS.log_dir
    )
    # Training
    if not ARGS.no_train:
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=train_data,
            y=train_edge,
            batch_size=ARGS.batch_size,
            num_epochs=None,
            shuffle=True
        )

        mlp_gnn_regressor.train(input_fn=train_input_fn,
                                steps=ARGS.train_steps)

    # Evaluation
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=test_data,
        y=test_edge,
        num_epochs=1,
        batch_size=ARGS.batch_size,
        shuffle=False
    )
    eval_results = mlp_gnn_regressor.evaluate(input_fn=eval_input_fn)
    # print("Validation set:", eval_results)

    # Prediction
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=test_data[:10],
        batch_size=ARGS.batch_size,
        shuffle=False)

    prediction = mlp_gnn_regressor.predict(input_fn=predict_input_fn)
    prediction = [(pred['state_next_step'], pred['edge_type_prob']) for pred in prediction]
    state_next_step, edge_type_prob = zip(*prediction)
    np.save(os.path.join(ARGS.log_dir, 'prediction_{}.npy'.format(ARGS.pred_steps)), state_next_step)
    np.save(os.path.join(ARGS.log_dir, 'infered_edge_type.npy'), edge_type_prob)


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
    parser.add_argument('--supervised', action='store_true', default=False,
                        help='use edge type labels to constrain encoder')
    parser.add_argument('--no-train', action='store_true', default=False,
                        help='skip training and use for evaluation only')
    ARGS = parser.parse_args()

    tf.logging.set_verbosity(tf.logging.INFO)

    main()
