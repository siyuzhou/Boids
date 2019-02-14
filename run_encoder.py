import os
import argparse
import json

import numpy as np
import tensorflow as tf

import gnn
from gnn.data import load_data


def encoder_model_fn(features, labels, mode, params):
    logits = gnn.encoder.encoder_fn[params['encoder']](
        features,
        params['edge_types'],
        params['encoder_params'],
        training=(mode == tf.estimator.ModeKeys.TRAIN))

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=-1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    accuracy = tf.metrics.accuracy(
        labels=labels, predictions=predictions["classes"], name="accuracy")

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

    eval_metric_ops = {"accuracy": accuracy}
    # mode == tf.estimator.ModeKeys.EVAL
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main():
    with open(ARGS.config) as f:
        model_params = json.load(f)

    mlp_encoder_classifier = tf.estimator.Estimator(
        model_fn=encoder_model_fn,
        params=model_params,
        model_dir=ARGS.log_dir)

    if ARGS.train:
        train_data, train_edge = load_data(ARGS.data_dir, ARGS.data_transpose, edge=True,
                                           prefix='train')
        train_edge = gnn.utils.one_hot(train_edge, model_params['edge_types'], np.float32)

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=train_data,
            y=train_edge,
            batch_size=ARGS.batch_size,
            num_epochs=None,
            shuffle=True
        )

        mlp_encoder_classifier.train(input_fn=train_input_fn,
                                     steps=ARGS.train_steps)

    # Evaluation
    if ARGS.eval:
        valid_data, valid_edge = load_data(ARGS.data_dir, ARGS.data_transpose, edge=True,
                                           prefix='valid')
        valid_edge = gnn.utils.one_hot(valid_edge, model_params['edge_types'], np.float32)

        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=valid_data,
            y=valid_edge,
            num_epochs=1,
            shuffle=False
        )
        eval_results = mlp_encoder_classifier.evaluate(input_fn=eval_input_fn)

        if not ARGS.verbose:
            print('Evaluation results: {}'.format(eval_results))

    # Predictoin
    if ARGS.test:
        test_data, test_edge = load_data(ARGS.data_dir, ARGS.data_transpose, edge=True,
                                         prefix='test')
        test_edge = gnn.utils.one_hot(test_edge, model_params['edge_types'], np.float32)

        pred_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=test_data,
            shuffle=False
        )
        prediction = mlp_encoder_classifier.predict(input_fn=pred_input_fn)
        predicted_edge_type = [pred['classes'] for pred in prediction]
        np.save(os.path.join(ARGS.log_dir, 'prediction.npy'), predicted_edge_type)


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
