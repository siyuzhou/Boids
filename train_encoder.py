import argparse

import numpy as np
import tensorflow as tf

import gnn
from data_loader import load_data


def encoder_model_fn(features, labels, mode, params):
    logits = gnn.encoder.encoder_fn[params['encoder']](
        features,
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
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0004)
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {"accuracy": accuracy}
    # mode == tf.estimator.ModeKeys.EVAL
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main():
    model_params = {
        'encoder': 'mlp',
        'encoder_params': {
            'hidden_units': [ARGS.hidden_units, ARGS.hidden_units],
            'dropout': ARGS.dropout,
            'batch_norm': ARGS.batch_norm,
            'edge_types': ARGS.edge_types
        }
    }

    print('Loading data...')
    train_data, train_edge, test_data, test_edge = load_data(
        ARGS.data_dir, ARGS.data_transpose)

    mlp_encoder_classifier = tf.estimator.Estimator(
        model_fn=encoder_model_fn,
        params=model_params,
        model_dir=ARGS.log_dir)

    if not ARGS.no_train:
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=train_data,
            y=train_edge,
            batch_size=ARGS.batch_size,
            num_epochs=None,
            shuffle=True
        )

        mlp_encoder_classifier.train(input_fn=train_input_fn,
                                     steps=ARGS.steps)

    # Evaluation
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=test_data,
        y=test_edge,
        num_epochs=1,
        shuffle=False
    )
    eval_results = mlp_encoder_classifier.evaluate(input_fn=eval_input_fn)
    print("Validation set:", eval_results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str,
                        help='data directory')
    parser.add_argument('--data-transpose', type=int, nargs=4, default=None,
                        help='axes for data transposition')
    parser.add_argument('--log-dir', type=str,
                        help='log directory')
    parser.add_argument('--hidden-units', type=int,
                        help='number of units in a hidden layer')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='dropout rate')
    parser.add_argument('--edge-types', type=int,
                        help='number of edge types')
    parser.add_argument('--steps', type=int, default=1000,
                        help='number of training steps')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--batch-norm', action='store_true', default=False,
                        help='turn on batch normalization')
    parser.add_argument('--no-train', action='store_true', default=False,
                        help='skip training and use for evaluation only')
    ARGS = parser.parse_args()

    tf.logging.set_verbosity(tf.logging.INFO)

    main()
