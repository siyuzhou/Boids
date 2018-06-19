import os
import argparse

import numpy as np
import tensorflow as tf

from data_loader import load_data
from gnn import utils
from gnn.decoder import decoder_fn


# def train_input_fn(features, labels, batch_size):
#     dataset = tf.data.Dataset.from_tensor_slices((features, labels))
#     dataset = dataset.shuffle(1000).repeat().batch(batch_size)
#     return dataset.make_one_shot_iterator().get_next()


# def eval_input_fn(features, labels, batch_size):
#     if labels is None:
#         return tf.data.Dataset.from_tensor_slices(features).batch(batch_size)

#     return tf.data.Dataset.from_tensor_slices((features, labels)).batch(batch_size)

def decoder_model_fn(features, labels, mode, params):
    pred = decoder_fn[params['decoder']](features, params['decoder_params'],
                                         training=(mode == tf.estimator.ModeKeys.TRAIN))

    predictions = {'state_next_step': pred}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.mean_squared_error(features['time_series'][:, 1:, :, :],
                                        pred[:, :-1, :, :])

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0004)
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # eval accuracy
    eval_metric_ops = {'eval_loss': tf.metrics.mean_squared_error(features['time_series'][:, 1:, :, :],
                                                                  pred[:, :-1, :, :])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main():
    model_params = {
        'decoder': 'mlp',
        'decoder_params': {
            'hidden_units': [ARGS.hidden_units, ARGS.hidden_units],
            'dropout': ARGS.dropout,
            'batch_norm': ARGS.batch_norm,
        }
    }

    print('Loading data...')
    train_data, train_edge, test_data, test_edge = load_data(
        ARGS.data_dir, ARGS.data_transpose)

    train_edge = utils.one_hot(train_edge, ARGS.edge_types, np.float32)
    test_edge = utils.one_hot(test_edge, ARGS.edge_types, np.float32)

    mlp_encoder_classifier = tf.estimator.Estimator(
        model_fn=decoder_model_fn,
        params=model_params,
        model_dir=ARGS.log_dir)

    if not ARGS.no_train:
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'time_series': train_data,
               'edge_type': train_edge},
            y=train_edge,
            batch_size=ARGS.batch_size,
            num_epochs=None,
            shuffle=True
        )

        mlp_encoder_classifier.train(input_fn=train_input_fn,
                                     steps=ARGS.steps)

    # Evaluation
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'time_series': test_data,
           'edge_type': test_edge},
        y=test_edge,
        num_epochs=1,
        shuffle=False
    )
    eval_results = mlp_encoder_classifier.evaluate(input_fn=eval_input_fn)
    print("Validation set:", eval_results)

    # Prediction
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'time_series': test_data[:1],
           'edge_type': test_edge[:1]},
        shuffle=False)

    prediction = mlp_encoder_classifier.predict(input_fn=predict_input_fn)
    prediction = np.array([pred['state_next_step'] for pred in prediction])
    np.save(os.path.join(ARGS.log_dir, 'prediction.npy'), prediction)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str,
                        help='data directory')
    parser.add_argument('--data-transpose', type=int, nargs=4, default=None,
                        help='axes for data transposition')
    parser.add_argument('--log-dir', type=str,
                        help='log directory')
    parser.add_argument('--edge-types', type=int,
                        help='number of edge types')
    parser.add_argument('--hidden-units', type=int,
                        help='number of units in a hidden layer')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='dropout rate')
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
