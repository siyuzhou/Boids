# Graph Neural Network model

A TensorFlow implementation of Graph Neural Network from ([Kipf, 2018](https://arxiv.org/abs/1802.04687)) for predicting the trajectories of a particle system.

## Package Dependencies

- TensorFlow >= 1.8
- NumPy

## Usage

By default, trajectory data is saved as a `.npy` file in the shape of $[s, t, n, d]$, where $s$ is the number of simulations, $t$ is the number of time steps, $n$ is the number of particles, and $d$ is the dimension of one particle's state vector. Labeled edge type data is saved also as a `.npy` file, with the shape of $[s, n, n]$. For a network with `k` distinct edge types, the edge type tensor is filled with integers in range $[0, k)$.

`train_encoder.py` trains the encoder with timeseries data as input and computes inferred edge types.

`train_decoder.py` trains the decoder with timeseries data and labeled edge types as inputs and computes predicted states of next steps.

`train.py` trains the encoder and decoder combined with only timeseries as input. The output contains unsupervised edge type inference, along with predicted states of next steps.

For system arguments, see the code, or use the `-h` switch.

Example

```bash
python train.py --data-dir=data/ --log-dir=logs/ --config=data/config.json --train-steps=5000 -pred-steps=5
```
