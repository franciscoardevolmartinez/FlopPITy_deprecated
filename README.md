# MulteXBI
 Multiround eXoplanet sBI

## Requirements
[ARCiS](https://github.com/michielmin/ARCiS): Contact Michiel Min.

The following python packages are needed:
- [`numpy`](https://numpy.org/install/)
- [`matplotlib`](https://matplotlib.org/stable/users/getting_started/)
- [`corner`](https://corner.readthedocs.io/en/latest/install/)
- [`torch`](https://pytorch.org/get-started/locally/#mac-installation)
- [`sbi`](https://www.mackelab.org/sbi/install/)
- [`sklearn`](https://scikit-learn.org/stable/install.html)
- [`tqdm`](https://github.com/tqdm/tqdm#installation)

## Quick start guide
The best way to run a retrieval is with a script. The most basic example of such a script can be found in `example.sh`.
Inference stops when the evidence changes by less than a certain tolerance, or when a number of rounds equal to num_rounds has been reached.

Here's a brief description of some of the most important options:
- `device`: Used to specify the device on which the flows are trained. Can be `cpu` or `gpu`. Default: `cpu`.
- `embedding`: Switches on a neural network (called embedding network) to reduce the dimensionality of the data.
- `embed_size`: The size of the embedding network's output.
- `Ztol`: Specifies the maximum change in log(Z) to stop inference. For parameter estimation, large values (e.g. 10, set by default) are fine. For model comparison, it should be <1.
- `processes`: The number of parallel instances of ARCiS running to compute the training set. Can be used to speed up the inference if multiple cores are available.
