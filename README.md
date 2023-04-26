# exoFlows
Allows the user to easily train normalizing flows in multiple rounds to perform inference. Specifically written to work in conjunction with ARCiS.

## Requirements
[ARCiS](https://github.com/michielmin/ARCiS): Contact [Michiel Min](mailto:m.min@sron.nl).

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
- `embedding`: Switches on a neural network (called embedding network) to reduce the dimensionality of the data. Off by default.
- `embed_size`: The size of the embedding network's output.
- `Ztol`: Specifies the maximum change in log(Z) to stop inference. For parameter estimation, large values (e.g. 1, set by default) are fine. For model comparison, it should be <1.
- `processes`: The number of parallel instances of ARCiS running to compute the training set. Can be used to speed up the inference if multiple cores are available.
- `dont_reject`: By default, a round will be rejected if the evidence doesn't improve with respect to the previous one. This option turns it off. 


### Output
exoFlows produces a lot of output files:

- `post_equal_weights.txt`: File containing 10,000 posterior points.
- `arcis_spec_round.p`: `pickle` file containing a dictionary with the forward models for all rounds. If there are multiple observations, the modeled spectra for the different observations are concatenated one after the other, so the shape of the arrays is `[number of models per round, total number of wavelength bins]`.
- `Y_round.p`: `pickle` file containing a dictionary with the parameters (normalized) used to compute the forward models for all rounds.
- `P.npy`: `numpy` array containing the pressure levels of the model.
- `T_round_XX.npy`: `numpy` arrays containing the temperature structure for all forward models in round XX.
- `samples.p`: `pickle` file containing 10,000 samples from the posterior computed at each round.
- `posteriors.pt`: File containing the `torch` distributions for the posterior predicted at every round. Useful to sample and obtain log(P) for any point.
- `xscaler.p`: `pickle` file containing the normalizing class for the spectra.
- `yscaler.p`: `pickle` file containing the normalizing class for the parameters.
