# FlopPITy
(normalising **Flo**w exo**p**lanet **P**arameter **I**nference **T**oolk**y**t)

Allows the user to easily train normalizing flows in multiple rounds to perform atmospheric retrievals of exoplanets. Specifically written to work in conjunction with [ARCiS](https://github.com/michielmin/ARCiS). [TauREx](https://taurex3-public.readthedocs.io/en/latest/) and [petitRADTRANS](https://petitradtrans.readthedocs.io/en/latest/) versions coming soon.

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
A basic example of a script to run a retrieval can be found in `example.sh`.

Here's a brief description of the most important options:
- `input`: Path to ARCiS input file.
- `output`: Directory where the output will be saved.
- `n_rounds`: Number of rounds to train for.
- `samples_per_round`: Number of forward models to use for training at each round.
- `processes`: The number of parallel instances of ARCiS running to compute the training set. Can be used to speed up the inference if multiple cores are available.
- `resume`: Use this option to resume a previous retrieval. 


### Output
exoFlows produces a lot of output files:

- `post_equal_weights.txt`: File containing 10,000 posterior points.
- `arcis_spec_round.p`: `pickle` file containing a dictionary with the forward models computed for training. If there are multiple observations, the modeled spectra for the different observations are concatenated one after the other, so the shape of the arrays is `[number of models per round, total number of wavelength bins]`.
- `Y_round.p`: `pickle` file containing a dictionary with the parameters (normalized) used to compute the forward models.
- `P.npy`: `numpy` array containing the pressure levels of the model.
- `T_round_XY.npy`: `numpy` arrays containing the temperature structure for all forward models computed for process Y in training round X.
- `samples.p`: `pickle` file containing samples from the posterior computed at each round.
- `posteriors.pt`: File containing the `torch` distributions for the posterior estimated at every round.
- `xscaler.p`: `pickle` file containing the normalizing class for the spectra.
- `yscaler.p`: `pickle` file containing the normalizing class for the parameters.
