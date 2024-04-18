# FlopPITy
(normalising **Flo**w exo**p**lanet **P**arameter **I**nference **T**oolk**y**t)

Allows the user to easily train normalizing flows in multiple rounds to perform atmospheric retrievals of exoplanets. Specifically written to work in conjunction with [ARCiS](https://github.com/michielmin/ARCiS). [TauREx](https://taurex3-public.readthedocs.io/en/latest/) and [petitRADTRANS](https://petitradtrans.readthedocs.io/en/latest/) versions coming soon.

## Requirements
[ARCiS](https://github.com/michielmin/ARCiS)

To take full advantage of FlopPITy, compile ARCiS in single core mode and set `processes` to the number of cores available (see below).

Please get in touch if you'd like to use FlopPITy with an atmospheric simulator of your choice.

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

## Advanced options
- `xnorm`: Standardizes the spectra (I recommend using it).
- `ynorm`: Normalizes the parameters by transforming the priors so the bounds are -1 and 1 (I recommend using it).
- `custom_nsf`: Use this option if you'd like to tweak the hyperparameters of the neural spline flow (NSF).
- `transforms`: Number of transformations in the NSF.
- `bins`: Number of bins in the transformation.
- `hidden`: Number of neurons per layer in the res-net.
- `blocks`: Number of residual blocks in the res-net. Each block has 2 dense layers.
- `atoms`: Number of atoms to use for training.
- `patience`: Number of traning epochs without improvement to wait for before finishing traning.
- `reuse_prior_samples`: Use this option to reuse the first round's training set from another retrieval. Setup must be identical to current retrieval.
- `prior_dir`: Directory to take the prior samples from.
- `fit_offset`: Whether to fit an additive offset to an observation if more than one are fitted simultaneously.
- `prior_offset`: String with comma separated bounds for the offset priors.

  
- `input2`: You can input a 2nd ARCiS input file to mix models (e.g. binary objects, two terminators...).
- `binary`: If you use this option, the contributions from the two spectra are added. Otherwise they are averaged.
- `fit_frac`: Only used if `binary` is not used, the two spectra are averaged with a specific weight, which is fitted for.
- `n_global`: Number of parameters in `input` and `input2` to keep the same between both spectra (e.g. if trying to fit two terminators to a transmission spectrum, you want to keep the radius and gravity the same for both). When setting up both ARCiS input files, the 'global' parameters go first (and they must have identical priors). The other parameters will be fit independently for both contributions. 



### Output
FlopPITy produces a lot of output files:

- `post_equal_weights.txt`: File containing 10,000 posterior points.
- `arcis_spec_round.p`: `pickle` file containing a dictionary with the forward models computed for training. If there are multiple observations, the modeled spectra for the different observations are concatenated one after the other, so the shape of the arrays is `[number of models per round, total number of wavelength bins]`.
- `Y_round.p`: `pickle` file containing a dictionary with the parameters (normalized) used to compute the forward models.
- `P.npy`: `numpy` array containing the pressure levels of the model.
- `T_round_XY.npy`: `numpy` arrays containing the temperature structure for all forward models computed for process Y in training round X.
- `samples.p`: `pickle` file containing samples from the posterior computed at each round.
- `map.dat`: ARCiS input file to compute the forward model corresponding to the MAP parameters.
- `posteriors.pt`: File containing the `torch` distributions for the posterior estimated at every round.
- `xscaler.p`: `pickle` file containing the normalizing class for the spectra.
- `yscaler.p`: `pickle` file containing the normalizing class for the parameters.
