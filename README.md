Code for "An Exponentially Converging Particle Method for the Mixed Nash Equilibrium of Continuous Games"
===

## Set up
Requires julia and jupyter notebook. 
The required julia packages are listed in the project file `Project.toml`.

### Detailed steps for a possible setup using conda
1. Install conda and julia. (Tested with conda v4.14.0, julia v1.6.3.)
2. In Unix shell:
```bash
conda create -n particle_MNE
conda activate particle_MNE
conda install jupyter # or conda install -c conda-forge jupyterlab
```
3. In julia REPL:
```julia
using Pkg # or type ] to enter Pkg REPL-mode
pkg"activate ."
pkg"instantiate"
```
4. In Unix shell:
```bash
# conda activate particle_MNE
jupyter notebook # or jupyter lab
```

## Reproducing the figures of the paper
Run the notebooks in the directory `generate_plots/`. The figures are saved in png format to the relevant subdirectory of `results/`.

The notebooks are almost identical to the ones described below (e.g., `generate_plots/particle_random_fourier_1D_genplots.ipynb` is almost identical to `particle_random_fourier_1D.ipynb`), with adjustments only to the plotting parameters.

## Experiments

Set `extrasteps=1` for CP-MDA (or Mirror Descent-Ascent for finite games), `extrasteps=2` for CP-MP (or Mirror Prox for finite games).

The dependency of the exponential convergence rate with respect to the step-size can be tested easily by setting `scaling` and varying `alpha`, and checking whether the slope of the NI error is independent of `alpha`; equivalently (provided that the local convergence regime always kicks in immediately), whether the NI error at the last iterate is independent of `alpha`.

### Finite games (Mirror Descent-Ascent, Mirror Prox)
- Notebook `weightonly_gaussian_gmat.ipynb`: finite game with random payoff matrix with i.i.d Gaussian components.
- Notebook `weightonly_random_fourier_1D.ipynb`: finite game with random payoff matrix obtained by discretizing a periodic Gaussian process payoff function, in dimension dimx=1 and dimy=1, on a uniform grid of $[0,1] \times [0,1]$.

Make use of functions defined in `utils/` and in `weightonly/`.

### Payoff functions drawn from a periodic Gaussian process (dimx=dimy=1)
Notebook `particle_random_fourier_1D.ipynb`.
Makes use of functions defined in `utils/` and in `particle_1D/`.

### A synthetic example where the continuous-time flow of CP-PP does not converge
Notebook `particle_YPhsieh_1D.ipynb`.
Makes use of functions defined in `utils/` and in `particle_1D/`.

### Max-F1-margin classification with 2LNNs
Notebook `maxF1margin.ipynb`. The code is self-contained.

### Distributionally-robust classification with 2LNNs
Notebook `distrib_robust.ipynb`. The code is self-contained.

## Additional code
For reference we also provide an implementation of CP-MP for computing the MNE of continuous games with payoff functions drawn from a periodic Gaussian process in dimension dimx, dimy > 1 in the notebook `particle_random_fourier_multidim`, which uses functions defined in `particle_multidim/`.
It is possible to check manually that the CP-MP iterates converge to a sparse measure, but evaluating their NI errors accurately takes a very long time.
