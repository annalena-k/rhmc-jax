# `rhmc-jax`: A JAX package for Reflection Hamiltonian Monte Carlo (RHMC)
`rhmc-jax` is a JAX implementation of Reflection HMC based on the paper [Reflection, Refraction, and Hamiltonian Monte Carlo](https://papers.nips.cc/paper_files/paper/2015/hash/8303a79b1e19a194f1875981be5bdb6f-Abstract.html).

Although HMC is usually performed on infinite support, specific settings exist where the distribution of interest is defined on a constrained space. While a transformation such as a sigmoid or tanh can be applied to map the limited support to unconstrained intervals in most cases, this is not high energy physics (HEP). The reason for this is that divergences in the distribution can appear at the boundary which would lead to non-zero probability mass at $+\infty$ or $-\infty.$

In order to perform HMC on HEP distributions, HMC has to be performed on the unit hypercube of arbitary dimension. If standard HMC is used, chains need to be rejected that land outside the defined support which results in a decreased acceptance rate. Employing reflection at the boundaries of the unit hypercube can mitigate this problem and increase the efficiency. 

The modified version of HMC is called Reflection HMC (RHMC) and it is based on the paper [Reflection, Refraction, and Hamiltonian Monte Carlo](https://papers.nips.cc/paper_files/paper/2015/hash/8303a79b1e19a194f1875981be5bdb6f-Abstract.html). 
This package is built on and extends the HMC implementation of [blackjax](https://blackjax-devs.github.io/blackjax/) to include reflection.

## Installation
The `rhmc-jax` package can be installed in its own environment via the following steps: Firstly, clone the repository in its intended folder by executing
```
git clone https://github.com/annalena-k/rhmc-jax.git
```
Next, create a virtual environement via, e.g.,
```
python3 -m venv rhmc-venv
```
To activate the environment with the name `rhmc-venv`, execute
```
source rhmc-venv/bin/activate
```
Now, we enter the new repository
```
cd rhmc-jax
```
and install the package in editable mode using

```
pip install -e .
```
This will automatically install all dependenies listed in `pyproject.toml`.

## Usage
Since `rhmc-jax` extends the HMC implementation of [`blackjax`](https://blackjax-devs.github.io/blackjax/) to RHMC, its usage is equivalent to `blackjax` and the original [`blackjax` documentation](https://blackjax.readthedocs.io/en/latest/) might be helpful.
### Minimal `RHMC` example
```python
import jax
import jax.numpy as jnp
from rhmcjax.rhmc.rhmc import rhmc
from rhmcjax.targets.gaussians_on_circle import GaussiansOnCircle

# Define target distribution
target = GaussiansOnCircle()
dim = 2
# Specify parameters for RHMC
inv_mass_matrix = jnp.array([0.1]*dim)
num_integration_steps = 60
num_chains = 10
num_samples_per_chain = 1_000
step_size = 1e-3
logdensity_fn = target.log_prob
key = jax.random.PRNGKey(1)

# Define a function to run RHMC
def run_rhmc(key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, key):
        state, info = kernel(key, state)
        return state, (state, info)
    keys = jax.random.split(key, num_samples)
    final_state, (states, info) = jax.lax.scan(one_step, initial_state, keys)
    return states, info

# Initialize RHMC
key, subkey = jax.random.split(key)
initial_positions = jax.random.uniform(subkey, shape=[num_chains, dim])
rhmc_sampler = rhmc(logdensity_fn, step_size, inv_mass_matrix, num_integration_steps)
# Run RHMC for all chains
chains = []
for init_p in initial_positions:
    key, subkey = jax.random.split(key)
    initial_state = rhmc_sampler.init(init_p)
    state, info = run_rhmc(subkey, rhmc_sampler.step, initial_state, num_samples_per_chain)
    chains.append(state.position[info.is_accepted].squeeze())
mcmc_chains = jnp.vstack(chains)
```

To illustrate the use of `rhmc-jax`, several examples are provided which will be introduced in the following.

## Examples
The [`examples/`](https://github.com/annalena-k/rhmc-jax/tree/main/examples) directory contains several use cases of `blackjax` and `rhmc-jax`.

- The notebook [`hmc_2d_gaussians.ipynb`](https://github.com/annalena-k/rhmc-jax/blob/main/examples/hmc_2d_gaussians.ipynb) introduces the standard distribution of Gaussians located on a circle and shows how HMC is performed with `blackjax`.
- The noteook [`rhmc_2d_gaussians.ipynb`](https://github.com/annalena-k/rhmc-jax/blob/main/examples/rhmc_2d_gaussians.ipynb) illustrates how the acceptance rate of HMC decreases if this distribution is restricted to the unit square. This motivates the use of RHMC and a direct comparison shows that including reflection improves the acceptance rate to approximately 99 %. 
![til](./images/rhmc.gif)
- The file [`reflection_algorithm_in_detail.pynb`](https://github.com/annalena-k/rhmc-jax/blob/main/examples/reflection_algorithm_in_detail.ipynb) introduces and visualizes details of the reflection algorithm employed in this package. It includes code for visualizing subsequent reflections at the boundary of the unit square.

![til](./images/reflection.gif)

Since employing RHMC is motivated by the use case of HEP matrix elements, we showcase RHMC with a complex HEP distribution:
- The 3-body decay $\Lambda_c^+ \rightarrow pK^- \pi^+$ is based on [this publication](https://doi.org/10.1007/JHEP07(2023)228) and the implementation depends on the publicly available [code](https://doi.org/10.5281/zenodo.7544989). The matrix element is defined on the 2D phasespace and can be visualized in a Dalitz plot. It has a complex structure resulting from multiple resonances in the different decay channels and their interference. The notebook [`rhmc_2d_Lc2pKpi.ipynb`](https://github.com/annalena-k/rhmc-jax/blob/main/examples/rhmc_2d_Lc2pKpi.ipynb) shows how the Dalitz plot can be transformed to the unit square and provides RHMC results for this challenging distribution.
![alt text](https://github.com/annalena-k/rhmc-jax/blob/main/images/Lc2pKpi.png)
