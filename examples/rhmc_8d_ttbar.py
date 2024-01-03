import matplotlib
import matplotlib.pyplot as plt
import corner
import numpy as np
import pickle

import jax
import jax.numpy as jnp
from functools import partial

from rhmcjax.targets.madjax_target import MadjaxTarget
from rhmcjax.targets.target_util import read_madgraph_phasespace_points
from rhmcjax.rhmc.rhmc import rhmc


# Load target distribution
path_to_madjax_matrix_element = "rhmcjax.targets.madjax_ee_to_ttbar_wb"
dim = 8
name = "ee_to_ttbar_wb"
center_of_mass_energy = 1000 # [GeV]
model_parameters = {}
epsilon_boundary = 1e-5
target = MadjaxTarget(path_to_madjax_matrix_element, dim, name, center_of_mass_energy, model_parameters, epsilon_boundary)

n_flexible_dims = 2

inv_mass_matrix = jnp.array([0.1]*n_flexible_dims)
num_integration_steps = 60
num_chains = 10
num_samples_per_chain = 50_000
step_size = 1e-3

logdensity_fn = target.log_prob
key = jax.random.PRNGKey(1)

@partial(jax.jit, static_argnums=1)
def new_logensity_fn(x, x_const=jnp.array([0.5]*(dim - n_flexible_dims))):
    x = jnp.hstack([x, x_const])
    return logdensity_fn(x)

def run_rhmc(key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, key):
        state, info = kernel(key, state)
        return state, (state, info)

    keys = jax.random.split(key, num_samples)
    final_state, (states, info) = jax.lax.scan(one_step, initial_state, keys)

    return states, info

key, subkey = jax.random.split(key)
initial_positions = jax.random.uniform(subkey, shape=[num_chains, n_flexible_dims])
rhmc_sampler = rhmc(new_logensity_fn, step_size, inv_mass_matrix, num_integration_steps)

chains = []
for init_p in initial_positions:
    key, subkey = jax.random.split(key)
    initial_state = rhmc_sampler.init(init_p)
    state, info = run_rhmc(subkey, rhmc_sampler.step, initial_state, num_samples_per_chain)
    chains.append(state.position[info.is_accepted].squeeze())
samples = jnp.vstack(chains)

# Save samples to file
filename_rhmc = "./rhmc_samples_8d.pkl"
with open(filename_rhmc, "wb") as f:
    pickle.dump(samples, f)


filename_madgraph = './madgraph_samples_8d_ttbar.lhe.gz'
num_samples = len(samples)
samples_madgraph_full = read_madgraph_phasespace_points(filename_madgraph, target, num_samples)[:len(samples), :]
samples_madgraph = samples_madgraph_full[:, :n_flexible_dims]

# Plot histograms over marginals
if samples.shape[-1] == 2:
    plot_dims = [1, 3]
elif samples.shape[-1] == 4:
    plot_dims = [3,2]
elif samples.shape[-1] == 8:
    plot_dims = [3,3]

fig, axes = plt.subplots(plot_dims[0], plot_dims[1], figsize=(2*plot_dims[1], 2*plot_dims[0]))
axs = axes.flatten()
for i in range(samples.shape[-1]):
    axs[i].hist(samples[:,i], histtype='step', color="blue", label="RHMC Samples")
    axs[i].hist(samples_madgraph[:,i], histtype='step', color="orange", label="Madgraph Samples")
    axs[i].set_title(f"Dim {i}", fontsize=10)
    axs[i].get_yaxis().set_ticks([])
axs[i+1].legend(handles=[
            (matplotlib.lines.Line2D([], [], color="blue", label="RHMC Samples")),
            (matplotlib.lines.Line2D([], [], color="orange", label="Madgraph Samples"))
        ], loc="center", fontsize=10)
for i in range(samples.shape[-1], len(axs), 1):
    axs[i].axis('off')
plt.tight_layout()

filename_out = f'./rhmc_8d_ttbar_{n_flexible_dims}.png'
plt.savefig(filename_out)