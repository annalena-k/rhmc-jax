import jax
import jax.numpy as jnp
from rhmcjax.rhmc.rhmc import rhmc
from rhmcjax.targets.gaussians_on_circle import GaussiansOnCircle

def test_minimal_example():
    """ Test minimal example that is included in README.md
    """
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

    assert len(mcmc_chains) > 1