import jax.numpy as jnp

import blackjax.mcmc.metrics as metrics
from blackjax.mcmc.integrators import IntegratorState

from rhmcjax.rhmc.rhmc import init, reflection_velocity_verlet
from rhmcjax.targets.gaussians_on_circle import GaussiansOnCircle

def test_reflection_velocity_verlet():
    constrained_target = GaussiansOnCircle(restrict_to_unit_hypercube=True)
    log_density_fn = constrained_target.log_prob
    inverse_mass_matrix = jnp.array([0.5, 0.01])
    _, kinetic_energy_fn, _ = metrics.gaussian_euclidean(
            inverse_mass_matrix
        )
    one_step_fn = reflection_velocity_verlet(log_density_fn, kinetic_energy_fn)

    position = jnp.array([0.25, 0.25])
    momentum = jnp.array([0.5, 0.2])
    hmc_state= init(position, log_density_fn)
    integrator_state = IntegratorState(
            hmc_state.position, momentum, hmc_state.logdensity, hmc_state.logdensity_grad
        )
    step_size = 1.
    out = one_step_fn(integrator_state, step_size)

    assert jnp.all(out.position <= 1.) and jnp.all(out.position >= 0.)
    