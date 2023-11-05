"""Rewrite of Blackjax HMC kernel to include reflection."""
from typing import Callable, NamedTuple, Tuple, Union

import chex
import jax

import blackjax.mcmc.metrics as metrics
import blackjax.mcmc.proposal as proposal
import blackjax.mcmc.trajectory as trajectory
from blackjax.mcmc.hmc import HMCInfo, HMCState
from blackjax.mcmc.trajectory import hmc_energy
from blackjax.types import Array, PRNGKey, PyTree

from blackjax.mcmc.integrators import EuclideanIntegrator, EuclideanKineticEnergy, IntegratorState


def reflection_velocity_verlet(
    logdensity_fn: Callable,
    kinetic_energy_fn: EuclideanKineticEnergy,
) -> EuclideanIntegrator:
    a1 = 0
    b1 = 0.5
    a2 = 1 - 2 * a1

    logdensity_and_grad_fn = jax.value_and_grad(logdensity_fn)
    kinetic_energy_grad_fn = jax.grad(kinetic_energy_fn)

    def one_step(state: IntegratorState, step_size: float) -> IntegratorState:
        position, momentum, logdensity_grad = state.position, state.momentum, state.logdensity_grad

        momentum = jax.tree_util.tree_map(
            lambda momentum, logdensity_grad: momentum
            + b1 * step_size * logdensity_grad,
            momentum,
            logdensity_grad,
        )

        kinetic_grad = kinetic_energy_grad_fn(momentum)
        position = jax.tree_util.tree_map(
            lambda position, kinetic_grad: position + a2 * step_size * kinetic_grad,
            position,
            kinetic_grad,
        )
        # TODO: include reflection here

        logdensity, logdensity_grad = logdensity_and_grad_fn(position)

        momentum = jax.tree_util.tree_map(
            lambda momentum, logdensity_grad: momentum
            + b1 * step_size * logdensity_grad,
            momentum,
            logdensity_grad,
        )

        return IntegratorState(position, momentum, logdensity, logdensity_grad)

    return one_step


def kernel(
    integrator: Callable = reflection_velocity_verlet,
    divergence_threshold: float = 1000,
):

    def one_step(
        rng_key: PRNGKey,
        state: HMCState,
        logdensity_fn: Callable,
        step_size: float,
        inverse_mass_matrix: Array,
        num_integration_steps: int,
    ) -> Tuple[HMCState, HMCInfo]:
        """Generate a new sample with the HMC kernel."""

        momentum_generator, kinetic_energy_fn, _ = metrics.gaussian_euclidean(
            inverse_mass_matrix
        )
        symplectic_integrator = integrator(logdensity_fn, kinetic_energy_fn)
        proposal_generator = hmc_proposal(
            symplectic_integrator,
            kinetic_energy_fn,
            step_size,
            num_integration_steps,
            divergence_threshold,
        )

        key_momentum, key_integrator = jax.random.split(rng_key, 2)

        position, logdensity, logdensity_grad = state
        momentum = momentum_generator(key_momentum, state.position)

        integrator_state = IntegratorState(
            position, momentum, logdensity, logdensity_grad
        )
        proposal, info = proposal_generator(key_integrator, integrator_state)
        proposal = HMCState(
            proposal.position, proposal.logdensity, proposal.logdensity_grad
        )

        return proposal, info

    return one_step


def hmc_proposal(
    integrator: Callable,
    kinetic_energy: Callable,
    step_size: Union[float, PyTree],
    num_integration_steps: int = 1,
    divergence_threshold: float = 1000,
    *,
    sample_proposal: Callable = proposal.static_binomial_sampling,
) -> Callable:
    
    build_trajectory = trajectory.static_integration(integrator)
    init_proposal, generate_proposal = proposal.proposal_generator(
        hmc_energy(kinetic_energy), divergence_threshold
    )

    def generate(
        rng_key, state: IntegratorState
    ) -> Tuple[IntegratorState, HMCInfo]:
        """Generate a new chain state."""
        end_state = build_trajectory(state, step_size, num_integration_steps)
        end_state = flip_momentum(end_state)
        proposal = init_proposal(state)
        new_proposal, is_diverging = generate_proposal(proposal.energy, end_state)
        sampled_proposal, *info = sample_proposal(rng_key, proposal, new_proposal)
        do_accept, p_accept = info

        info = HMCInfo(
            state.momentum,
            p_accept,
            do_accept,
            is_diverging,
            new_proposal.energy,
            new_proposal,
            num_integration_steps,
        )

        return sampled_proposal.state, info

    return generate


def flip_momentum(
    state: IntegratorState
) -> IntegratorState:
    flipped_momentum = jax.tree_util.tree_map(lambda m: -1.0 * m, state.momentum)
    return state._replace(momentum=flipped_momentum)
