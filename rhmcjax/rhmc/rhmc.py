"""Rewrite of Blackjax HMC kernel to include reflection."""
from typing import Callable
import jax

import blackjax.mcmc.metrics as metrics
from blackjax.base import SamplingAlgorithm
from blackjax.mcmc.hmc import HMCInfo, HMCState, hmc_proposal, init
from blackjax.types import Array, ArrayLikeTree, PRNGKey
from blackjax.mcmc.integrators import EuclideanIntegrator, EuclideanKineticEnergy, IntegratorState

from rhmcjax.rhmc.intersection_with_boundary import find_next_intersection, reflection_necessary


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

        # Half-step evolution of momentum (stays the same)
        momentum = jax.tree_util.tree_map(
            lambda momentum, logdensity_grad: momentum
            + b1 * step_size * logdensity_grad,
            momentum,
            logdensity_grad,
        )

        def cond_fn(val):
            position, momentum, t = val
            return reflection_necessary(position, momentum, kinetic_energy_grad_fn, step_size, t)

        def body_fn(val):
            position, momentum, t = val
            position, momentum, t_x, ind_boundary = find_next_intersection(position, momentum, kinetic_energy_grad_fn, step_size, t)
            t += t_x
            # Reverse momentum component perpendicular to boundary
            momentum = momentum.at[ind_boundary].set(-momentum[ind_boundary])
            return [position, momentum, t]

        t = 0
        init_val = [position, momentum, t]
        position, momentum, t = jax.lax.while_loop(cond_fn, body_fn, init_val)

        # Update final position (after all reflections)
        kinetic_grad = kinetic_energy_grad_fn(momentum)
        position = jax.tree_util.tree_map(
            lambda position, kinetic_grad: position + a2 * (step_size - t) * kinetic_grad,
            position,
            kinetic_grad,
        )

        # Half-step evolution of momentum (stays the same)
        logdensity, logdensity_grad = logdensity_and_grad_fn(position)
        momentum = jax.tree_util.tree_map(
            lambda momentum, logdensity_grad: momentum
            + b1 * step_size * logdensity_grad,
            momentum,
            logdensity_grad,
        )

        return IntegratorState(position, momentum, logdensity, logdensity_grad)

    return one_step


def build_rhmc_kernel(
    integrator: Callable = reflection_velocity_verlet,
    divergence_threshold: float = 1000,
):
    """Build a RHMC kernel.

    Parameters
    ----------
    integrator
        The symplectic integrator to use to integrate the Hamiltonian dynamics. Default: reflection_velocity_verlet
    divergence_threshold
        Value of the difference in energy above which we consider that the transition is divergent.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """
    def kernel(
        rng_key: PRNGKey,
        state: HMCState,
        logdensity_fn: Callable,
        step_size: float,
        inverse_mass_matrix: Array,
        num_integration_steps: int,
    ) -> tuple[HMCState, HMCInfo]:
        """Generate a new sample with the RHMC kernel."""

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
        momentum = momentum_generator(key_momentum, position)
        integrator_state = IntegratorState(
            position, momentum, logdensity, logdensity_grad
        )
        proposal, info = proposal_generator(key_integrator, integrator_state)
        proposal = HMCState(
            proposal.position, proposal.logdensity, proposal.logdensity_grad
        )

        return proposal, info

    return kernel

class rhmc:
    """Implements the (basic) user interface for the RHMC kernel.

    The general rhmc kernel builder can be
    cumbersome to manipulate. Since most users only need to specify the kernel
    parameters at initialization time, we provide a helper function that
    specializes the general kernel.

    We also add the general kernel and state generator as an attribute to this class so
    users only need to pass `blackjax.hmc` to SMC, adaptation, etc. algorithms.

    Examples
    --------

    A new RHMC kernel can be initialized and used with the following code:

    .. code::

        rhmc = rhmc(logdensity_fn, step_size, inverse_mass_matrix, num_integration_steps)
        state = rhmc.init(position)
        new_state, info = rhmc.step(rng_key, state)

    Kernels are not jit-compiled by default so you will need to do it manually:

    .. code::

       step = jax.jit(rhmc.step)
       new_state, info = step(rng_key, state)

    Should you need to you can always use the base kernel directly:


    .. code::

       from rhmcjax.rhmc.rhmc import build_rhmc_kernel, reflection_velocity_verlet, rhmc

       rhmc_kernel = build_rhmc_kernel(reflection_velocity_verlet)
       state = rhmc.init(position, logdensity_fn)
       state, info = rhmc_kernel(rng_key, state, logdensity_fn, step_size, inverse_mass_matrix, num_integration_steps)

    Parameters
    ----------
    logdensity_fn
        The log-density function we wish to draw samples from.
    step_size
        The value to use for the step size in the symplectic integrator.
    inverse_mass_matrix
        The value to use for the inverse mass matrix when drawing a value for
        the momentum and computing the kinetic energy.
    num_integration_steps
        The number of steps we take with the symplectic integrator at each
        sample step before returning a sample.
    divergence_threshold
        The absolute value of the difference in energy between two states above
        which we say that the transition is divergent. The default value is
        commonly found in other libraries, and yet is arbitrary.
    integrator
        (algorithm parameter) The symplectic integrator to use to integrate the trajectory.\

    
    Returns
    -------
    A ``SamplingAlgorithm``.
    """

    init = staticmethod(init)
    build_kernel = staticmethod(build_rhmc_kernel)

    def __new__(  # type: ignore[misc]
        cls,
        logdensity_fn: Callable,
        step_size: float,
        inverse_mass_matrix: Array,
        num_integration_steps: int,
        *,
        divergence_threshold: int = 1000,
        integrator: Callable = reflection_velocity_verlet,
    ) -> SamplingAlgorithm:
        kernel = cls.build_kernel(integrator, divergence_threshold)

        def init_fn(position: ArrayLikeTree):
            return cls.init(position, logdensity_fn)

        def step_fn(rng_key: PRNGKey, state):
            return kernel(
                rng_key,
                state,
                logdensity_fn,
                step_size,
                inverse_mass_matrix,
                num_integration_steps,
            )

        return SamplingAlgorithm(init_fn, step_fn)
    