from typing import Optional, Union
import chex
import jax
import jax.numpy as jnp

from blackjax.mcmc.integrators import EuclideanKineticEnergy

EuclideanKineticEnergyGrad = EuclideanKineticEnergy

def check_outside_of_boundary(
        position: chex.Array
    ) -> chex.Array:
    """ Check whether position is within D-dimensional unit hypercube boundaries or outside.
    Input: position: Array of shape [D]
    Output: indices: Array of shape [D], 0 if position is inside for this dimension, -1. if position < 0, +1 if position > 1.
    """
    # TODO: Adjust for arbitrary PyTrees, currently only working for Arrays
    ind_smaller_0 = jnp.where(position<0, -1., 0.)
    ind_larger_1 = jnp.where(position>1, 1., 0.)
    indices = ind_larger_1 + ind_smaller_0

    return indices

def check_position_on_boundary(
        position: chex.Array, 
        ) -> chex.Array:
    """ Check whether positon is located directly on the boundary of the D-dimensional unit hypercube.
    If position[i]==0, inds_on_boundary = -1., if position[i]==1, inds_on_boundary = 1.
    Input: position: Array of shape [D], current position vector
    Output: inds_on_boundary: Array of shape [D]
    """
    # TODO: Adjust for arbitrary PyTrees, currently only working for Arrays
    ind_on_boundary_0 = jnp.where(position==0, -1., 0.)
    ind_on_boundary_1 = jnp.where(position==1, 1., 0.)
    inds_on_boundary = ind_on_boundary_0 + ind_on_boundary_1

    return inds_on_boundary

def revert_momentum_for_boundary_positions(
        on_boundary: chex.Array, 
        momentum: chex.Array
        ) -> chex.Array:
    """If position is located exactly on the boundary (at 0. or 1.) and the momentum points outside of the unit hypercube,
       revert momentum component perpendicular to the boundary.
    Input: on_boundary: Array of shape [D], output of function check_position_on_boundary
           momentum: Array of shape [D], momentum associated with position
    Output: momentum: Array of shape [D], modified momentum array
    """
    assert on_boundary.shape == momentum.shape

    momentum = jnp.where(on_boundary * momentum > 0, -momentum, momentum)

    return momentum

def reflection_necessary(
        position: chex.Array, 
        momentum: chex.Array, 
        kinetic_energy_grad_fn: EuclideanKineticEnergyGrad,
        step_size: float, 
        t: float,
        a2: Optional[float] = 1.
        ):
    """Check for the existence of intersection points of next step with boundary of D-dimensional unit hypercube.
    Input: position: Array of shape [D], current position vector
           momentum: Array of shape [D], momentum at current position
           kinetic_energy_grad_fn: Callable that calculates the gradient of the EuclideanKineticEnergy function
           step_size: float, step size of next step
           t: float, amount of step_size that was already moved in previous reflection, t <= step_size
           a2: Optional[float], scales position update in velocity verlet, standard value a2 = 1.
    Returns: bool: True if next step starting at q along momentum vector p will intersect boundary of unit hypercube
                   False otherwise
    """
    on_boundary = check_position_on_boundary(position)
    momentum = revert_momentum_for_boundary_positions(on_boundary, momentum)

    # Go step
    kinetic_grad = kinetic_energy_grad_fn(momentum)
    position = jax.tree_util.tree_map(
        lambda position, kinetic_grad: position + a2 * (step_size - t) * kinetic_grad,
        position,
        kinetic_grad,
    )
    
    # Determine indices of boundaries that were crossed
    outside_of_boundary = check_outside_of_boundary(position)
    
    # Determine whether boundary was crossed and reflection is necessary
    refl_necessary = jnp.where(jnp.sum(jnp.abs(outside_of_boundary))>0, True, False)
    
    return refl_necessary
    

def find_next_intersection(
        position: chex.Array, 
        momentum: chex.Array, 
        kinetic_energy_grad_fn: EuclideanKineticEnergyGrad,
        step_size: float, 
        t: float,
        a2: Optional[float] = 1.
        ) -> Union[chex.Array, chex.Array, float, int]:
    """Finds intersection point of next step with boundary of D-dimensional unit hypercube.
    Input: position: Array of shape [D], current position vector
           momentum: Array of shape [D], momentum at position
           kinetic_energy_grad_fn: Callable that calculates the gradient of the EuclideanKineticEnergy function
           step_size: float, step size of next step
           t: float, amount of step_size that was already moved in previous reflection, t <= step_size
           a2: Optional[float], scales position update in velocity verlet, standard value a2 = 1.
    Returns: x: Array of shape [D] intersection point with boundary
             p: Array of shape [D] momentum at q (include output for consistency)
             t_x: float, remaining lenth of momentum vector after hitting boundary, t_x <= t <= step_size
             ind_refl_boundary: int, index of reflection boundary
                                value indicates axis that is perpendicular to the reflection surface
    """
    dim = len(position)
    on_boundary = check_position_on_boundary(position)
    momentum = revert_momentum_for_boundary_positions(on_boundary, momentum)

    # Go step
    kinetic_grad = kinetic_energy_grad_fn(momentum)
    position_new = jax.tree_util.tree_map(
        lambda position, kinetic_grad: position + a2 * (step_size - t) * kinetic_grad,
        position,
        kinetic_grad,
    )

    # Determine indices of boundaries that were crossed
    outside_of_boundary = check_outside_of_boundary(position_new)
    
    # Calculate distances to boundary intersection points
    # Construct hyperplanes
    hyper_plane = jnp.identity(dim)
    hyper_planes = jnp.array([jnp.delete(hyper_plane, i, axis=1) for i in range(dim)])
    def body_fn(i, dist_to_intersections):
        ind = outside_of_boundary[i]
        hyper_plane = hyper_planes[i]
        plane_offset = jnp.where(ind>0, 1., 0.)
        # Construct elements of linear equation
        y = position - plane_offset
        A = jnp.concatenate([jnp.expand_dims(position - position_new, 1), hyper_plane], axis=1)
        # Solve linear equation y = Ax
        params = jnp.linalg.solve(a=A, b=y)
        mask_params = jnp.where(ind!=0, params[0], 0)
        dist_to_intersections = dist_to_intersections.at[i].set(mask_params)

        return dist_to_intersections

    dist_to_intersections = jax.lax.fori_loop(0, dim, body_fn, jnp.zeros(outside_of_boundary.shape))
    
    # Find first intersection
    mask_outside_boundary = jnp.where(outside_of_boundary!=0, True, False)
    # If position_new is not outside the boundary, position is located on boundary
    # -> lambda_min = remaining step
    reflection_remaining = jnp.sum(mask_outside_boundary, dtype=bool)
    # If there are no reflections remaining, we have to modify the mask, because the jnp.min will throw an error otherwise.
    mask_outside_boundary = jnp.where(reflection_remaining, mask_outside_boundary, True)
    masked_dists = jnp.where(mask_outside_boundary, dist_to_intersections, jnp.inf)
    lambda_min = jnp.where(reflection_remaining, jnp.min(masked_dists), 1.)

    ind_min = jnp.where(dist_to_intersections==lambda_min, 1., 0.)
    ind_refl_boundary = jnp.where(reflection_remaining, jnp.argmax(ind_min), jnp.argmax(on_boundary))

    # Update position and distance t
    x = jax.tree_util.tree_map(
        lambda position, position_new: position + a2 * step_size * lambda_min * (position_new - position),
        position,
        position_new,
    )
    t_x = jax.tree_util.tree_map(
        lambda position, x, momentum: jnp.linalg.norm(x - position)/(step_size * jnp.linalg.norm(momentum)),
        position,
        x,
        momentum
    )

    return x, momentum, t_x, ind_refl_boundary