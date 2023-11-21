from typing import Optional, Union
import chex
import jax
import jax.numpy as jnp

from blackjax.mcmc.integrators import EuclideanKineticEnergy

EuclideanKineticEnergyGrad = EuclideanKineticEnergy

def get_indices_of_crossed_boundaries(
        position: chex.Array
    ) -> chex.Array:
    """ Get indices of D-dimensional unit hypercube boundaries where position is located outside.
        The indices with negative sign correspond to the lower boundary (at 0), while the positive indices correspond to the upper boundary (at 1).
        The absolute value of the index indicates the dimension d \in [1,...,D].
    Input: position: Array of shape [D]
    Output: indices: Array of variable shape
    """
    # TODO: Adjust for fixed shapes of index array, currently index arrays have variable length -> not possible to jit
    ind_larger_1 = jnp.where(position>1)[0]+1
    ind_smaller_0 = -(jnp.where(position<0)[0]+1)
    indices = jnp.concatenate([ind_smaller_0, ind_larger_1], axis=0)

    return indices

def check_for_position_on_boundary(
        position: chex.Array, 
        momentum: chex.Array
        ) -> Union[bool, chex.Array]:
    """ Check whether positon is located directly on the boundary of the D-dimensional unit hypercube.
    If this is the case, the momentum component perpendicular to this boundary has to be reversed.
    Input: position: Array of shape [D], current position vector
           momentum: Array of shape [D], momentum at current position
    Output: on_boundary: bool, True if position is located on boundary
            inds_on_boundary: Array of variable shape [B], depending on how many boundaries the positon is located
            momentum: Array of shape [D], momentum with reversed component that is perpendicular to the boundary 
    """
    # TODO: Adjust for arbitrary PyTrees, currently only working for Arrays
    # TODO: Adjust for fixed shapes of index arrays, currently index arrays have variable length -> not possible to jit
    ind_on_boundary_lower = -(jnp.where(position==0)[0]+1)
    ind_on_boundary_upper = jnp.where(position==1)[0]+1
    inds_on_boundary = jnp.concatenate([ind_on_boundary_lower, ind_on_boundary_upper], axis=0)
    on_boundary = False
    if inds_on_boundary.shape != (0,):
        on_boundary = True
        for ind in inds_on_boundary:
            # If momentum component points outside of hypercube: revert it
            if ind * momentum[jnp.abs(ind)-1] > 0:
                # Revert momentum component perpendicular to boundary
                ind = jnp.abs(ind)-1
                p_perp = jnp.zeros_like(momentum).at[ind].set(momentum[ind])
                p_parallel = momentum.at[ind].set(0)
                p_perp *= (-1)
                momentum = p_parallel + p_perp

    return on_boundary, inds_on_boundary, momentum


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
    boundary, inds_on_boundary, momentum = check_for_position_on_boundary(position, momentum)

    # Go step
    kinetic_grad = kinetic_energy_grad_fn(momentum)
    position = jax.tree_util.tree_map(
        lambda position, kinetic_grad: position + a2 * (step_size - t) * kinetic_grad,
        position,
        kinetic_grad,
    )
    
    # Determine indices of boundaries that were crossed
    inds = get_indices_of_crossed_boundaries(position)
    # Check if crossed boundaries are different from current boundary position
    if boundary:
        # Remove boundary indices of current position
        i_keep = jnp.where(inds!=inds_on_boundary)
        inds = inds[i_keep]
    
    # Determine whether boundary was crossed and reflection is necessary
    if inds.shape == (0,):
        refl_necessary = False
    else:
        refl_necessary = True
    
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
    boundary, inds_on_boundary, momentum = check_for_position_on_boundary(position, momentum)

    # Go step
    kinetic_grad = kinetic_energy_grad_fn(momentum)
    position_new = jax.tree_util.tree_map(
        lambda position, kinetic_grad: position + a2 * (step_size - t) * kinetic_grad,
        position,
        kinetic_grad,
    )

    # Determine indices of boundaries that were crossed
    inds = get_indices_of_crossed_boundaries(position_new)
    if boundary:
        # Remove boundaries where q is located at the moment
        i_keep = jnp.where(inds!=inds_on_boundary)
        inds = inds[i_keep]
    
    # Calculate distances to boundary intersection points
    dist_to_intersections = jnp.zeros(inds.shape)
    for i, ind in enumerate(inds):
        # Construct hyperplane
        hyper_plane = jnp.identity(dim)
        hyper_plane = jnp.delete(hyper_plane, jnp.abs(ind)-1, axis=1)
        if ind < 0:
            plane_offset = jnp.zeros([dim])
        else:
            plane_offset = jnp.zeros([dim])
            plane_offset = plane_offset.at[jnp.abs(ind)-1].set(1)
        # Construct elements of linear equation
        y = position - plane_offset
        A = jnp.concatenate([jnp.expand_dims(position - position_new, 1), hyper_plane], axis=1)
        # Solve linear equation y = Ax
        params = jnp.linalg.solve(a=A, b=y)
        dist_to_intersections = dist_to_intersections.at[i].set(params[0])
    # Remove distances of 0
    if boundary:
        inds_non0 = jnp.nonzero(dist_to_intersections)
        dist_to_intersections = dist_to_intersections[inds_non0]
    
    # Find first intersection
    lambda_min = jnp.min(dist_to_intersections)
    ind_min = jnp.where(dist_to_intersections==lambda_min)[0][0]
    x = jax.tree_util.tree_map(
        lambda position, position_new: position + a2 * step_size * lambda_min * (position_new - position),
        position,
        position_new,
    )
    t_x = jax.tree_util.tree_map(
        lambda position, x, momentum: jnp.linalg.norm(x - position)/(step_size * jnp.linalg.norm(momentum)),
        position,
        x, # Do I need to include x here?
        momentum
    )
    ind_refl_boundary = jnp.abs(inds[ind_min])-1

    return x, momentum, t_x, ind_refl_boundary