import jax.numpy as jnp

def reflection_necessary(q, p, step_size, t):
    """Check for the existence of intersection points of next step with boundary of D-dimensional unit hypercube.
    Input: q: [D], current position vector
           p: [D], momentum at q
           step_size: float, step size of next step
           t: float, amount of step_size that was already moved in previous reflection, t <= step_size
    Returns: bool: True if next step starting at q along momentum vector p will intersect boundary of unit hypercube
                   False otherwise
    """
    # Check for q on the boundary
    ind_on_boundary_0 = -(jnp.where(q==0)[0]+1)
    ind_on_boundary_1 = jnp.where(q==1)[0]+1
    inds_on_boundary = jnp.concatenate([ind_on_boundary_0, ind_on_boundary_1], axis=0)
    boundary = False
    if inds_on_boundary.shape != (0,):
        boundary = True
        for ind in inds_on_boundary:
            # if p component points outside of hypercube: revert it
            if ind * p[jnp.abs(ind)-1] > 0:
                #revert p component perpendicular to bondary
                ind = jnp.abs(ind)-1
                p_perp = jnp.zeros_like(p).at[ind].set(p[ind])
                p_parallel = p.at[ind].set(0)
                p_perp *= (-1)
                p = p_parallel + p_perp
    # Go step
    q_new = q + (step_size - t)*p
    # Determine boundaries that were crossed
    ind_larger_1 = jnp.where(q_new>1)[0]+1
    ind_smaller_0 = -(jnp.where(q_new<0)[0]+1)
    inds = jnp.concatenate([ind_smaller_0, ind_larger_1], axis=0)
    if boundary:
        # Remove boundary indices where q is located at the moment
        i_keep = jnp.where(inds!=inds_on_boundary)
        inds = inds[i_keep]
    if inds.shape == (0,):
        return False
    else:
        return True
    

def find_next_intersection(q, p, step_size, t):
    """Finds intersection point of next step with boundary of D-dimensional unit hypercube.
    Input: q: [D], current position vector
           p: [D], momentum at q
           step_size: float, step size of next step
           t: float, amount of step_size that was already moved in previous reflection, t <= step_size
    Returns: x: [D] intersection point with boundary
             p: [D] momentum at q (include output for consistency)
             t_x: float, remaining lenth of momentum vector after hitting boundary, t_x <= t <= step_size
             ind_refl_boundary: int, index of reflection boundary
                                value indicates axis that is perpendicular to the reflection surface
    """
    dim = len(q)
    # If q is already on the boundary, we need to reverse the momentum perpendicular to the boundary first
    ind_on_boundary_0 = -(jnp.where(q==0)[0]+1)
    ind_on_boundary_1 = jnp.where(q==1)[0]+1
    inds_on_boundary = jnp.concatenate([ind_on_boundary_0, ind_on_boundary_1], axis=0)
    boundary = False
    if inds_on_boundary.shape != (0,):
        boundary = True
        for ind in inds_on_boundary:
            # If p component points outside of hypercube: revert it
            if ind * p[jnp.abs(ind)-1] > 0:
                #Revert p component perpendicular to bondary
                ind = jnp.abs(ind)-1
                p_perp = jnp.zeros_like(p).at[ind].set(p[ind])
                p_parallel = p.at[ind].set(0)
                p_perp *= (-1)
                p = p_parallel + p_perp
    # Go step
    q_new = q + (step_size - t)*p
    # Determine boundaries that were crossed
    ind_larger_1 = jnp.where(q_new>1)[0]+1
    ind_smaller_0 = -(jnp.where(q_new<0)[0]+1)
    inds = jnp.concatenate([ind_smaller_0, ind_larger_1], axis=0)
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
        y = q - plane_offset
        A = jnp.concatenate([jnp.expand_dims(q - q_new, 1), hyper_plane], axis=1)
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
    x = q + (q_new - q)*lambda_min * step_size
    t_x = jnp.linalg.norm(x-q)/(step_size * jnp.linalg.norm(p))
    ind_refl_boundary = jnp.abs(inds[ind_min])-1

    return x, p, t_x, ind_refl_boundary