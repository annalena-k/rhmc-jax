import chex
import jax.numpy as jnp

def check_restriction_to_unit_hypercube(samples: chex.Array) -> chex.Array:
    """Checks whether samples are inside or outside of unit hypercube boundary.
    Returns mask containing True if sample is within and False if sample is outside of boundary.
    Input:
        samples: jax.array of shape [N, dim]
    Output:
        mask: jax.array of type bool and shape [N, dim]
    """
    mask_larger1 = jnp.prod(jnp.where(samples>1., False, True), axis=1, dtype=bool)
    mask_smaller0 = jnp.prod(jnp.where(samples<0., False, True), axis=1, dtype=bool)
    mask = mask_larger1 * mask_smaller0
    assert mask.shape == samples[:,0].shape

    return mask