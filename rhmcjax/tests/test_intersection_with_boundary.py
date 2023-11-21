import jax
import jax.numpy as jnp

import blackjax.mcmc.metrics as metrics
from rhmcjax.rhmc.intersection_with_boundary import reflection_necessary, find_next_intersection


# Simple 2D Tests
q_center = jnp.array([0.5, 0.5])
p_along_x = jnp.array([1., 0.])
inverse_mass_matrix = jnp.array([1., 1.])
_, kinetic_energy_fn, _ = metrics.gaussian_euclidean(
            inverse_mass_matrix
        )
kinetic_energy_grad_fn = jax.grad(kinetic_energy_fn)
step_size = 1.
t_simple = 0.
assert reflection_necessary(q_center, p_along_x, kinetic_energy_grad_fn, step_size, t_simple) == True
q_boundary, p_remaining, t_refl, ind_refl_boundary = find_next_intersection(q_center, p_along_x, kinetic_energy_grad_fn, step_size, t_simple)
assert jnp.all(q_boundary == jnp.array([1., 0.5]))
assert jnp.all(p_remaining == p_along_x)
assert t_refl == 0.5
assert ind_refl_boundary == 0
p_along_y = jnp.array([0., 0.2])
assert reflection_necessary(q_center, p_along_y, kinetic_energy_grad_fn, step_size, t_simple) == False
print('Simple 2D tests successful.')

# Simple 4D Tests.
q_center = jnp.array([0.5, 0.5, 0.5, 0.5])
p_along_x3 = jnp.array([0., 0., -1., 0.])
inverse_mass_matrix = jnp.array([1., 1., 1., 1.])
_, kinetic_energy_fn, _ = metrics.gaussian_euclidean(
            inverse_mass_matrix
        )
kinetic_energy_grad_fn = jax.grad(kinetic_energy_fn)
step_size = 1.
t_simple = 0.
assert reflection_necessary(q_center, p_along_x3, kinetic_energy_grad_fn, step_size, t_simple) == True
q_boundary, p_remaining, t_refl, ind_refl_boundary = find_next_intersection(q_center, p_along_x3, kinetic_energy_grad_fn, step_size, t_simple)
assert jnp.all(q_boundary == jnp.array([0.5, 0.5, 0., 0.5]))
assert jnp.all(p_remaining == p_along_x3)
assert t_refl == 0.5
assert ind_refl_boundary == 2
p_along_y3 = jnp.array([0., 0.2, 0., 0.])
assert reflection_necessary(q_center, p_along_y3, kinetic_energy_grad_fn, step_size, t_simple) == False
print('Simple 4D tests successful.')

# Multiple-Reflection Test.

# half-step evoluton of momentum
q = jnp.array([0.5, 0.5])
p = jnp.array([2.0, 3.0])
inverse_mass_matrix = jnp.array([1., 1.])
_, kinetic_energy_fn, _ = metrics.gaussian_euclidean(
            inverse_mass_matrix
        )
kinetic_energy_grad_fn = jax.grad(kinetic_energy_fn)
final_length = jnp.linalg.norm(p)
step_size = 1.
t = 0
full_length = 0

while reflection_necessary(q, p, kinetic_energy_grad_fn, step_size, t):
    x, p_update, t_x, ind_boundary = find_next_intersection(q, p, kinetic_energy_grad_fn, step_size, t)
    full_length += jnp.linalg.norm(x - q)
    # Update values
    q = x
    p = p_update
    t += t_x
    # Decompose p into vector parallel / perpendicular to boundary
    p_perp = jnp.zeros_like(p).at[ind_boundary].set(p[ind_boundary])
    p_parallel = p.at[ind_boundary].set(0)
    p_perp *= (-1)
    p = p_parallel + p_perp

# update final position (after all reflections)
q_final = q + (step_size - t)*p
assert jnp.all(q_final <= 1.)
assert jnp.all(q_final >= 0.)
full_length += jnp.linalg.norm(q - q_final)
assert full_length == final_length
print('Test for multiple reflections successful.')