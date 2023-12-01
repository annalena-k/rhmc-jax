import jax
import jax.numpy as jnp
import pytest

import blackjax.mcmc.metrics as metrics
from rhmcjax.rhmc.intersection_with_boundary import *

#@pytest.mark.skip(reason="no way of currently testing this")
def test_check_outside_of_boundary():
    position = jnp.array([0., 1.5, 1.1, 0.2, 0.4, -1.4])
    outside_of_boundary = check_outside_of_boundary(position)
    assert jnp.all(outside_of_boundary == jnp.array([0., 1., 1., 0., 0., -1]))

def test_jit_check_outside_of_boundary():
    position = jnp.array([0., 1.5, 1.1, 0.2, 0.4, -1.4])
    get_inds_jit = jax.jit(check_outside_of_boundary)
    outside_of_boundary = get_inds_jit(position)
    assert jnp.all(outside_of_boundary == jnp.array([0., 1., 1., 0., 0., -1]))

def test_check_position_on_boundary():
    position = jnp.array([0., 1.5, 1., 0.2, 0.4, -1.4])
    inds_on_boundary = check_position_on_boundary(position)
    assert jnp.all(inds_on_boundary == jnp.array([-1., 0., 1., 0., 0., 0.]))

def test_jit_check_position_on_boundary():
    position = jnp.array([0., 1.5, 1., 0.2, 0.4, -1.4])
    get_inds_jit = jax.jit(check_position_on_boundary)
    inds_on_boundary = get_inds_jit(position)
    assert jnp.all(inds_on_boundary == jnp.array([-1., 0., 1., 0., 0., 0.]))

def test_momentum_for_boundary_positions():
    position = jnp.array([0., 1., 1., 0.2, 0., 1.])
    momentum = jnp.array([-0.2, 0.2, -0.2, 0.2, 0.2, 0.1])
    on_boundary = check_position_on_boundary(position)
    new_momentum = revert_momentum_for_boundary_positions(on_boundary, momentum)
    assert jnp.all(new_momentum ==  jnp.array([0.2, -0.2, -0.2, 0.2, 0.2, -0.1]))

def test_jit_momentum_for_boundary_positions():
    position = jnp.array([0., 1., 1., 0.2, 0., 1.])
    momentum = jnp.array([-0.2, 0.2, -0.2, 0.2, 0.2, 0.1])
    on_boundary = check_position_on_boundary(position)
    jit_rev_mom = jax.jit(revert_momentum_for_boundary_positions)
    new_momentum = jit_rev_mom(on_boundary, momentum)
    assert jnp.all(new_momentum ==  jnp.array([0.2, -0.2, -0.2, 0.2, 0.2, -0.1]))

def test_reflection_necessary():
    inverse_mass_matrix = jnp.array([1., 1.])
    _, kinetic_energy_fn, _ = metrics.gaussian_euclidean(
                inverse_mass_matrix
            )
    kinetic_energy_grad_fn = jax.grad(kinetic_energy_fn)
    step_size = 1.
    t_simple = 0.
    q_center = jnp.array([0.5, 0.5])
    p_along_x = jnp.array([1., 0.])
    assert reflection_necessary(q_center, p_along_x, kinetic_energy_grad_fn, step_size, t_simple) == True
    q_boundary = jnp.array([1., 0.5])
    assert reflection_necessary(q_boundary, p_along_x, kinetic_energy_grad_fn, step_size, t_simple) == True
    q_no_b = jnp.array([0.8, 0.5])
    p_along_negx = jnp.array([-0.8, 0.])
    assert reflection_necessary(q_no_b, p_along_negx, kinetic_energy_grad_fn, step_size, t_simple) == False

def test_jit_reflection_necessary():
    inverse_mass_matrix = jnp.array([1., 1.])
    _, kinetic_energy_fn, _ = metrics.gaussian_euclidean(
                inverse_mass_matrix
            )
    kinetic_energy_grad_fn = jax.grad(kinetic_energy_fn)
    step_size = 1.
    t_simple = 0.
    q_center = jnp.array([0.5, 0.5])
    p_along_x = jnp.array([1., 0.])
    def reflection_static_grad_fn(q, p, step_size, t):
        return reflection_necessary(q, p, kinetic_energy_grad_fn, step_size, t)
    refl_necessary_jit = jax.jit(reflection_static_grad_fn)
    refl_necessary = refl_necessary_jit(q_center, p_along_x, step_size, t_simple)
    assert refl_necessary == True

def test_reflection_2d():
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
    print(q_boundary)
    assert jnp.all(q_boundary == jnp.array([1., 0.5]))
    assert jnp.all(p_remaining == p_along_x)
    assert t_refl == 0.5
    assert ind_refl_boundary == 0
    p_along_y = jnp.array([0., 0.2])
    assert reflection_necessary(q_center, p_along_y, kinetic_energy_grad_fn, step_size, t_simple) == False

def test_reflection_2d_boundary():
    q_at_boundary = jnp.array([0.1, 1.])
    p_along_y = jnp.array([0., 0.2])
    inverse_mass_matrix = jnp.array([1., 1.])
    _, kinetic_energy_fn, _ = metrics.gaussian_euclidean(
                inverse_mass_matrix
            )
    kinetic_energy_grad_fn = jax.grad(kinetic_energy_fn)
    step_size = 1.
    t_simple = 0.
    q_boundary, p_remaining, t_refl, ind_refl_boundary = find_next_intersection(q_at_boundary, p_along_y, kinetic_energy_grad_fn, step_size, t_simple)
    print('out')
    assert jnp.all(q_boundary == jnp.array([0.1, 0.8]))
    assert jnp.all(p_remaining == jnp.array([0., -0.2]))
    assert jnp.allclose(t_refl, 1.)
    assert ind_refl_boundary == 1

def test_reflection_4d():
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

def test_multiple_reflections():
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

def test_jit_multiple_reflections():
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

    def run_reflections(q, p):
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
        full_length += jnp.linalg.norm(q - q_final)

        return q_final, full_length
    
    run_reflections_jit = jax.jit(run_reflections)
    q_final, full_length = run_reflections_jit(q, p)
    assert jnp.all(q_final <= 1.)
    assert jnp.all(q_final >= 0.)
    
    assert full_length == final_length

#if __name__ == "__main__":
#    test_reflection_2d()