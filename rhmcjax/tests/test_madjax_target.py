import jax.numpy as jnp

from rhmcjax.targets.target_util import load_target

def test_madjax_target():
    path_to_madjax_matrix_element = "rhmcjax.targets.madjax_ee_to_ttbar_wb"
    dim = 8
    name = "ee_to_ttbar_wb"
    center_of_mass_energy = 1000 # [GeV]
    model_parameters = {}
    epsilon = 1e-5

    target_load = load_target(name, path_to_madjax_matrix_element, dim, center_of_mass_energy, model_parameters, epsilon)
    sample = jnp.array([0.5]*dim)

    assert jnp.isfinite(target_load.log_prob(sample))
