from typing import Tuple
import os
import pickle
import sympy as sp
import chex
import jax
from jax import numpy as jnp
from jax import random as random

from tensorwaves.function.sympy import create_function

from rhmcjax.targets.target_util import Target

Arr = chex.Array
Sample = chex.Array
Prob = chex.Array
Det = chex.Array


class Lc2pKpi(Target):
    """ The 2D matrix element for the 3-body decay $\Lambda_c^+ \rightarrow pK^-\pi^+$ was first introduced in
    [this publication](https://doi.org/10.1007/JHEP07(2023)228) and the [associated code](https://doi.org/10.5281/zenodo.7544989)
    was used to save the model to the file `Lc2pKpi_model.pkl`. More information can be found on the
    [ComPWA website](https://compwa.github.io/polarimetry/index.html).
    
    The class `Lc2pKpi` includes a wrapper to load the matrix element code as well as methods which transform the distribution 
    from the Dalitz plot to the square Dalitz plot and finally to the unit square.
    """
    def __init__(self,
                path_to_ComPWA_matrix_element: str,
                dim: int = 2):
        self.path_ComPWA = path_to_ComPWA_matrix_element
        self.dim = dim
        self.name = "Lc2pKpi"
        self.load_matrix_element()
        
    def load_matrix_element(self):
        # Load pickled matrix element
        def load_file():
            if not os.path.exists(self.path_ComPWA):
                raise ValueError(f"The input file was not found at ${self.path_ComPWA}")
            with open(self.path_ComPWA, "rb") as f:
                return pickle.load(f)
        model_description = load_file()

        # Extract values for masses
        key_m0, key_m1, key_m2, key_m3 = sp.symbols("m:4", nonnegative=True)
        self.m0 = model_description["parameter_defaults"][key_m0]
        self.m1 = model_description["parameter_defaults"][key_m1]
        self.m2 = model_description["parameter_defaults"][key_m2]
        self.m3 = model_description["parameter_defaults"][key_m3]

        # Re-define density in terms of invariant masses s12, s23
        def fully_substitute(model_description):
            expr = (
                model_description["intensity_expr"]
                .xreplace(model_description["variables"])
                .xreplace(model_description["sigma3"])
                .xreplace(model_description["parameter_defaults"])
            )
            return expr
        intensity_on_2vars = fully_substitute(model_description)
        s12 = sp.symbols("sigma1:3", nonnegative=True)
        assert intensity_on_2vars.free_symbols == set(s12)
        self.density = create_function(intensity_on_2vars, backend="jax")

    def get_density(self):
        return self.density
    
    @staticmethod
    def kallen(x, y, z):
        return x**2 + y**2 + z**2 - 2*x*y - 2*y*z - 2*z*x

    def sqrtkallen(self, x, y, z):
        kall = self.kallen(x, y, z)
        return jnp.sqrt(kall)

    def kibble(self, s1: Arr, s2: Arr, s3: Arr) -> Arr:
        a = self.kallen(self.m0 ** 2, s1, self.m3 ** 2)
        b = self.kallen(self.m0 ** 2, s2, self.m1 ** 2)
        c = self.kallen(self.m0 ** 2, s3, self.m2 ** 2)
        return self.kallen(a, b, c)

    def is_physical(self, s23: Arr, s31: Arr) -> Arr:
        MASSES = jnp.array([self.m0, self.m1, self.m2, self.m3])
        s12 = jnp.sum(MASSES ** 2) - s23 - s31
        physical_region = self.kibble(s12, s23, s31)
        physical_region = jnp.where(physical_region <= 0, 1., 0.)
        return physical_region

    def is_physical_nan(self, s23: Arr, s31: Arr) -> Arr:
        MASSES = jnp.array([self.m0, self.m1, self.m2, self.m3])
        s12 = jnp.sum(MASSES ** 2) - s23 - s31
        physical_region = self.kibble(s12, s23, s31)
        physical_region = jnp.where(physical_region <= 0, 1., jnp.nan)
        return physical_region

    def get_decay_rate(self, samples: Sample) -> Prob:
        if len(samples.shape) == 1:
            samples = jnp.expand_dims(samples, axis=0)
        Is_final = self.density({"sigma1": samples[:, 0], "sigma2": samples[:, 1]})

        return Is_final.squeeze()

    def get_cos_theta_from_s23(self, samples: Sample) -> Tuple[Sample, Det]:
        if len(samples.shape) == 1:
            samples = jnp.expand_dims(samples, axis=0)
        s23, s31 = samples[:, 0], samples[:, 1]
        physical_region = self.is_physical(s23, s31)

        def calc_cos_theta_from_s23(s23, s31):
            nom = 2 * s31 * (self.m2 ** 2 + self.m3 ** 2 - s23) + (self.m0 ** 2 - s31 - self.m2 ** 2) \
                * (s31 + self.m3 ** 2 - self.m1 ** 2)
            denom = self.sqrtkallen(self.m0 ** 2, s31, self.m2 ** 2) * self.sqrtkallen(s31, self.m3 ** 2, self.m1 ** 2)
            cos_theta = nom * jnp.where(denom == 0, 0, 1 / denom)
            return cos_theta

        cos_theta, dcds = jax.vmap(jax.value_and_grad(calc_cos_theta_from_s23, argnums=0))(s23, s31)
        samples_out = jnp.dstack([cos_theta * physical_region, s31])
        samples_out = samples_out.reshape([-1, 2])
        if len(samples_out.shape) == 1:
            samples_out = jnp.expand_dims(samples_out, axis=0)

        return samples_out, dcds

    def get_s23_from_cos_theta(self, samples: Sample) -> Tuple[Sample, Det]:
        if len(samples.shape) == 1:
            samples = jnp.expand_dims(samples, axis=0)
        c, s31 = samples[:, 0], samples[:, 1]

        def calc_s23_from_cos_theta(c, s31):
            mask = jnp.where(s31 == 0, 0, 1)
            s23 = self.m2 ** 2 + self.m3 ** 2 + (self.m0 ** 2 - s31 - self.m2 ** 2) \
                * (s31 + self.m3 ** 2 - self.m1 ** 2) \
                * jnp.where(s31 == 0, 0, 1 / (2 * s31)) \
                - self.sqrtkallen(self.m0 ** 2, s31, self.m2 ** 2) \
                * self.sqrtkallen(s31, self.m3 ** 2, self.m1 ** 2) \
                * jnp.where(s31 == 0, 0, 1 / (2 * s31)) * c
            return s23 * mask

        s23, dsdc = jax.vmap(jax.value_and_grad(calc_s23_from_cos_theta, argnums=0))(c, s31)
        samples_out = jnp.dstack([s23, s31])
        samples_out = samples_out.reshape([-1, 2])
        if len(samples_out.shape) == 1:
            samples_out = jnp.expand_dims(samples_out, axis=0)

        return samples_out, dsdc

    def get_decay_rate_square(self, samples: Sample) -> Prob:
        s23vs, dsdc = self.get_s23_from_cos_theta(samples)
        Is = self.density({"sigma1": s23vs, "sigma2": samples[:, 1]})
        Is_final = Is * jnp.abs(dsdc)

        return Is_final

    def scale_samples_to_unit_interval(self, samples: Sample) -> Tuple[Sample, Det]:
        def scale_min_max(var, v_min, v_max):
            var_scaled = (var - v_min) * jnp.where((v_max - v_min) == 0, 0, v_max - v_min)
            return var_scaled

        if len(samples.shape) == 1:
            samples = jnp.expand_dims(samples, axis=0)

        cos_min, cos_max = jnp.full(samples[:, 0].shape, -1.), jnp.full(samples[:, 0].shape, 1.)
        s31_min_val = jnp.min(jnp.array([(self.m3 + self.m1) ** 2, (self.m0 - self.m2) ** 2]))
        s31_max_val = jnp.max(jnp.array([(self.m3 + self.m1) ** 2, (self.m0 - self.m2) ** 2]))
        s31_min = jnp.full(samples[:, 1].shape, s31_min_val)
        s31_max = jnp.full(samples[:, 1].shape, s31_max_val)

        x1_trafo, dxdcos = jax.vmap(jax.value_and_grad(scale_min_max, argnums=0))(samples[:, 0], cos_min, cos_max)
        x2_trafo, dyds31 = jax.vmap(jax.value_and_grad(scale_min_max, argnums=0))(samples[:, 1], s31_min, s31_max)
        samples_trafo = jnp.dstack([x1_trafo, x2_trafo]).squeeze()
        dets = jnp.dstack([dxdcos, dyds31]).squeeze()

        if len(samples_trafo.shape) == 1:
            samples_trafo = jnp.expand_dims(samples_trafo, axis=0)
        if len(dets.shape) == 1:
            dets = jnp.expand_dims(dets, axis=0)

        return samples_trafo, dets

    def scale_samples_to_original_interval(self, samples: Sample) -> Tuple[Sample, Det]:
        def inverse_scale_min_max(var, v_min, v_max):
            var_scaled = (v_max - v_min) * var + v_min
            return var_scaled

        if len(samples.shape) == 1:
            samples = jnp.expand_dims(samples, axis=0)

        cos_min, cos_max = jnp.full(samples[:, 0].shape, -1.), jnp.full(samples[:, 0].shape, 1.)
        s31_min_val = jnp.min(jnp.array([(self.m3 + self.m1) ** 2, (self.m0 - self.m2) ** 2]))
        s31_max_val = jnp.max(jnp.array([(self.m3 + self.m1) ** 2, (self.m0 - self.m2) ** 2]))
        s31_min = jnp.full(samples[:, 1].shape, s31_min_val)
        s31_max = jnp.full(samples[:, 1].shape, s31_max_val)

        cos_t_trafo, dcosdx = jax.vmap(jax.value_and_grad(inverse_scale_min_max, argnums=0))(samples[:, 0], cos_min,
                                                                                            cos_max)
        s31_trafo, ds31dy = jax.vmap(jax.value_and_grad(inverse_scale_min_max, argnums=0))(samples[:, 1], s31_min, s31_max)
        samples_trafo = jnp.dstack([cos_t_trafo, s31_trafo]).squeeze()
        dets = jnp.dstack([dcosdx, ds31dy]).squeeze()

        if len(samples_trafo.shape) == 1:
            samples_trafo = jnp.expand_dims(samples_trafo, axis=0)
        if len(dets.shape) == 1:
            dets = jnp.expand_dims(dets, axis=0)

        return samples_trafo, dets

    def evaluate_DP_on_unit_interval(self, samples: Sample) -> Prob:
        """Evaluate samples from flow on original Dalitz plot and transform density back to unit interval
        to compare to log q of flow."""
        samples_square_dp, detJ = self.scale_samples_to_original_interval(samples)
        dcosdx, dsdy = detJ[:, 0], detJ[:, 1]
        samples_dp, dsdcos = self.get_s23_from_cos_theta(samples_square_dp)
        I_dp = self.get_decay_rate(samples_dp)
        I_xy = I_dp * jnp.abs(dsdcos) * jnp.abs(dcosdx) * jnp.abs(dsdy)

        return I_xy.squeeze()

    def evaluate_DP_on_physical_space(self, samples: Sample) -> tuple[Sample, Prob]:
        """Evaluate samples from flow on Dalitz plot and return transformed coordinates and DP density."""
        samples_square_dp, _ = self.scale_samples_to_original_interval(samples)
        samples_dp, _ = self.get_s23_from_cos_theta(samples_square_dp)
        I_dp = self.get_decay_rate(samples_dp)

        return samples_dp.squeeze(), I_dp.squeeze()

    def transform_q_to_physical_space(self, samples: Sample, q: Prob) -> Tuple[Sample, Prob]:
        """Transform flow samples and corresponding q values to Dalitz plot space."""
        samples_square_dp, detJ = self.scale_samples_to_original_interval(samples)
        dcosdx, dsdy = detJ[:, 0], detJ[:, 1]
        samples_dp, dsdcos = self.get_s23_from_cos_theta(samples_square_dp)
        # Clip derivative to avoid divergent values in plotting of q (except 0)
        dsdcos = jnp.where(dsdcos == 0, 0, jnp.clip(dsdcos, a_min=0.1))
        q_dp = q * jnp.where(dsdcos == 0, 0, 1 / jnp.abs(dsdcos)) \
            * jnp.where(dcosdx == 0, 0, 1 / jnp.abs(dcosdx)) \
            * jnp.where(dsdy == 0, 0, 1 / jnp.abs(dsdy))

        return samples_dp.squeeze(), q_dp.squeeze()
    
    def transform_samples_to_physical_space(self, samples: Sample) -> Sample:
        """Transform flow samples to Dalitz plot space."""
        samples_square_dp, _ = self.scale_samples_to_original_interval(samples)
        samples_dp, _ = self.get_s23_from_cos_theta(samples_square_dp)

        return samples_dp.squeeze()
    
    def log_prob(self, samples: Sample) -> Prob:
        """Evaluate logprob of samples from unit interval."""
        log_probs = jnp.log(self.evaluate_DP_on_unit_interval(samples))
        return log_probs
    
    def prob_no_nan(self, samples: Sample) -> Prob:
        """Evaluate prob of samples from unit interval and replace NaN values with zero.
           Reason: baseline VEGAS cannot deal with NaN values"""
        probs = self.evaluate_DP_on_unit_interval(samples)
        probs = jnp.nan_to_num(probs, nan=0.)
        return probs
