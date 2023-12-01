import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as jsps

from rhmcjax.targets.target_util import check_restriction_to_unit_hypercube

class GaussiansOnCircle():
    """Create a 2D target distribution with Gaussians on a circle restricted to [0,1]^2.

    Args:
        n_gaussians: number of Gaussians. int
        radius: radius of the circle on which the Gaussian means are located. float
        variance: variance of the gaussians. float
        center: center location of ring. jax.Array of shape [dim]
        restrict_to_unit_hypercube: whether distribution is restricted to unit hypercube or not. bool
    """
    def __init__(
        self, 
        n_gaussians: int = 6,
        radius: float = .48,
        variance: float = .1,
        center: chex.Array = jnp.array([.5, .5]),
        restrict_to_unit_hypercube: bool = True
        
    ):
        self.n_gaussians = n_gaussians
        self.radius = radius
        self.variance = variance
        radial_pos = jnp.linspace(0, jnp.pi*2, num=n_gaussians, endpoint=False)
        self.mean_pos = center + radius * jnp.column_stack((jnp.sin(radial_pos), jnp.cos(radial_pos)))
        self.restrict_to_unit_hypercube = restrict_to_unit_hypercube
        if restrict_to_unit_hypercube:
            within_hypercube = check_restriction_to_unit_hypercube(center.reshape([-1, 2]))
            if not within_hypercube:
                print(f'Attention! Center {center} not within unit hypercube. Gaussian peaks can vanish.')
            if radius > 1.:
                print(f'Attention! Radius {radius} > 1. Gaussian peaks can vanish.')

        
    def prob(self, samples):
        if len(samples.shape) == 1:
            samples = jnp.expand_dims(samples, axis=0)
        probs = []
        for mean in self.mean_pos:
            standardized_samples = (samples - mean[None, :])/self.variance
            prob_vals = jsps.multivariate_normal.pdf(standardized_samples, mean=jnp.array([0,0]), cov=jnp.eye(2))
            probs.append(prob_vals)
        probs = jnp.sum(jnp.array(probs), axis=0)
        if self.restrict_to_unit_hypercube:
            mask = check_restriction_to_unit_hypercube(samples)
            probs = jnp.where(mask, probs, 0.)
        assert len(probs) == len(samples)

        return probs.squeeze()
    
    def log_prob(self, samples):
        return jnp.log(self.prob(samples))

    def sample(self, key, n_samples):
        key, subkey = jax.random.split(key)
        n_samples_per_gaussian = int(jnp.ceil(n_samples / self.n_gaussians))
        samples = []
        for mean in self.mean_pos:
            key, subkey= jax.random.split(key)
            noise = self.variance * jax.random.normal(subkey, shape=[2*n_samples_per_gaussian, 2])
            sampled_points = mean[None, :] + noise
            if self.restrict_to_unit_hypercube:
                mask = check_restriction_to_unit_hypercube(sampled_points)
                samples.append(sampled_points[mask][:n_samples_per_gaussian])
            else:
                samples.append(sampled_points[:n_samples_per_gaussian])
        
        samples = jnp.array(samples).reshape([-1,2])
        key, subkey = jax.random.split(key)
        p = jax.random.permutation(subkey, samples.shape[0])
        samples = samples[p][:n_samples]

        return samples
