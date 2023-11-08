import chex
import jax
import jax.numpy as jnp
import jax.scipy.stats as jsps

class GaussiansOnCircle():
    """Create a 2D target distribution with Gaussians on a circle restricted to [0,1]^2.

    Args:
        n_gaussians: number of Gaussians. int
        radius: radius of the circle on which the Gaussian means are located. float
        variance: variance of the gaussians. float
        center: center location of ring. 2D jax.Array
    """
    def __init__(
        self, 
        n_gaussians: int = 6,
        radius: float = .48,
        variance: float = .1,
        center: chex.Array = jnp.array([.5, .5])
        
    ):
        self.n_gaussians = n_gaussians
        self.radius = radius
        self.variance = variance
        radial_pos = jnp.linspace(0, jnp.pi*2, num=n_gaussians, endpoint=False)
        self.mean_pos = center + radius * jnp.column_stack((jnp.sin(radial_pos), jnp.cos(radial_pos)))
        
    def prob(self, samples):
        probs = []
        for mean in self.mean_pos:
            standardized_samples = (samples - mean[None, :])/self.variance
            prob_vals = jsps.multivariate_normal.pdf(standardized_samples, mean=jnp.array([0,0]), cov=jnp.eye(2))
            probs.append(prob_vals)
        probs = jnp.sum(jnp.array(probs), axis=0).squeeze()
        assert len(probs) == len(samples)
        return probs
    
    def log_prob(self, samples):
        return self.prob(samples)

    def sample(self, key, n_samples):
        key, subkey = jax.random.split(key)
        n_samples_per_gaussian = int(jnp.ceil(n_samples / self.n_gaussians))
        samples = []
        for mean in self.mean_pos:
            key, subkey= jax.random.split(key)
            noise = self.variance * jax.random.normal(subkey, shape=[2*n_samples_per_gaussian,2])
            sampled_points = mean[None, :] + noise
            
            val1 = jnp.where(sampled_points[:,0]>=0, True, False)
            val2 = jnp.where(sampled_points[:,0]<=1, True, False)
            val3 = jnp.where(sampled_points[:,1]>=0, True, False)
            val4 = jnp.where(sampled_points[:,1]<=1, True, False)
            valid = val1 * val2 * val3 * val4
            samples.append(sampled_points[valid][:n_samples_per_gaussian])
        
        key, subkey = jax.random.split(key)
        samples = jnp.array(samples).reshape([-1,2])
        p = jax.random.permutation(subkey, samples.shape[0])
        samples = samples[p][:n_samples]

        return samples
