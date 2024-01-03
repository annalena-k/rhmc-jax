from typing import Callable, Tuple
import chex
import jax
import jax.numpy as jnp
import pylhe
import madjax

LogProbTargetFn = Callable[[chex.Array], chex.Array]
ProbTargetFn = Callable[[chex.Array], chex.Array]
class Target:
    dim: int
    name: str
    log_prob: LogProbTargetFn
    prob: ProbTargetFn

def move_away_from_boundary(samples, epsilon):
    return jnp.minimum(jnp.maximum(samples, jnp.ones_like(samples)*epsilon), jnp.ones_like(samples)*(1.0 - epsilon))


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

def read_madgraph_phasespace_points(filename: str, target: Target, n_events: int):
    """Read lhe (=Les Houches Event) file and convert events to phase space points.
    filename: either .lhe or .lhe.gz
    target: Target that has the method 'get_phase_space_generator'
    n_events: number of events to load
    """
    def extract_incoming_and_outgoing_particles(event):
        particles = []
        for p in event.particles:
            if p.status==-1 or p.status==1: # -1: incoming, 1: outgoing & stable particles
                particles.append([p.e, p.px, p.py, p.pz])
        return jnp.array(particles)

    def get_n_lhe_events(lhe_event_generator, n_events):
        events = []
        for _ in range(n_events):
            try:
                event = lhe_event_generator.__next__()
            except:
                print("No More Events")
                return jnp.array(events)
            
            relevant_particles = extract_incoming_and_outgoing_particles(event)
            events.append(relevant_particles)
        
        return jnp.array(events)

    def get_phase_space_points_from_events(events):
        momenta = jnp.array(events)
        momenta_vec = [madjax.phasespace.vectors.Vector(p) for p in momenta]
        ps_points, _ = target.phase_space_generator.invertKinematics(target.E_cm, momenta_vec)
        return jnp.array(ps_points)

    get_phase_space_points_from_event_vec = jax.vmap(get_phase_space_points_from_events, (0))

    lhe_events = pylhe.read_lhe_with_attributes(filename)
    events = get_n_lhe_events(lhe_events, n_events)
    ps_points = get_phase_space_points_from_event_vec(events)

    return ps_points