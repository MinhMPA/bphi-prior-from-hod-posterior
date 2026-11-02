"""Gaussian HOD priors marginalized over a shared flat fNL, in product coordinates."""
import numpy as np
from scipy.special import logsumexp

FMAX = 300.
FMIN = 1.e-8
NPOSITIVE = 4096


def quadrature(npositive=NPOSITIVE, fmin=FMIN):
    # Resolve the narrow conditional fNL density near zero without evaluating zero.
    edges = np.geomspace(fmin, FMAX, npositive+1)
    centers = np.sqrt(edges[:-1]*edges[1:])
    grid = np.concatenate([-centers[::-1], centers])
    widths = np.concatenate([np.diff(edges)[::-1], np.diff(edges)])
    return grid, np.log(widths / widths.sum())


def log_kernel(products, grid, means, stds, xp=np):
    products = xp.asarray(products)
    means, stds, grid = xp.asarray(means), xp.asarray(stds), xp.asarray(grid)
    ratio = products[..., :, None] / grid
    terms = -.5*((ratio-means[:, None])/stds[:, None])**2
    terms -= xp.log(stds[:, None]) + .5*np.log(2*np.pi) + xp.log(xp.abs(grid))
    return xp.sum(terms, axis=-2)


def log_induced(products, means, stds, npositive=NPOSITIVE, fmin=FMIN):
    grid, weights = quadrature(npositive, fmin)
    return logsumexp(log_kernel(products, grid, means, stds)+weights, axis=-1)


def make_prior(names, means, stds):
    from desilike.likelihoods import BaseLikelihood
    from desilike.parameter import Parameter, ParameterCollection
    from desilike.jax import numpy as jnp
    from jax.scipy.special import logsumexp as jlogsumexp

    class GaussianProductPrior(BaseLikelihood):
        @staticmethod
        def _params(params, names=None):
            params = ParameterCollection(params)
            for index, name in enumerate(names or []):
                params.set(Parameter(f'product_prior_{index}', derived='{'+name+'}', fixed=True))
            return params

        def initialize(self, names, means, stds):
            super().initialize(name='gaussian_product_prior')
            self.names = list(names)
            self.means, self.stds = np.asarray(means), np.asarray(stds)
            self.grid, self.weights = quadrature()

        def calculate(self, **params):
            values = jnp.stack([jnp.asarray(params[f'product_prior_{i}']).reshape(())
                                for i in range(len(self.names))])
            self.loglikelihood = jlogsumexp(
                log_kernel(values, self.grid, self.means, self.stds, xp=jnp) + self.weights)

    return GaussianProductPrior(names=names, means=means, stds=stds)


def reconstruct(products, means, stds, chunk_size=128):
    """Mixture of p(f|B): remove Q(B) exactly once using the SAME fitting quadrature."""
    products = np.asarray(products)
    if products.ndim != 2 or len(products) == 0 or not np.all(np.isfinite(products)):
        raise ValueError('Need a nonempty finite (samples, tracers) product array.')
    grid, logweights = quadrature()
    accumulator = np.full(len(grid), -np.inf)
    for start in range(0, len(products), chunk_size):
        terms = log_kernel(products[start:start+chunk_size], grid, means, stds)+logweights
        terms -= logsumexp(terms, axis=1)[:, None]
        accumulator = np.logaddexp(accumulator, logsumexp(terms, axis=0))
    mass = np.exp(accumulator-logsumexp(accumulator))
    return grid, mass


def summarize_grid(grid, mass):
    cdf = np.cumsum(mass)-.5*mass
    q025, q16, median, q84, q975 = np.interp([.025, .16, .5, .84, .975], cdf, grid)
    return dict(mean=float(grid @ mass), median=float(median), q16=float(q16), q84=float(q84),
                width68=float(q84-q16), q025=float(q025), q975=float(q975), width95=float(q975-q025))
