"""Probability distributions for training with JAX/Flax."""

from typing import Any

import gin
import gymnasium as gym  # Modern fork of gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn


def logsoftmax_sample(
    logits: jnp.ndarray,
    temperature: float = 1.0,
    rng: jax.Array | None = None
) -> jnp.ndarray:
    """Sample from a categorical distribution using Gumbel-max trick.
    
    Args:
        logits: Unnormalized log probabilities of shape [..., n_categories].
        temperature: Sampling temperature. Higher = more uniform.
        rng: JAX random key. If None, uses numpy random (not recommended for JIT).
    
    Returns:
        Sampled indices of shape [...].
    """
    if rng is None:
        # Fallback to numpy random (works but not JIT-compatible)
        gumbel_noise = np.random.gumbel(size=logits.shape).astype(logits.dtype)
    else:
        gumbel_noise = jax.random.gumbel(rng, shape=logits.shape, dtype=logits.dtype)
    
    return jnp.argmax(logits / temperature + gumbel_noise, axis=-1)


class Distribution:
    """Abstract class for parametrized probability distributions."""

    @property
    def n_inputs(self) -> int:
        """Returns the number of inputs to the distribution (i.e. parameters)."""
        raise NotImplementedError

    def sample(
        self,
        inputs: jnp.ndarray,
        temperature: float = 1.0,
        rng: jax.Array | None = None
    ) -> jnp.ndarray:
        """Sample a point from the distribution.

        Args:
            inputs: Distribution inputs. Shape is subclass-specific.
                Broadcasts along the first dimensions. For example, in the
                categorical distribution parameter shape is (C,), where C is
                the number of categories. If (B, C) is passed, the object will
                represent a batch of B categorical distributions with different
                parameters.
            temperature: Sampling temperature; 1.0 is default, at 0.0 chooses
                the most probable (preferred) action.
            rng: JAX random key for sampling.

        Returns:
            Sampled point of shape dependent on the subclass and on the shape
            of inputs.
        """
        raise NotImplementedError

    def log_prob(self, inputs: jnp.ndarray, point: jnp.ndarray) -> jnp.ndarray:
        """Retrieve log probability (or log probability density) of a point.

        Args:
            inputs: Distribution parameters.
            point: Point from the distribution. Shape should be consistent
                with inputs.

        Returns:
            Array of log probabilities of points in the distribution.
        """
        raise NotImplementedError

    def log_prob_layer(self) -> nn.Module:
        """Build a Flax module that computes log probability."""
        return LogProbLayer(self)

    @staticmethod
    def from_name(name: str, **kwargs) -> "Distribution":
        """Factory function for distributions.
        
        Returns instance of distribution specified by name.
        """
        distributions = {
            "categorical": Categorical,
            "gaussian": Gaussian,
            "laplace": Laplace,
            "cauchy": Cauchy,
            "tstudent": TStudent,
        }
        if name not in distributions:
            raise ValueError(f"Distribution {name} does not exist!")
        return distributions[name](**kwargs)


class LogProbLayer(nn.Module):
    """Flax module that computes log probability for a distribution."""
    distribution: Distribution

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, point: jnp.ndarray) -> jnp.ndarray:
        return self.distribution.log_prob(inputs, point)


@gin.configurable(module="code.distributions", denylist=["n_categories", "shape"])
class Categorical(Distribution):
    """Categorical distribution parametrized by logits."""

    def __init__(self, n_categories: int, shape: tuple[int, ...] = ()):
        """Initialize Categorical distribution.

        Args:
            n_categories: Number of categories.
            shape: Shape of the sample.
        """
        self._n_categories = n_categories
        self._shape = shape

    @property
    def n_inputs(self) -> int:
        return int(np.prod(self._shape, dtype=np.int32)) * self._n_categories

    def _unflatten_inputs(self, inputs: jnp.ndarray) -> jnp.ndarray:
        return jnp.reshape(
            inputs, inputs.shape[:-1] + self._shape + (self._n_categories,)
        )

    def sample(
        self,
        inputs: jnp.ndarray,
        temperature: float = 1.0,
        rng: jax.Array | None = None
    ) -> jnp.ndarray:
        # No need for LogSoftmax with sampling - softmax normalization is
        # subtracting a constant from every logit, and sampling is taking
        # a max over logits plus noise, so invariant to adding a constant.
        if temperature == 0.0:
            return jnp.argmax(self._unflatten_inputs(inputs), axis=-1)
        return logsoftmax_sample(self._unflatten_inputs(inputs), temperature, rng)

    def log_prob(self, inputs: jnp.ndarray, point: jnp.ndarray) -> jnp.ndarray:
        log_probs = jax.nn.log_softmax(self._unflatten_inputs(inputs))
        return jnp.sum(
            # Select the logits specified by point.
            log_probs * jax.nn.one_hot(point, self._n_categories),
            # Sum over the parameter dimensions.
            axis=[-a for a in range(1, len(self._shape) + 2)],
        )

    def entropy(self, inputs: jnp.ndarray) -> jnp.ndarray:
        log_probs = jax.nn.log_softmax(inputs)
        probs = jnp.exp(log_probs)
        return -jnp.sum(probs * log_probs, axis=-1)


class ScaleLocationDistribution(Distribution):
    """Independent multivariate distribution parametrized by location and scale."""

    def __init__(
        self,
        shape: tuple[int, ...],
        scale: float,
        learn_scale: str | None
    ):
        """Initialize the distribution.

        Args:
            shape: Shape of the sample.
            scale: E.g. standard deviation, shared across the whole sample.
            learn_scale: How to learn the scale - 'shared' to have a single,
                shared scale parameter, or 'separate' to have separate
                parameters for each dimension.
        """
        self._shape = shape
        self._scale = scale
        self._learn_scale = learn_scale

    def sampling_function(
        self,
        location: jnp.ndarray,
        scale: jnp.ndarray,
        temperature: float,
        rng: jax.Array | None = None
    ) -> jnp.ndarray:
        raise NotImplementedError()

    @property
    def _n_dims(self) -> int:
        return int(np.prod(self._shape, dtype=np.int32))

    def _params(
        self,
        inputs: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Extract the location and scale parameters from the inputs."""
        if inputs.shape[-1] != self.n_inputs:
            raise ValueError(
                f"Invalid distribution parametrization - expected {self.n_inputs} "
                f"parameters, got {inputs.shape[-1]}. Input shape: {inputs.shape}."
            )
        n_dims = self._n_dims
        # Split the distribution inputs into two parts: mean and std.
        location = inputs[..., :n_dims]
        if self._learn_scale is not None:
            scale = inputs[..., n_dims:]
            # Std should be non-negative - but it might not be when it comes
            # from a network output. Therefore we will use Softplus on it.
            scale = jax.nn.softplus(scale) + self._scale
        else:
            scale = self._scale
        # In case of constant or shared std, upsample it to the same
        # dimensionality as the means.
        scale = jnp.broadcast_to(scale, location.shape)
        return (location, scale)

    @property
    def n_inputs(self) -> int:
        n_dims = self._n_dims
        return {
            None: n_dims,
            "shared": n_dims + 1,
            "separate": n_dims * 2,
        }[self._learn_scale]

    def sample(
        self,
        inputs: jnp.ndarray,
        temperature: float = 1.0,
        rng: jax.Array | None = None
    ) -> jnp.ndarray:
        location, scale = self._params(inputs)
        location = jnp.reshape(location, location.shape[:-1] + self._shape)
        scale = jnp.reshape(scale, scale.shape[:-1] + self._shape)
        if temperature == 0:
            return location
            # This seemingly strange if solves the problem
            # of calling np/jnp.random in the metric PreferredMove

        return self.sampling_function(location, scale, temperature, rng)


@gin.configurable(module="code.distributions", denylist=["shape"])
class Gaussian(ScaleLocationDistribution):
    """Independent multivariate Gaussian distribution."""

    def __init__(
        self,
        shape: tuple[int, ...] = (),
        scale: float = 1.0,
        learn_scale: str | None = None
    ):
        super().__init__(shape, scale, learn_scale)

    def sampling_function(
        self,
        location: jnp.ndarray,
        scale: jnp.ndarray,
        temperature: float,
        rng: jax.Array | None = None
    ) -> jnp.ndarray:
        if rng is None:
            noise = np.random.normal(size=location.shape).astype(location.dtype)
        else:
            noise = jax.random.normal(rng, shape=location.shape, dtype=location.dtype)
        return location + noise * scale * temperature

    def log_prob(self, inputs: jnp.ndarray, point: jnp.ndarray) -> jnp.ndarray:
        point = point.reshape(inputs.shape[:-1] + (-1,))
        mean, std = self._params(inputs)
        return -jnp.sum(
            # Scaled distance.
            (point - mean) ** 2 / (2 * std**2) +
            # Normalizing constant.
            (jnp.log(std) + jnp.log(jnp.sqrt(2 * jnp.pi))),
            axis=-1,
        )

    def entropy(self, inputs: jnp.ndarray) -> jnp.ndarray:
        _, std = self._params(inputs)
        return jnp.sum(jnp.exp(std) + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e), axis=-1)


@gin.configurable(module="code.distributions", denylist=["shape"])
class Laplace(ScaleLocationDistribution):
    """Independent multivariate Laplace distribution."""

    def __init__(
        self,
        shape: tuple[int, ...] = (),
        scale: float = 1.0,
        learn_scale: str | None = None
    ):
        super().__init__(shape, scale, learn_scale)

    def sampling_function(
        self,
        location: jnp.ndarray,
        scale: jnp.ndarray,
        temperature: float,
        rng: jax.Array | None = None
    ) -> jnp.ndarray:
        if rng is None:
            noise = np.random.laplace(size=location.shape).astype(location.dtype)
        else:
            noise = jax.random.laplace(rng, shape=location.shape, dtype=location.dtype)
        return location + noise * scale * temperature

    def log_prob(self, inputs: jnp.ndarray, point: jnp.ndarray) -> jnp.ndarray:
        point = point.reshape(inputs.shape[:-1] + (-1,))
        location, diversity = self._params(inputs)
        return -jnp.sum(
            # Scaled distance.
            jnp.abs(point - location) / diversity +
            # Normalizing constant.
            jnp.log(2 * diversity),
            axis=-1,
        )

    def entropy(self, inputs: jnp.ndarray) -> jnp.ndarray:
        _, diversity = self._params(inputs)
        return jnp.sum(jnp.log(2.0 * diversity * jnp.e), axis=-1)


@gin.configurable(module="code.distributions", denylist=["shape"])
class Cauchy(ScaleLocationDistribution):
    """Independent multivariate Cauchy distribution."""

    def __init__(
        self,
        shape: tuple[int, ...] = (),
        scale: float = 1.0,
        learn_scale: str | None = None
    ):
        super().__init__(shape, scale, learn_scale)

    def sampling_function(
        self,
        location: jnp.ndarray,
        scale: jnp.ndarray,
        temperature: float,
        rng: jax.Array | None = None
    ) -> jnp.ndarray:
        if rng is None:
            noise = np.random.standard_cauchy(size=location.shape).astype(location.dtype)
        else:
            noise = jax.random.cauchy(rng, shape=location.shape, dtype=location.dtype)
        return location + noise * scale * temperature

    def log_prob(self, inputs: jnp.ndarray, point: jnp.ndarray) -> jnp.ndarray:
        point = point.reshape(inputs.shape[:-1] + (-1,))
        location, scale = self._params(inputs)
        return -jnp.sum(
            # Scaled distance.
            jnp.log((point - location) ** 2 + scale**2) +
            # Normalizing constant.
            jnp.log(jnp.pi / scale),
            axis=-1,
        )

    def entropy(self, inputs: jnp.ndarray) -> jnp.ndarray:
        _, scale = self._params(inputs)
        return jnp.sum(jnp.log(4.0 * scale * jnp.pi), axis=-1)


@gin.configurable(module="code.distributions", denylist=["shape"])
class TStudent(ScaleLocationDistribution):
    """Independent multivariate T-student distribution with 5 degrees of freedom.
    
    Parametrized by location and scale.

    WARNING: it does not implement proper log_prob function (does not sum up 
    to 0), only an equivalent from the perspective of gradient optimization.
    """

    def __init__(
        self,
        shape: tuple[int, ...] = (),
        scale: float = 1.0,
        learn_scale: str | None = None
    ):
        super().__init__(shape, scale, learn_scale)
        self.degree = 5

    def sampling_function(
        self,
        location: jnp.ndarray,
        scale: jnp.ndarray,
        temperature: float,
        rng: jax.Array | None = None
    ) -> jnp.ndarray:
        if rng is None:
            noise = np.random.standard_t(self.degree, size=location.shape).astype(location.dtype)
        else:
            # JAX doesn't have t-distribution directly, use transformation
            # t = normal / sqrt(chi2 / df)
            rng1, rng2 = jax.random.split(rng)
            normal = jax.random.normal(rng1, shape=location.shape, dtype=location.dtype)
            # Chi-squared with df degrees of freedom
            chi2 = jax.random.gamma(rng2, self.degree / 2, shape=location.shape, dtype=location.dtype) * 2
            noise = normal / jnp.sqrt(chi2 / self.degree)
        return location + noise * scale * temperature

    def log_prob(self, inputs: jnp.ndarray, point: jnp.ndarray) -> jnp.ndarray:
        point = point.reshape(inputs.shape[:-1] + (-1,))
        location, scale = self._params(inputs)
        return jnp.sum(
            jnp.log(1 + (point - location) ** 2 / scale**2 / self.degree)
            * (-(self.degree + 1) / 2)
            - jnp.log(scale),
            axis=-1,
        )


def create_distribution(space: gym.Space) -> Distribution:
    """Create a Distribution for the given Gymnasium space."""
    if isinstance(space, gym.spaces.Discrete):
        return Categorical(shape=(), n_categories=int(space.n))
    elif isinstance(space, gym.spaces.MultiDiscrete):
        assert space.nvec.size
        assert min(space.nvec) == max(space.nvec), (
            f"Every dimension must have the same number of categories, got {space.nvec}."
        )
        return Categorical(shape=(len(space.nvec),), n_categories=int(space.nvec[0]))
    elif isinstance(space, gym.spaces.Box):
        return Gaussian(shape=space.shape)
    else:
        raise TypeError(f"Space {space} unavailable as a distribution support.")


class LogLoss(nn.Module):
    """Flax module that computes log loss for a Distribution."""
    distribution: Distribution

    @nn.compact
    def __call__(
        self,
        inputs: jnp.ndarray,
        point: jnp.ndarray,
        weights: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        """Compute weighted negative log probability.
        
        Args:
            inputs: Distribution parameters.
            point: Target points.
            weights: Optional weights for each sample.
        
        Returns:
            Scalar loss value.
        """
        log_prob = self.distribution.log_prob(inputs, point)
        neg_log_prob = -log_prob
        
        if weights is not None:
            return jnp.sum(neg_log_prob * weights) / jnp.sum(weights)
        return jnp.mean(neg_log_prob)


def log_loss_fn(
    distribution: Distribution,
    inputs: jnp.ndarray,
    point: jnp.ndarray,
    weights: jnp.ndarray | None = None
) -> jnp.ndarray:
    """Functional version of log loss computation.
    
    Args:
        distribution: The distribution to compute loss for.
        inputs: Distribution parameters.
        point: Target points.
        weights: Optional weights for each sample.
    
    Returns:
        Scalar loss value.
    """
    log_prob = distribution.log_prob(inputs, point)
    neg_log_prob = -log_prob
    
    if weights is not None:
        return jnp.sum(neg_log_prob * weights) / jnp.sum(weights)
    return jnp.mean(neg_log_prob)
