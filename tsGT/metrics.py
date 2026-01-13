"""Custom metrics and loss functions for training."""

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


def _smooth_target(
    targets: jnp.ndarray,
    n_categories: int,
    precision: int,
    std: float
) -> jnp.ndarray:
    """Create smoothed target distribution using discretized normal.
    
    Args:
        targets: Target indices of shape [...].
        n_categories: Number of categories.
        precision: Precision parameter for varying std across positions.
        std: Base standard deviation for smoothing.
    
    Returns:
        Smoothed probability distribution of shape [..., n_categories].
    """
    # Create position-dependent standard deviations
    stds = jnp.ones(targets.shape[-1])
    for i in range(precision):
        stds = stds.at[i::precision].set(i)
    stds = jnp.expand_dims(std * n_categories ** stds, axis=-1)

    # Compute smoothed targets using discretized normal distribution
    target_smooth = (
        jnp.zeros(targets.shape + (n_categories,))
        + jnp.arange(n_categories)
        - jnp.expand_dims(targets, axis=-1)
    ) / stds

    # Convert to log probabilities and normalize
    log_pdf = -target_smooth ** 2 / 2
    normalizer = logsumexp(log_pdf, axis=-1, keepdims=True)
    return jnp.exp(log_pdf - normalizer)


def weighted_smoothed_category_cross_entropy(
    model_output: jnp.ndarray,
    targets: jnp.ndarray,
    weights: jnp.ndarray,
    std: float,
    precision: int
) -> jnp.ndarray:
    """Compute weighted cross entropy with smoothed categorical labels.
    
    Uses a discretized normal distribution to smooth the target labels,
    which can help with training stability and generalization.
    
    Args:
        model_output: Logits of shape [..., n_categories].
        targets: Target indices of shape [...].
        weights: Sample weights of shape [...].
        std: Base standard deviation for label smoothing.
        precision: Precision parameter for position-dependent smoothing.
    
    Returns:
        Scalar weighted cross entropy loss.
    """
    n_categories = model_output.shape[-1]
    target_smooth = _smooth_target(targets, n_categories, precision, std)
    model_log_distributions = jax.nn.log_softmax(model_output)
    cross_entropies = -jnp.sum(target_smooth * model_log_distributions, axis=-1)
    return jnp.sum(cross_entropies * weights) / jnp.sum(weights)


def create_weighted_smoothed_category_cross_entropy(
    std: float,
    precision: int
):
    """Factory function to create a smoothed cross entropy loss function.
    
    Args:
        std: Base standard deviation for label smoothing.
        precision: Precision parameter for position-dependent smoothing.
    
    Returns:
        Loss function with signature (model_output, targets, weights) -> scalar.
    """
    def loss_fn(
        model_output: jnp.ndarray,
        targets: jnp.ndarray,
        weights: jnp.ndarray
    ) -> jnp.ndarray:
        return weighted_smoothed_category_cross_entropy(
            model_output, targets, weights, std, precision
        )
    
    return loss_fn


# Alias for backward compatibility with Trax naming convention
def WeightedSmoothedCategoryCrossEntropy(std: float, precision: int):
    """Create smoothed cross entropy loss (Trax-style factory).
    
    Args:
        std: Base standard deviation for label smoothing.
        precision: Precision parameter for position-dependent smoothing.
    
    Returns:
        Loss function with signature (model_output, targets, weights) -> scalar.
    """
    return create_weighted_smoothed_category_cross_entropy(std, precision)
