"""Normalization utilities for time series data."""

from typing import NamedTuple, Callable

import jax.numpy as jnp
import numpy as np


class Normalizer:
    """Base class for data normalizers."""
    
    def __init__(self, regularizer: float, apply: str):
        """Initialize normalizer.
        
        Args:
            regularizer: Small constant to prevent division by zero.
            apply: Name of the normalization method.
        """
        self.regularizer = regularizer
        self.apply = apply

    def normalize(self, data: jnp.ndarray, mask: jnp.ndarray | None = None):
        """Normalize data.
        
        Args:
            data: Input data array.
            mask: Optional mask array.
        
        Returns:
            Tuple of (normalized_data, normalization_params, mask_modifier).
        """
        raise NotImplementedError()

    def denormalize(self, series: jnp.ndarray, normalization_params):
        """Denormalize data.
        
        Args:
            series: Normalized data.
            normalization_params: Parameters from normalize().
        
        Returns:
            Denormalized data.
        """
        raise NotImplementedError()

    def as_autoregressive_pipeline_fn(
        self, 
        use_mask: bool
    ) -> Callable:
        """Create a function for use in autoregressive pipeline.
        
        Normalizes both series and target separately using exactly the same
        method. Assumes series == target and inputs is discarded by model body.
        Used by SerialPredictor.
        
        Args:
            use_mask: Whether to use the mask for normalization.
        
        Returns:
            Function that normalizes (series, inputs, target, mask) tuples.
        """
        assert isinstance(use_mask, bool), "use_mask must be a boolean"

        def wrapper(
            series: jnp.ndarray,
            inputs: jnp.ndarray,
            target: jnp.ndarray,
            mask: jnp.ndarray
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            assert not use_mask or mask is not None, "mask required when use_mask=True"
            mask_ = mask if use_mask else None
            
            norm_series, _norm_params, _series_mask_modifier = self.normalize(
                data=series, mask=mask_
            )
            norm_target, _norm_params, target_mask_modifier = self.normalize(
                data=target, mask=mask_
            )
            return norm_series, inputs, norm_target, mask * target_mask_modifier

        return wrapper

    # Legacy alias for backward compatibility
    def as_autoregressive_pipeline_layer(self, use_mask: bool) -> Callable:
        """Legacy alias for as_autoregressive_pipeline_fn."""
        return self.as_autoregressive_pipeline_fn(use_mask)

    @staticmethod
    def from_name(name: str | None, regularizer: float) -> "Normalizer":
        """Factory function to create normalizer by name.
        
        Args:
            name: Normalizer name ('per_ts', 'per_batch', 'causal', or None).
            regularizer: Regularization constant.
        
        Returns:
            Normalizer instance.
        """
        normalizers = {
            "per_ts": PerTsNormalizer,
            "per_batch": PerBatchNormalizer,
            "causal": CausalNormalizer,
            None: NOPNormalizer,
            False: NOPNormalizer,
        }
        
        if name not in normalizers:
            raise ValueError(f"Normalization method `{name}` does not exist!")
        
        return normalizers[name](regularizer=regularizer)


class PerTsNormalizer(Normalizer):
    """Per-time-series normalization using absolute mean."""
    
    class _NormalizationParams(NamedTuple):
        scaling_factor: np.ndarray

    def __init__(self, regularizer: float):
        super().__init__(regularizer, "per_ts")

    def normalize(
        self, 
        data: jnp.ndarray, 
        mask: jnp.ndarray | None = None
    ) -> tuple[jnp.ndarray, _NormalizationParams, jnp.ndarray]:
        """Normalize each time series by its absolute mean.
        
        Args:
            data: Input data of shape (batch, length).
            mask: Optional mask of same shape.
        
        Returns:
            Tuple of (normalized_data, params, mask_modifier).
        """
        assert mask is None or data.shape == mask.shape

        if mask is None:
            use_for_statistics = jnp.ones_like(data)
        else:
            use_for_statistics = mask

        # Compute absolute mean per time series
        absmean = jnp.abs(data).mean(
            axis=1, where=(use_for_statistics > 0), keepdims=True
        )

        scaling_factor = absmean + self.regularizer
        scaled = jnp.divide(data, scaling_factor)

        return scaled, self._NormalizationParams(scaling_factor), jnp.ones_like(scaled)

    def denormalize(
        self, 
        series: jnp.ndarray, 
        normalization_params: _NormalizationParams
    ) -> jnp.ndarray:
        """Denormalize by multiplying with scaling factor."""
        scaling_factor = normalization_params.scaling_factor
        assert jnp.squeeze(scaling_factor, axis=1).shape == (series.shape[0],)
        return jnp.multiply(series, scaling_factor)


class PerBatchNormalizer(Normalizer):
    """Per-batch normalization using global absolute mean."""
    
    class _NormalizationParams(NamedTuple):
        scaling_factor: np.ndarray

    def __init__(self, regularizer: float):
        super().__init__(regularizer, "per_batch")

    def normalize(
        self, 
        data: jnp.ndarray, 
        mask: jnp.ndarray | None = None
    ) -> tuple[jnp.ndarray, _NormalizationParams, jnp.ndarray]:
        """Normalize entire batch by global absolute mean.
        
        Args:
            data: Input data of shape (batch, length).
            mask: Optional mask of same shape.
        
        Returns:
            Tuple of (normalized_data, params, mask_modifier).
        """
        assert mask is None or data.shape == mask.shape

        if mask is None:
            use_for_statistics = jnp.ones_like(data)
        else:
            use_for_statistics = 1 - mask  # Use non-masked values

        # Compute global absolute mean
        absmean = jnp.abs(data).mean(where=(use_for_statistics > 0), keepdims=True)

        scaling_factor = absmean + self.regularizer
        scaled = jnp.divide(data, scaling_factor)

        return scaled, self._NormalizationParams(scaling_factor), jnp.ones_like(scaled)

    def denormalize(
        self, 
        series: jnp.ndarray, 
        normalization_params: _NormalizationParams
    ) -> jnp.ndarray:
        """Denormalize by multiplying with scaling factor."""
        scaling_factor = normalization_params.scaling_factor
        assert jnp.squeeze(scaling_factor).shape == ()
        return jnp.multiply(series, scaling_factor)


class CausalNormalizer(Normalizer):
    """Causal normalization using cumulative statistics.
    
    This normalizer ensures no information leakage from future timesteps
    by using only past values for normalization at each position.
    """
    
    # Number of initial timesteps to exclude from gradient computation
    GRADIENT_STOP_INDEX = 10
    
    class _NormalizationParams(NamedTuple):
        first_value: np.ndarray

    def __init__(self, regularizer: float):
        super().__init__(regularizer, "causal")

    def normalize(
        self, 
        data: jnp.ndarray, 
        mask: jnp.ndarray | None = None
    ) -> tuple[jnp.ndarray, _NormalizationParams, jnp.ndarray]:
        """Normalize using causal (cumulative) statistics.
        
        Args:
            data: Input data of shape (batch, length).
            mask: Optional mask (not used, causal normalization prevents leak).
        
        Returns:
            Tuple of (normalized_data, params, mask_modifier).
        """
        assert mask is None or data.shape == mask.shape

        # Compute cumulative absolute mean (causal)
        abssum = jnp.cumsum(jnp.abs(data), axis=-1)
        indices = jnp.arange(data.shape[-1]) + 1
        absmean = abssum / indices

        scaling_factor = absmean + self.regularizer
        scaled = jnp.divide(data, scaling_factor)

        params = self._NormalizationParams(first_value=data[:, 0])

        # Mask out initial timesteps for gradient computation
        mask_diff = np.ones(scaled.shape)
        mask_diff[:, :self.GRADIENT_STOP_INDEX] = 0

        return scaled, params, mask_diff

    def denormalize(
        self, 
        series: jnp.ndarray, 
        normalization_params: _NormalizationParams
    ) -> jnp.ndarray:
        """Denormalize causally normalized data.
        
        This requires iterative computation since each value depends
        on the previous denormalized values.
        """
        first_value = normalization_params.first_value

        prefix_sum = np.abs(first_value)
        scaled = np.empty_like(series)
        
        for n in range(series.shape[1]):
            if n == 0:
                scaled[:, n] = first_value
            else:
                predicted_value = series[:, n]
                true_value = (
                    predicted_value * (prefix_sum + self.regularizer * (n + 1))
                ) / (n + 1 - np.abs(predicted_value))
                prefix_sum += np.abs(true_value)
                scaled[:, n] = true_value

        return scaled


class NOPNormalizer(Normalizer):
    """No-op normalizer that passes data through unchanged."""
    
    def __init__(self, regularizer: float):
        super().__init__(regularizer, "NOP")

    def normalize(
        self, 
        data: jnp.ndarray, 
        mask: jnp.ndarray | None = None
    ) -> tuple[jnp.ndarray, None, jnp.ndarray]:
        """Return data unchanged."""
        return data, None, jnp.ones_like(data)

    def denormalize(
        self, 
        series: jnp.ndarray, 
        normalization_params
    ) -> jnp.ndarray:
        """Return data unchanged."""
        return series
