"""Distribution-based predictor for time series forecasting."""

from functools import partial
from typing import Any, Callable

import gin
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

import decoding
import distributions
from input_injection import InjectInputs
from normalization import Normalizer
from time_series_predictor import TimeSeriesPredictor


# =============================================================================
# Decoder model
# =============================================================================

class DistributionDecoderModel(nn.Module):
    """Decoder that outputs distribution parameters.
    
    Attributes:
        model_body: The transformer body model.
        d_emb: Embedding dimension.
        input_vocab_sizes: Vocab sizes for auxiliary inputs.
        output_size: Number of distribution parameters to output.
    """
    model_body: nn.Module
    d_emb: int
    output_size: int
    input_vocab_sizes: list[int] | None = None
    
    @nn.compact
    def __call__(
        self,
        context: jnp.ndarray,
        inputs: jnp.ndarray,
        deterministic: bool = False,
        decode: bool = False
    ) -> jnp.ndarray:
        """Forward pass.
        
        Args:
            context: Context values of shape (batch, seq_len).
            inputs: Auxiliary inputs of shape (batch, seq_len, n_inputs).
            deterministic: Whether to disable dropout.
            decode: Whether in decode mode.
        
        Returns:
            Distribution parameters of shape (batch, seq_len, output_size).
        """
        # Add embedding dimension to context
        # (batch, seq_len) -> (batch, seq_len, 1)
        context_emb = jnp.expand_dims(context, axis=-1)
        
        # Project to d_emb dimension
        context_emb = nn.Dense(self.d_emb)(context_emb)
        
        # Inject auxiliary inputs
        if self.input_vocab_sizes is not None:
            inject = InjectInputs(
                input_vocab_sizes=self.input_vocab_sizes,
                d_emb=self.d_emb,
                name='inject_inputs'
            )
            combined = inject(context_emb, inputs)
        else:
            combined = context_emb
        
        # Apply model body
        output = self.model_body(combined, deterministic=deterministic, decode=decode)
        
        # Project to distribution parameters
        params = nn.Dense(self.output_size, name='output_dense')(output)
        
        return params


def create_distribution_decoder(
    model_body: nn.Module,
    d_emb: int,
    input_vocab_sizes: list[int] | None,
    output_size: int,
    mode: str
) -> DistributionDecoderModel:
    """Create a distribution decoder model.
    
    Args:
        model_body: Transformer body model.
        d_emb: Embedding dimension.
        input_vocab_sizes: Vocab sizes for auxiliary inputs.
        output_size: Number of distribution parameters.
        mode: 'train', 'eval', or 'predict'.
    
    Returns:
        DistributionDecoderModel instance.
    """
    return DistributionDecoderModel(
        model_body=model_body,
        d_emb=d_emb,
        output_size=output_size,
        input_vocab_sizes=input_vocab_sizes,
    )


# =============================================================================
# Training wrapper
# =============================================================================

class DistributionTrainingModel(nn.Module):
    """Training wrapper that handles normalization.
    
    Attributes:
        decoder_model: The distribution decoder model.
        use_mask: Whether to use mask for normalization.
    """
    decoder_model: nn.Module
    use_mask: bool
    
    @nn.compact
    def __call__(
        self,
        series: jnp.ndarray,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        mask: jnp.ndarray,
        normalizer: Normalizer,
        deterministic: bool = False
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Forward pass for training.
        
        Args:
            series: Input series of shape (batch, seq_len).
            inputs: Auxiliary inputs of shape (batch, seq_len, n_inputs).
            targets: Target series of shape (batch, seq_len).
            mask: Mask of shape (batch, seq_len).
            normalizer: Normalizer instance.
            deterministic: Whether to disable dropout.
        
        Returns:
            Tuple of (distribution_params, normalized_targets, mask).
        """
        # Normalize
        mask_ = mask if self.use_mask else None
        norm_series, _, _ = normalizer.normalize(series, mask_)
        norm_targets, _, target_mask_mod = normalizer.normalize(targets, mask_)
        mask = mask * target_mask_mod
        
        # Forward through decoder
        params = self.decoder_model(
            norm_series,
            inputs,
            deterministic=deterministic,
            decode=False
        )
        
        return params, norm_targets, mask


# =============================================================================
# Main predictor class
# =============================================================================

@gin.configurable(module="code.predictors")
class DistributionPredictor(TimeSeriesPredictor):
    """Time series predictor using parametric distributions.
    
    Generates parameters for a distribution (e.g., mean and std for Gaussian)
    from which predictions are sampled.
    """

    def __init__(
        self,
        model_body_fn: Callable = gin.REQUIRED,
        accelerate_predict_model: bool = True,
        decoder_fn: Callable | None = None,
        d_in: int = 256,
        input_vocab_sizes: list[int] | None = None,
        normalization: str = "per_ts",
        normalization_regularizer: float = 1.0,
        distribution: str = "gaussian",
        learn_scale: str = "shared",  # "shared", "separate", or None for fixed scale
    ):
        """Initialize DistributionPredictor.

        Args:
            model_body_fn: Function returning the transformer body model.
            accelerate_predict_model: Whether to JIT the prediction model.
            decoder_fn: Optional custom decoder function.
            d_in: Depth of input embedding.
            input_vocab_sizes: Vocab sizes for auxiliary inputs.
            normalization: Normalization method name.
            normalization_regularizer: Regularization constant.
            distribution: Distribution type ('gaussian', 'laplace', 'cauchy', 'tstudent').
            learn_scale: How to learn scale - "shared" (1 param), "separate" 
                (per-dim params), or None (fixed scale).
        """
        self._distribution: distributions.Distribution = (
            distributions.Distribution.from_name(distribution, learn_scale=learn_scale)
        )
        self._d_in = d_in
        self.output_size = self._distribution.n_inputs

        if decoder_fn is None:
            decoder_fn = create_distribution_decoder
        
        self._decoder_fn_base = decoder_fn
        decoder_fn = partial(
            decoder_fn,
            d_emb=d_in,
            input_vocab_sizes=input_vocab_sizes,
            output_size=self.output_size,
        )

        super().__init__(
            model_body_fn=model_body_fn,
            accelerate_predict_model=accelerate_predict_model,
            normalization=normalization,
            normalization_regularizer=normalization_regularizer,
            context_type=np.float32,
            input_vocab_sizes=input_vocab_sizes,
            decoder_fn=decoder_fn,
        )

    def make_train_eval_model(self, mode: str) -> DistributionTrainingModel:
        """Create model for training or evaluation.
        
        Args:
            mode: 'train' or 'eval'.
        
        Returns:
            DistributionTrainingModel instance.
        """
        assert mode in ("train", "eval")
        use_mask = mode == "eval"
        
        # Create decoder model
        model_body = self._model_body_fn()
        decoder_model = self._decoder_fn_base(
            model_body=model_body,
            d_emb=self._d_in,
            input_vocab_sizes=self._input_vocab_sizes,
            output_size=self.output_size,
            mode=mode,
        )
        
        return DistributionTrainingModel(
            decoder_model=decoder_model,
            use_mask=use_mask,
        )

    def make_loss(self) -> Callable:
        """Create loss function.
        
        Returns:
            Negative log-likelihood loss function for the distribution.
        """
        distribution = self._distribution
        
        def loss_fn(
            params: jnp.ndarray,
            targets: jnp.ndarray,
            mask: jnp.ndarray
        ) -> jnp.ndarray:
            """Compute negative log-likelihood loss.
            
            Args:
                params: Distribution parameters of shape (batch, seq_len, n_params).
                targets: Target values of shape (batch, seq_len).
                mask: Sample weights/mask of shape (batch, seq_len).
            
            Returns:
                Scalar loss value.
            """
            # Compute log probability
            log_prob = distribution.log_prob(params, targets)
            
            # Negative log-likelihood with masking
            nll = -log_prob * mask
            
            return jnp.sum(nll) / (jnp.sum(mask) + 1e-8)
        
        return loss_fn

    def predict(
        self,
        weights: Any,
        context: np.ndarray,
        inputs: np.ndarray,
        horizon_length: int,
    ) -> np.ndarray:
        """Predict future values by sampling from the distribution.
        
        Args:
            weights: Model parameters.
            context: Past values of shape (batch_size, context_length).
            inputs: Auxiliary inputs of shape (batch, context + horizon, n_inputs).
            horizon_length: Number of steps to predict.
        
        Returns:
            Predictions of shape (batch_size, horizon_length).
        """
        batch_size, _ = context.shape
        
        # Initialize model state
        rng = jax.random.key(0)
        variables = self.init_state(batch_size, rng)
        
        # If weights provided, use them
        if weights is not None:
            if isinstance(weights, dict) and 'params' in weights:
                params = weights['params']
            else:
                params = weights
            variables = {'params': params, **{k: v for k, v in variables.items() if k != 'params'}}
        
        # Normalize context
        norm_context, scaling_params, _ = self._normalizer.normalize(context)
        
        # Sample function using the distribution
        def sample_fn(dist_params: jnp.ndarray, rng: jax.Array | None = None) -> jnp.ndarray:
            """Sample from the distribution."""
            return self._distribution.sample(dist_params, rng=rng)
        
        # Autoregressive sampling
        norm_pred = decoding.autoregressive_sample(
            model=self.predict_model,
            variables=variables,
            sample_fn=sample_fn,
            context=norm_context,
            inputs=inputs,
            horizon_length=horizon_length,
            rng=rng,
        )
        
        # Denormalize
        norm_series = jnp.concatenate([norm_context, norm_pred], axis=-1)
        pred = self._normalizer.denormalize(norm_series, scaling_params)
        
        return np.array(pred[:, -norm_pred.shape[-1]:])

    @property
    def distribution(self) -> distributions.Distribution:
        """Get the distribution used by this predictor."""
        return self._distribution
