"""IQN (Implicit Quantile Network) predictor for time series."""

from functools import partial
from typing import Any, Callable

import gin
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

import decoding
from input_injection import InjectInputs
from normalization import Normalizer
from time_series_predictor import TimeSeriesPredictor


# =============================================================================
# Quantile layers
# =============================================================================

class QuantileLayer(nn.Module):
    """Embed quantile tau using cosine basis functions.
    
    Attributes:
        d_emb: Output embedding dimension.
        n_cos_embedding: Number of cosine basis functions.
    """
    d_emb: int
    n_cos_embedding: int = 64
    
    @nn.compact
    def __call__(self, tau: jnp.ndarray) -> jnp.ndarray:
        """Embed quantile values.
        
        Args:
            tau: Quantile values of shape (batch, seq_len).
        
        Returns:
            Embeddings of shape (batch, seq_len, d_emb).
        """
        # Create integer indices for cosine embedding
        # shape: (n_cos_embedding,)
        indices = jnp.arange(self.n_cos_embedding)
        
        # Compute cosine embedding
        # tau: (batch, seq_len) -> (batch, seq_len, 1)
        tau_expanded = jnp.expand_dims(tau, axis=-1)
        # cos(pi * tau * i) for i in 0..n_cos_embedding
        cos_emb = jnp.cos(jnp.pi * tau_expanded * indices)
        # shape: (batch, seq_len, n_cos_embedding)
        
        # Transform through MLP
        x = nn.Dense(self.n_cos_embedding)(cos_emb)
        x = nn.leaky_relu(x, negative_slope=0.1)
        x = nn.Dense(self.d_emb)(x)
        
        return x


class ImplicitQuantileModule(nn.Module):
    """Compute quantile value from embedding and sampled tau.
    
    Takes model output embedding and returns (quantile_value, tau).
    
    Attributes:
        d_emb: Embedding dimension.
    """
    d_emb: int
    
    @nn.compact
    def __call__(
        self, 
        model_output: jnp.ndarray,
        tau: jnp.ndarray | None = None,
        rng: jax.Array | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute quantile values.
        
        Args:
            model_output: Model embeddings of shape (batch, seq_len, d_emb).
            tau: Optional pre-specified quantiles of shape (batch, seq_len).
                If None, samples uniformly from [0, 1].
            rng: Random key for sampling tau.
        
        Returns:
            Tuple of (quantile_values, tau) with shapes 
            (batch, seq_len) and (batch, seq_len).
        """
        # Sample tau if not provided
        if tau is None:
            if rng is None:
                # Use numpy for sampling (non-JIT compatible fallback)
                tau = np.random.uniform(size=model_output.shape[:-1]).astype(np.float32)
                tau = jnp.array(tau)
            else:
                tau = jax.random.uniform(rng, shape=model_output.shape[:-1])
        
        # Embed tau
        tau_emb = QuantileLayer(d_emb=self.d_emb)(tau)
        
        # Combine embeddings: output * (1 + tau_emb)
        combined = model_output * (1.0 + tau_emb)
        
        # Output layer: Dense -> Softplus -> Dense -> squeeze
        x = nn.Dense(self.d_emb)(combined)
        x = nn.softplus(x)
        x = nn.Dense(1)(x)
        quantile_value = jnp.squeeze(x, axis=-1)
        
        return quantile_value, tau


def create_implicit_quantile_module(d_emb: int) -> ImplicitQuantileModule:
    """Factory function for ImplicitQuantileModule."""
    return ImplicitQuantileModule(d_emb=d_emb)


# =============================================================================
# Loss function
# =============================================================================

def quantile_loss(
    quantile_forecast: jnp.ndarray,
    tau: jnp.ndarray,
    target: jnp.ndarray,
    weights: jnp.ndarray | None = None
) -> jnp.ndarray:
    """Compute quantile loss (pinball loss).
    
    Args:
        quantile_forecast: Predicted quantile values of shape (batch, seq_len).
        tau: Quantile levels of shape (batch, seq_len).
        target: Target values of shape (batch, seq_len).
        weights: Optional sample weights of shape (batch, seq_len).
    
    Returns:
        Scalar loss value.
    """
    # Pinball loss
    error = target - quantile_forecast
    loss = jnp.where(
        error >= 0,
        tau * error,
        (tau - 1) * error
    )
    
    # Equivalent formulation:
    # loss = jnp.abs(error * ((target <= quantile_forecast).astype(jnp.float32) - tau))
    
    if weights is not None:
        weighted_loss = loss * weights
        return jnp.sum(weighted_loss) / (jnp.sum(weights) + 1e-8)
    else:
        return jnp.mean(loss)


class QuantileLoss(nn.Module):
    """Quantile loss layer for use in training."""
    
    @nn.compact
    def __call__(
        self,
        quantile_forecast: jnp.ndarray,
        tau: jnp.ndarray,
        target: jnp.ndarray,
        weights: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        return quantile_loss(quantile_forecast, tau, target, weights)


# =============================================================================
# Decoder model
# =============================================================================

class IQNDecoderModel(nn.Module):
    """IQN decoder that adds input embedding and processes through model body.
    
    Attributes:
        model_body: The transformer body model.
        d_emb: Embedding dimension.
        input_vocab_sizes: Vocab sizes for auxiliary inputs.
    """
    model_body: nn.Module
    d_emb: int
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
            Model output embeddings of shape (batch, seq_len, d_emb).
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
        
        return output


def create_iqn_decoder(
    model_body: nn.Module,
    d_emb: int,
    input_vocab_sizes: list[int] | None,
    mode: str,
    iqm: Any = None  # Not used, kept for API compatibility
) -> IQNDecoderModel:
    """Create an IQN decoder model.
    
    Args:
        model_body: Transformer body model.
        d_emb: Embedding dimension.
        input_vocab_sizes: Vocab sizes for auxiliary inputs.
        mode: 'train', 'eval', or 'predict'.
        iqm: Unused, kept for compatibility.
    
    Returns:
        IQNDecoderModel instance.
    """
    return IQNDecoderModel(
        model_body=model_body,
        d_emb=d_emb,
        input_vocab_sizes=input_vocab_sizes,
    )


# =============================================================================
# Training wrapper
# =============================================================================

class IQNTrainingModel(nn.Module):
    """Training wrapper that handles normalization and IQM.
    
    Attributes:
        decoder_model: The IQN decoder model.
        d_emb: Embedding dimension.
        use_mask: Whether to use mask for normalization.
    """
    decoder_model: nn.Module
    d_emb: int
    use_mask: bool
    
    @nn.compact
    def __call__(
        self,
        series: jnp.ndarray,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        mask: jnp.ndarray,
        normalizer: Normalizer,
        deterministic: bool = False,
        rng: jax.Array | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Forward pass for training.
        
        Args:
            series: Input series of shape (batch, seq_len).
            inputs: Auxiliary inputs of shape (batch, seq_len, n_inputs).
            targets: Target series of shape (batch, seq_len).
            mask: Mask of shape (batch, seq_len).
            normalizer: Normalizer instance.
            deterministic: Whether to disable dropout.
            rng: Random key for tau sampling.
        
        Returns:
            Tuple of (quantile_forecast, tau, target, mask).
        """
        # Normalize
        mask_ = mask if self.use_mask else None
        norm_series, _, _ = normalizer.normalize(series, mask_)
        norm_targets, _, target_mask_mod = normalizer.normalize(targets, mask_)
        mask = mask * target_mask_mod
        
        # Forward through decoder
        output = self.decoder_model(
            norm_series,
            inputs,
            deterministic=deterministic,
            decode=False
        )
        
        # Apply IQM
        iqm = ImplicitQuantileModule(d_emb=self.d_emb)
        quantile_forecast, tau = iqm(output, rng=rng)
        
        return quantile_forecast, tau, norm_targets, mask


# =============================================================================
# Main predictor class
# =============================================================================

@gin.configurable(module="code.predictors")
class IQNPredictor(TimeSeriesPredictor):
    """Time series predictor using Implicit Quantile Networks.
    
    Based on 'Probabilistic Time Series Forecasting with Implicit Quantile Networks'.
    Generates quantile predictions for probabilistic forecasting.
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
    ):
        """Initialize IQNPredictor.

        Args:
            model_body_fn: Function returning the transformer body model.
            accelerate_predict_model: Whether to JIT the prediction model.
            decoder_fn: Optional custom decoder function.
            d_in: Depth of input embedding.
            input_vocab_sizes: Vocab sizes for auxiliary inputs.
            normalization: Normalization method name.
            normalization_regularizer: Regularization constant.
        """
        self._d_in = d_in
        self._iqm = None
        
        if decoder_fn is None:
            decoder_fn = create_iqn_decoder
        
        self._decoder_fn_base = decoder_fn
        decoder_fn = partial(
            decoder_fn,
            d_emb=d_in,
            input_vocab_sizes=input_vocab_sizes,
            iqm=None
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

    def make_train_eval_model(self, mode: str) -> IQNTrainingModel:
        """Create model for training or evaluation.
        
        Args:
            mode: 'train' or 'eval'.
        
        Returns:
            IQNTrainingModel instance.
        """
        assert mode in ("train", "eval")
        use_mask = mode == "eval"
        
        # Create decoder model
        model_body = self._model_body_fn()
        decoder_model = self._decoder_fn_base(
            model_body=model_body,
            d_emb=self._d_in,
            input_vocab_sizes=self._input_vocab_sizes,
            mode=mode,
            iqm=None,
        )
        
        # Store IQM for prediction
        self._iqm = ImplicitQuantileModule(d_emb=self._d_in)
        
        return IQNTrainingModel(
            decoder_model=decoder_model,
            d_emb=self._d_in,
            use_mask=use_mask,
        )

    def make_loss(self) -> Callable:
        """Create loss function.
        
        Returns:
            Quantile loss function.
        """
        def loss_fn(quantile_forecast, tau, target, mask):
            return quantile_loss(quantile_forecast, tau, target, mask)
        return loss_fn

    def predict(
        self,
        weights: Any,
        context: np.ndarray,
        inputs: np.ndarray,
        horizon_length: int,
    ) -> np.ndarray:
        """Predict future values autoregressively using median (tau=0.5).
        
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
        
        # Create IQM for sampling
        iqm = ImplicitQuantileModule(d_emb=self._d_in)
        
        # Sample function: apply IQM with median tau=0.5
        def sample_fn(model_output: jnp.ndarray, rng: jax.Array | None = None) -> jnp.ndarray:
            """Sample next value using median quantile."""
            # Add time dimension if needed
            if model_output.ndim == 2:
                model_output = jnp.expand_dims(model_output, axis=1)
            
            # Use tau=0.5 for median prediction
            tau = jnp.full(model_output.shape[:-1], 0.5)
            
            # Initialize IQM variables
            iqm_vars = iqm.init(rng, model_output, tau)
            quantile_value, _ = iqm.apply(iqm_vars, model_output, tau)
            
            # Remove time dimension
            if quantile_value.ndim == 2 and quantile_value.shape[1] == 1:
                quantile_value = jnp.squeeze(quantile_value, axis=1)
            
            return quantile_value
        
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

    def predict_quantiles(
        self,
        weights: Any,
        context: np.ndarray,
        inputs: np.ndarray,
        horizon_length: int,
        quantiles: list[float] = [0.1, 0.5, 0.9],
    ) -> dict[float, np.ndarray]:
        """Predict multiple quantiles.
        
        Args:
            weights: Model parameters.
            context: Past values of shape (batch_size, context_length).
            inputs: Auxiliary inputs.
            horizon_length: Number of steps to predict.
            quantiles: List of quantile levels to predict.
        
        Returns:
            Dictionary mapping quantile level to predictions.
        """
        results = {}
        for q in quantiles:
            # TODO: Implement proper quantile-specific prediction
            # For now, use median
            results[q] = self.predict(weights, context, inputs, horizon_length)
        return results
