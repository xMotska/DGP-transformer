"""Serial predictor for time series using discretization."""

from functools import partial
from typing import Any, Callable

import gin
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

import decoding
import distributions
import serializers
from input_injection import InjectInputs
from metrics import WeightedSmoothedCategoryCrossEntropy
from normalization import Normalizer
from time_series_predictor import TimeSeriesPredictor


# =============================================================================
# Serialization utilities (replacing trax.rl.serialization_utils)
# =============================================================================

def serialize(data: np.ndarray, serializer) -> np.ndarray:
    """Serialize data using the given serializer.
    
    Args:
        data: Array to serialize.
        serializer: Serializer instance.
    
    Returns:
        Serialized representation.
    """
    return serializer.serialize(data)


def representation_mask(mask: jnp.ndarray, serializer) -> jnp.ndarray:
    """Expand mask to match serialized representation length.
    
    Args:
        mask: Original mask of shape (batch, seq_len).
        serializer: Serializer instance.
    
    Returns:
        Expanded mask of shape (batch, seq_len * repr_len).
    """
    repr_len = serializer.representation_length
    # Repeat each mask value repr_len times
    expanded = jnp.repeat(mask, repeats=repr_len, axis=-1)
    return expanded


def significance_weights(
    mask: jnp.ndarray,
    serializer,
    decay: float
) -> jnp.ndarray:
    """Compute significance weights based on digit position.
    
    More significant digits (earlier in representation) get higher weights.
    
    Args:
        mask: Mask of shape (batch, seq_len).
        serializer: Serializer instance.
        decay: Decay factor for exponential weighting.
    
    Returns:
        Weights of shape (batch, seq_len * repr_len).
    """
    repr_len = serializer.representation_length
    batch_size, seq_len = mask.shape
    
    # Create significance weights for each digit position
    # significance_map gives digit index (0 = most significant)
    sig_map = serializer.significance_map  # shape (repr_len,)
    
    # Compute weights: decay^significance
    digit_weights = decay ** sig_map  # shape (repr_len,)
    
    # Expand mask to representation length
    repr_mask = representation_mask(mask, serializer)  # (batch, seq_len * repr_len)
    
    # Tile digit weights across all timesteps
    tiled_weights = jnp.tile(digit_weights, seq_len)  # (seq_len * repr_len,)
    
    # Apply mask
    weights = repr_mask * tiled_weights
    
    return weights


def upsample_inputs(inputs: jnp.ndarray, repr_len: int) -> jnp.ndarray:
    """Upsample inputs to match serialized sequence length.
    
    Args:
        inputs: Input array of shape (batch, seq_len, n_inputs).
        repr_len: Representation length per timestep.
    
    Returns:
        Upsampled inputs of shape (batch, seq_len * repr_len, n_inputs).
    """
    return jnp.repeat(inputs, repeats=repr_len, axis=1)


# =============================================================================
# Loss functions
# =============================================================================

def weighted_category_cross_entropy(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    weights: jnp.ndarray
) -> jnp.ndarray:
    """Weighted categorical cross-entropy loss.
    
    Args:
        logits: Predicted logits of shape (..., vocab_size).
        targets: Target indices of shape (...).
        weights: Sample weights of shape (...).
    
    Returns:
        Scalar loss value.
    """
    # One-hot encode targets
    n_classes = logits.shape[-1]
    targets_onehot = jax.nn.one_hot(targets, n_classes)
    
    # Compute log softmax
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    
    # Compute cross-entropy
    cross_entropy = -jnp.sum(targets_onehot * log_probs, axis=-1)
    
    # Apply weights
    weighted_loss = cross_entropy * weights
    
    return jnp.sum(weighted_loss) / (jnp.sum(weights) + 1e-8)


# =============================================================================
# Decoder model
# =============================================================================

class SerialDecoderModel(nn.Module):
    """Decoder model that adds embedding and output layers to a model body.
    
    Attributes:
        model_body: The transformer body model.
        vocab_size: Size of the discrete vocabulary.
        d_emb: Embedding dimension.
        input_vocab_sizes: Vocab sizes for auxiliary inputs.
    """
    model_body: nn.Module
    vocab_size: int
    d_emb: int
    input_vocab_sizes: list[int] | None = None
    
    @nn.compact
    def __call__(
        self,
        context_repr: jnp.ndarray,
        inputs: jnp.ndarray,
        deterministic: bool = False,
        decode: bool = False
    ) -> jnp.ndarray:
        """Forward pass.
        
        Args:
            context_repr: Serialized context of shape (batch, seq_len).
            inputs: Auxiliary inputs of shape (batch, seq_len, n_inputs).
            deterministic: Whether to disable dropout.
            decode: Whether in decode mode.
        
        Returns:
            Logits of shape (batch, seq_len, vocab_size).
        """
        # Embed the serialized tokens
        context_emb = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.d_emb,
            name='token_embed'
        )(context_repr.astype(jnp.int32))
        
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
        
        # Project to vocab logits
        logits = nn.Dense(self.vocab_size, name='output_dense')(output)
        
        return logits


def create_serial_decoder(
    model_body: nn.Module,
    serializer,
    d_emb: int,
    input_vocab_sizes: list[int] | None,
    mode: str
) -> SerialDecoderModel:
    """Create a serial decoder model.
    
    Args:
        model_body: Transformer body model.
        serializer: Serializer instance.
        d_emb: Embedding dimension.
        input_vocab_sizes: Vocab sizes for auxiliary inputs.
        mode: 'train', 'eval', or 'predict'.
    
    Returns:
        SerialDecoderModel instance.
    """
    return SerialDecoderModel(
        model_body=model_body,
        vocab_size=serializer.vocab_size,
        d_emb=d_emb,
        input_vocab_sizes=input_vocab_sizes,
    )


# =============================================================================
# Training wrapper
# =============================================================================

class SerialTrainingModel(nn.Module):
    """Training wrapper that handles serialization and normalization.
    
    Attributes:
        decoder_model: The decoder model.
        serializer: Serializer instance.
        significance_decay: Decay factor for significance weights.
        normalizer: Normalizer instance.
        use_mask: Whether to use mask for normalization.
    """
    decoder_model: nn.Module
    vocab_size: int
    repr_len: int
    significance_decay: float
    use_mask: bool
    
    def setup(self):
        # Store serializer info needed for forward pass
        pass
    
    @nn.compact
    def __call__(
        self,
        series: jnp.ndarray,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        mask: jnp.ndarray,
        serializer,
        normalizer: Normalizer,
        deterministic: bool = False
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Forward pass for training.
        
        Args:
            series: Input series of shape (batch, seq_len).
            inputs: Auxiliary inputs of shape (batch, seq_len, n_inputs).
            targets: Target series of shape (batch, seq_len).
            mask: Mask of shape (batch, seq_len).
            serializer: Serializer instance.
            normalizer: Normalizer instance.
            deterministic: Whether to disable dropout.
        
        Returns:
            Tuple of (logits, target_repr, weights).
        """
        # Normalize
        mask_ = mask if self.use_mask else None
        norm_series, _, _ = normalizer.normalize(series, mask_)
        norm_targets, _, target_mask_mod = normalizer.normalize(targets, mask_)
        mask = mask * target_mask_mod
        
        # Serialize (note: serialize expects numpy arrays, so we convert outside gradient path)
        # The serialization is a non-differentiable operation anyway
        context_repr = serializer.serialize(jnp.asarray(norm_series))
        target_repr = serializer.serialize(jnp.asarray(norm_targets))
        
        # Flatten serialized representations
        context_repr = jnp.array(context_repr).reshape(context_repr.shape[0], -1)
        target_repr = jnp.array(target_repr).reshape(target_repr.shape[0], -1)
        
        # Upsample inputs
        inputs_up = upsample_inputs(inputs, self.repr_len)
        
        # Forward through decoder
        logits = self.decoder_model(
            context_repr,
            inputs_up,
            deterministic=deterministic,
            decode=False
        )
        
        # Compute significance weights
        weights = significance_weights(mask, serializer, self.significance_decay)
        weights = weights.reshape(weights.shape[0], -1)
        
        return logits, target_repr, weights


# =============================================================================
# Main predictor class
# =============================================================================

@gin.configurable(module="code.predictors")
class SerialPredictor(TimeSeriesPredictor):
    """Time series predictor based on serialization.
    
    Turns time series into sequences of discrete symbols that can be
    modeled using transformer language models.
    """

    def __init__(
        self,
        model_body_fn: Callable = gin.REQUIRED,
        decoder_fn: Callable | None = None,
        d_in: int = 256,
        vocab_size: int = 64,
        precision: int = 2,
        significance_decay: float = 0.7,
        low: float = 0.0,
        high: float = 1.0,
        accelerate_predict_model: bool = True,
        input_vocab_sizes: list[int] | None = None,
        normalization: str = "per_ts",
        normalization_regularizer: float = 1.0,
        label_smoothing: float | None = None,
        first_digit_mode: str = "uniform",
        clip_or_squash: str = "clip",
    ):
        """Initialize SerialPredictor.

        Args:
            model_body_fn: Function returning the transformer body model.
            decoder_fn: Optional custom decoder function. If None, uses default.
            d_in: Depth of the symbol embedding.
            vocab_size: Vocabulary size (number of distinct symbols).
            precision: Number of symbols to encode each float.
            significance_decay: Decay factor for exponential weighting.
            low: Minimum representable value.
            high: Maximum representable value.
            accelerate_predict_model: Whether to JIT the prediction model.
            input_vocab_sizes: Vocab sizes for auxiliary inputs.
            normalization: Normalization method name.
            normalization_regularizer: Regularization constant.
            label_smoothing: If set, smooth labels with this std.
            first_digit_mode: 'uniform' or 'quantile' encoding.
            clip_or_squash: Serialization strategy.
        """
        self._serializer = serializers.BoxSpaceSerializer(
            space=gym.spaces.Box(shape=(), low=low, high=high, dtype=np.float32),
            vocab_size=vocab_size,
            precision=precision,
            first_digit_mode=first_digit_mode,
            clip_or_squash=clip_or_squash,
        )

        if decoder_fn is None:
            decoder_fn = create_serial_decoder
        
        self._decoder_fn_base = decoder_fn
        decoder_fn = partial(
            decoder_fn,
            serializer=self._serializer,
            d_emb=d_in,
            input_vocab_sizes=input_vocab_sizes,
        )

        self._d_in = d_in
        self._precision = precision
        self._vocab_size = vocab_size
        self._significance_decay = significance_decay
        self._label_smoothing = label_smoothing
        self._categorical = distributions.Categorical(n_categories=vocab_size)
        self._input_vocab_sizes = input_vocab_sizes

        super().__init__(
            model_body_fn=partial(model_body_fn, precision=precision),
            accelerate_predict_model=accelerate_predict_model,
            normalization=normalization,
            normalization_regularizer=normalization_regularizer,
            context_type=np.int32,
            input_vocab_sizes=input_vocab_sizes,
            decoder_fn=decoder_fn,
        )

    def make_train_eval_model(self, mode: str) -> SerialTrainingModel:
        """Create model for training or evaluation.
        
        Args:
            mode: 'train' or 'eval'.
        
        Returns:
            SerialTrainingModel instance.
        """
        assert mode in ("train", "eval")
        use_mask = mode == "eval"  # In eval, normalize only based on 'seen' part
        
        # Create decoder model
        model_body = self._model_body_fn()
        decoder_model = self._decoder_fn_base(
            model_body=model_body,
            serializer=self._serializer,
            d_emb=self._d_in,
            input_vocab_sizes=self._input_vocab_sizes,
            mode=mode,
        )
        
        return SerialTrainingModel(
            decoder_model=decoder_model,
            vocab_size=self._vocab_size,
            repr_len=self._serializer.representation_length,
            significance_decay=self._significance_decay,
            use_mask=use_mask,
        )

    def make_loss(self) -> Callable:
        """Create loss function.
        
        Returns:
            Loss function with signature (logits, targets, weights) -> scalar.
        """
        if self._label_smoothing is None:
            return weighted_category_cross_entropy
        return WeightedSmoothedCategoryCrossEntropy(
            self._label_smoothing, self._precision
        )

    def before_training(self, inputs: Callable) -> None:
        """Fit serializer on training data.
        
        Args:
            inputs: Callable returning Inputs object.
        """
        def normalize_one(inp):
            series, _, _, _ = inp
            return self._normalizer.normalize(series)[0]

        # Get training stream and normalize for serializer fitting
        input_stream = map(normalize_one, inputs().train_batches())
        self._serializer.fit(input_stream)

    def predict(
        self,
        weights: Any,
        context: np.ndarray,
        inputs: np.ndarray,
        horizon_length: int,
    ) -> np.ndarray:
        """Predict future values autoregressively.
        
        Args:
            weights: Model parameters (Flax params dict).
            context: Past values of shape (batch_size, context_length).
            inputs: Auxiliary inputs of shape (batch, context + horizon, n_inputs).
            horizon_length: Number of steps to predict.
        
        Returns:
            Predictions of shape (batch_size, horizon_length).
        """
        batch_size, context_length = context.shape
        
        # Initialize model state
        rng = jax.random.key(0)
        variables = self.init_state(batch_size, rng)
        
        # If weights provided, use them (they may be from training model)
        if weights is not None:
            # Extract decoder params if nested
            if isinstance(weights, dict) and 'params' in weights:
                params = weights['params']
            else:
                params = weights
            variables = {'params': params, **{k: v for k, v in variables.items() if k != 'params'}}
        
        # Normalize context
        norm_context, scaling_params, _ = self._normalizer.normalize(context)
        
        # Serialize context
        context_repr = self._serializer.serialize(norm_context)
        repr_len = self._serializer.representation_length
        
        # context_repr shape: (batch, context_length, repr_len) -> flatten to (batch, context_length * repr_len)
        context_repr = context_repr.reshape(batch_size, -1)
        context_repr_len = context_repr.shape[1]
        
        # The autoregressive decoder expects inputs for context + horizon
        # inputs shape: (batch, seq_len, n_inputs)
        # We need to upsample to (batch, (context_length + horizon_length) * repr_len, n_inputs)
        horizon_repr_len = repr_len * horizon_length
        total_repr_len = context_repr_len + horizon_repr_len
        
        # Slice inputs to context + horizon length first
        inputs_needed = inputs[:, :context_length + horizon_length, :]
        
        # Upsample inputs to match serialized length
        inputs_upsampled = np.repeat(inputs_needed, repeats=repr_len, axis=1)
        
        # Autoregressive sampling
        pred_repr = decoding.autoregressive_sample(
            model=self.predict_model,
            variables=variables,
            sample_fn=self._categorical.sample,
            context=context_repr,
            inputs=inputs_upsampled,
            batch_size=batch_size,
            horizon_length=horizon_repr_len,
            rng=rng,
        )
        
        # Deserialize predictions
        pred_repr = np.reshape(pred_repr, (-1, repr_len))
        norm_pred = self._serializer.deserialize(pred_repr)
        norm_pred = np.reshape(norm_pred, (batch_size, -1))
        
        # Denormalize
        norm_series = jnp.concatenate([norm_context, norm_pred], axis=-1)
        pred = self._normalizer.denormalize(norm_series, scaling_params)
        
        # Return only the predicted horizon
        return np.array(pred[:, -norm_pred.shape[-1]:])

    @property
    def serializer(self):
        """Get the serializer."""
        return self._serializer
