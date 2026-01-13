"""Custom layers for time series models using Flax."""

import math
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn


class CausalConv(nn.Module):
    """Causal (masked) convolution for [batch, time, depth] sequences.
    
    Maintains causality along time axis. Used in language modeling tasks.
    
    Attributes:
        features: Number of output features (filters).
        kernel_size: Width of the convolution kernel.
        use_bias: Whether to add a bias term.
        kernel_init: Initializer for kernel weights.
        bias_init: Initializer for bias.
        dtype: Data type for computation.
    """
    features: int
    kernel_size: int = 3
    use_bias: bool = True
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.normal(stddev=1e-6)
    dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, decode: bool = False) -> jnp.ndarray:
        """Apply causal convolution.
        
        Args:
            x: Input tensor of shape [batch, time, depth].
            decode: If True, use cached state for autoregressive decoding.
        
        Returns:
            Output tensor of shape [batch, time, features].
        """
        if decode:
            # Autoregressive decoding with cached state
            is_initialized = self.has_variable('cache', 'cached_input')
            cached_input = self.variable(
                'cache', 'cached_input',
                lambda: jnp.zeros((x.shape[0], self.kernel_size - 1, x.shape[-1]), 
                                  dtype=self.dtype)
            )
            
            if is_initialized:
                # Concatenate cached input with current input
                x_extended = jnp.concatenate([cached_input.value, x], axis=1)
                # Update cache with most recent (kernel_size - 1) timesteps
                cached_input.value = x_extended[:, -(self.kernel_size - 1):, :]
                # Use the extended input for convolution
                x = x_extended[:, -(self.kernel_size + x.shape[1] - 1):, :]
            else:
                # First call: initialize cache
                pad = self.kernel_size - 1
                x_padded = jnp.pad(x, [[0, 0], [pad, 0], [0, 0]], mode='constant')
                cached_input.value = x_padded[:, -(self.kernel_size - 1):, :]
                x = x_padded
        else:
            # Training/eval mode: left-pad for causal convolution
            pad = self.kernel_size - 1
            x = jnp.pad(x, [[0, 0], [pad, 0], [0, 0]], mode='constant')
        
        # Apply 1D convolution with VALID padding (we already handled causal padding)
        return nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,),
            padding='VALID',
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
        )(x)


class DigitEncoding(nn.Module):
    """Digit-based positional encoding.
    
    Encodes positions using their value modulo a precision parameter,
    allowing the model to learn patterns that repeat at fixed intervals.
    
    Attributes:
        max_len: Maximum sequence length.
        precision: The modular base for digit encoding.
        dropout_rate: Dropout probability for training.
        dropout_broadcast_dims: Axes for broadcasting dropout mask.
        use_bfloat16: Whether to use bfloat16 for embeddings.
        embedding_init: Initializer for embedding weights.
    """
    max_len: int = 2048
    precision: int = 10
    dropout_rate: float = 0.0
    dropout_broadcast_dims: Sequence[int] = (-2,)
    use_bfloat16: bool = False
    embedding_init: nn.initializers.Initializer = nn.initializers.glorot_uniform()
    
    @nn.compact
    def __call__(
        self, 
        x: jnp.ndarray, 
        deterministic: bool = False,
        decode: bool = False
    ) -> jnp.ndarray:
        """Add digit-based positional encoding to input.
        
        Args:
            x: Input tensor of shape [batch, seq_len, d_feature].
            deterministic: If True, disable dropout.
            decode: If True, use cached position for autoregressive decoding.
        
        Returns:
            Input with positional encoding added.
        """
        d_feature = x.shape[-1]
        dtype = jnp.bfloat16 if self.use_bfloat16 else jnp.float32
        
        # Initialize embedding weights: [precision, d_feature]
        weights = self.param(
            'embedding',
            self.embedding_init,
            (self.precision, d_feature)
        )
        weights = weights.astype(dtype)
        
        if decode:
            # Autoregressive decoding
            position = self.variable('cache', 'position',
                                     lambda: jnp.array(0, dtype=jnp.int32))
            token_ids = jnp.arange(x.shape[1]) + position.value
            position.value = position.value + x.shape[1]
        else:
            token_ids = jnp.arange(x.shape[1])
        
        # Get digit ids (position mod precision)
        digit_ids = jnp.mod(token_ids, self.precision)
        
        # Look up embeddings: [seq_len, d_feature]
        emb = weights[digit_ids]
        # Broadcast to batch: [1, seq_len, d_feature]
        emb = emb[None, :, :]
        
        # Apply dropout during training
        if not deterministic and self.dropout_rate > 0.0:
            rng = self.make_rng('dropout')
            noise_shape = list(emb.shape)
            for dim in self.dropout_broadcast_dims:
                noise_shape[dim] = 1
            keep_prob = 1.0 - self.dropout_rate
            mask = jax.random.bernoulli(rng, keep_prob, tuple(noise_shape))
            emb = emb * mask.astype(x.dtype) / keep_prob
        
        return x + emb.astype(x.dtype)


class PositionalDigitEncoding(nn.Module):
    """Combined sinusoidal positional encoding with optional digit encoding.
    
    Combines standard sinusoidal positional encodings with digit-based
    encodings for capturing periodic patterns.
    
    Attributes:
        max_len: Maximum sequence length.
        d_feature: Output feature dimension (if projecting).
        d_digit: Dimension for digit encoding (0 to disable).
        precision: Modular base for digit encoding.
        dropout_rate: Dropout probability for training.
        dropout_broadcast_dims: Axes for broadcasting dropout mask.
        use_bfloat16: Whether to use bfloat16.
        start_from_zero_prob: Probability of starting from position 0 during training.
        max_offset_to_add: Maximum random offset during training.
    """
    max_len: int = 2048
    d_feature: int | None = None
    d_digit: int = 0
    precision: int | None = None
    dropout_rate: float = 0.0
    dropout_broadcast_dims: Sequence[int] = (-2,)
    use_bfloat16: bool = False
    start_from_zero_prob: float = 1.0
    max_offset_to_add: int = 0
    
    def setup(self):
        """Initialize positional encoding matrix."""
        if self.d_digit > 0 and self.precision is None:
            raise ValueError('precision must be specified when d_digit > 0')
    
    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = False,
        decode: bool = False
    ) -> jnp.ndarray:
        """Add positional encoding to input.
        
        Args:
            x: Input tensor of shape [batch, seq_len, d_feature].
            deterministic: If True, disable dropout and random offsets.
            decode: If True, use cached position for autoregressive decoding.
        
        Returns:
            Input with positional encoding added.
        """
        d_input = x.shape[-1]
        d_feature = self.d_feature if self.d_feature is not None else d_input
        d_position = d_feature - self.d_digit
        
        # Build sinusoidal positional encoding
        pe = self._build_positional_encoding(d_position)
        
        # Add digit encoding if specified
        if self.d_digit > 0 and self.precision is not None:
            pe_digit = self._build_digit_encoding()
            pe = jnp.concatenate([pe_digit, pe], axis=-1)
        
        if self.use_bfloat16:
            pe = pe.astype(jnp.bfloat16)
        
        # Store as a parameter (trainable)
        weights = self.param('positional_encoding', lambda rng: pe)
        
        # Optional projection if d_feature != d_input
        if self.d_feature is not None and self.d_feature != d_input:
            ff = self.param(
                'projection',
                nn.initializers.glorot_uniform(),
                (d_feature, d_input)
            )
        else:
            ff = None
        
        if decode:
            # Autoregressive decoding
            position = self.variable('cache', 'position',
                                     lambda: jnp.array(0, dtype=jnp.int32))
            emb = jax.lax.dynamic_slice_in_dim(
                weights, position.value, x.shape[1], axis=0
            )
            position.value = position.value + x.shape[1]
            
            if ff is not None:
                emb = jnp.dot(emb, ff)
            
            return x + emb[None, :, :].astype(x.dtype)
        
        # Training/eval mode
        seq_len = x.shape[1]
        
        if deterministic or self.start_from_zero_prob >= 1.0:
            # Always start from position 0
            start = 0
        else:
            # Random offset during training
            rng1, rng2 = jax.random.split(self.make_rng('dropout'))
            start = jax.random.randint(rng1, (), 0, self.max_offset_to_add + 1)
            start_from_zero = jax.random.uniform(rng2, ())
            start = jnp.where(
                start_from_zero < self.start_from_zero_prob,
                jnp.array(0, dtype=jnp.int32),
                start
            )
        
        emb = jax.lax.dynamic_slice_in_dim(weights, start, seq_len, axis=0)
        
        if ff is not None:
            emb = jnp.dot(emb, ff)
        
        emb = emb[None, :, :]  # [1, seq_len, d_feature]
        
        # Apply dropout
        if not deterministic and self.dropout_rate > 0.0:
            rng = self.make_rng('dropout')
            noise_shape = list(emb.shape)
            for dim in self.dropout_broadcast_dims:
                noise_shape[dim] = 1
            keep_prob = 1.0 - self.dropout_rate
            mask = jax.random.bernoulli(rng, keep_prob, tuple(noise_shape))
            emb = emb * mask.astype(x.dtype) / keep_prob
        
        return x + emb.astype(x.dtype)
    
    def _build_positional_encoding(self, d_position: int) -> jnp.ndarray:
        """Build sinusoidal positional encoding matrix."""
        pe = np.zeros((self.max_len, d_position), dtype=np.float32)
        position = np.arange(self.max_len)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, d_position, 2) * -(np.log(10000.0) / d_position)
        )
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return jnp.array(pe)
    
    def _build_digit_encoding(self) -> jnp.ndarray:
        """Build digit-based positional encoding matrix."""
        d_digit = self.d_digit
        precision = self.precision
        pe_digit = np.zeros((self.max_len, d_digit), dtype=np.float32)
        position = np.arange(self.max_len)[:, np.newaxis]
        div_term_digit = 2 ** np.arange(d_digit // 2) * 2 * np.pi / precision
        pe_digit[:, 0::2] = np.sin(position * div_term_digit)
        pe_digit[:, 1::2] = np.cos(position * div_term_digit)
        return jnp.array(pe_digit)


def unsqueeze(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """Add a dimension to a tensor.
    
    Args:
        x: Input tensor.
        axis: Position where to add the new dimension.
    
    Returns:
        Tensor with an additional dimension.
    """
    return jnp.expand_dims(x, axis)


class Unsqueeze(nn.Module):
    """Layer that adds a dimension to a tensor.
    
    Attributes:
        axis: Position where to add the new dimension.
    """
    axis: int = -1
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.expand_dims(x, self.axis)


class Stack(nn.Module):
    """Layer that stacks multiple tensors along an axis.
    
    Attributes:
        axis: Axis along which to stack.
    """
    axis: int = -1
    
    @nn.compact
    def __call__(self, xs: Sequence[jnp.ndarray]) -> jnp.ndarray:
        """Stack input tensors.
        
        Args:
            xs: Sequence of tensors to stack.
        
        Returns:
            Stacked tensor.
        """
        return jnp.stack(xs, axis=self.axis)


# Functional aliases for convenience
def stack(xs: Sequence[jnp.ndarray], axis: int = -1) -> jnp.ndarray:
    """Stack tensors along an axis."""
    return jnp.stack(xs, axis=axis)
