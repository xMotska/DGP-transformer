"""Rotary Position Embedding Attention for Flax.

Migrated from Trax to Flax Linen with modern JAX APIs.
"""

import math
from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn


def calculate_sin_cos_rotary(
    rotary_dim: int,
    n_ctx: int,
    base: int = 10000
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate sin and cos for rotary embeddings.
    
    Args:
        rotary_dim: Dimension of rotary embedding.
        n_ctx: Maximum context length.
        base: Base for frequency calculation.
    
    Returns:
        Tuple of (sin, cos) arrays, each of shape [n_ctx, rotary_dim].
    """
    pos = jnp.arange(n_ctx, dtype=jnp.float32)
    dim = jnp.arange(rotary_dim // 2, dtype=jnp.float32)
    # A set of frequencies evenly spaced in log space.
    # Each element of freq encodes one clock hand angular velocity.
    # Take the `base` to the power [0, 2/dim, ..., (dim-2)/dim].
    freq = base ** (dim / (rotary_dim / 2))
    # Repeat "pixels", so [f0, f1, ...] -> [f0, f0, f1, f1, ...]
    # Repetition needed to compute rotary embedding: x * cos + rotate(x) * sin
    freq = jnp.repeat(freq, 2)
    # Create a n_ctx x rotary_dim tensor, where each column is an
    # arithmetic sequence of angles in that frequency
    angles = pos[:, None] / freq[None, :]
    return jnp.sin(angles), jnp.cos(angles)


def rotate_every_two(x: jnp.ndarray) -> jnp.ndarray:
    """Splits x into blocks along the final axis and maps [x0, x1] to [-x1, x0]."""
    x0, x1 = x[..., ::2], x[..., 1::2]
    # Stack and rearrange since JAX arrays are immutable.
    x = jnp.stack([-x1, x0], axis=-1)  # [..., d_head / 2, 2]
    # Rearrange to [..., d_head]
    # Equivalent to einops.rearrange(x, '... d j -> ... (d j)')
    return jnp.reshape(x, x.shape[:-2] + (-1,))


def apply_rotary_embedding(
    x: jnp.ndarray,
    sin: jnp.ndarray,
    cos: jnp.ndarray,
    rotary_dim: int,
    offset: int = 0
) -> jnp.ndarray:
    """Apply rotary positional embedding to x.
    
    Args:
        x: Input tensor of shape [batch, heads, seq_len, d_head] or 
           [batch * heads, seq_len, d_head].
        sin: Precomputed sin values [n_ctx, rotary_dim].
        cos: Precomputed cos values [n_ctx, rotary_dim].
        rotary_dim: Number of dimensions to rotate.
        offset: Position offset for incremental decoding.
    
    Returns:
        Tensor with rotary embeddings applied.
    """
    seq_len = x.shape[-2]
    
    x_rot = x[..., :rotary_dim]
    x_unrotated = x[..., rotary_dim:]
    x_flip = rotate_every_two(x_rot)
    
    # Slice sin/cos for current sequence positions
    cos_slice = jax.lax.dynamic_slice_in_dim(cos, offset, seq_len, axis=0)
    sin_slice = jax.lax.dynamic_slice_in_dim(sin, offset, seq_len, axis=0)
    
    x_rotated = x_rot * cos_slice + x_flip * sin_slice
    return jnp.concatenate([x_rotated, x_unrotated], axis=-1)


class DotProductCausalRotaryAttention(nn.Module):
    """Causal self-attention with rotary position embeddings.
    
    For rotary embedding, see:
    * The original paper Su, et. al. RoFormer, https://arxiv.org/abs/2104.09864.
    * The blog post by EleutherAI https://blog.eleuther.ai/rotary-embeddings/.
    
    Attributes:
        num_heads: Number of attention heads.
        d_head: Dimension per head. If None, inferred from input.
        fraction_to_rotate: Fraction of d_head to apply rotary embedding to.
        dropout_rate: Dropout rate for attention weights.
        max_len: Maximum sequence length for positional embeddings.
        dtype: Data type for computations.
    """
    num_heads: int
    d_head: Optional[int] = None
    fraction_to_rotate: float = 0.25
    dropout_rate: float = 0.0
    max_len: int = 2048
    dtype: jnp.dtype = jnp.float32
    
    def _to_even(self, x: float) -> int:
        return math.floor(x / 2.0) * 2
    
    def _get_rotary_dim(self, d_head: int) -> int:
        """Compute rotary dimension from d_head."""
        if self.fraction_to_rotate > 1.0 or self.fraction_to_rotate <= 0.0:
            raise ValueError(
                f'fraction_to_rotate must be in (0, 1], got {self.fraction_to_rotate}.')
        return self._to_even(d_head * self.fraction_to_rotate)
    
    @nn.compact
    def __call__(
        self,
        inputs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        deterministic: bool = False,
        decode: bool = False,
    ) -> jnp.ndarray:
        """Apply causal rotary attention.
        
        Args:
            inputs: Tuple of (queries, keys, values), each of shape
                [batch, heads, seq_len, d_head] or [batch * heads, seq_len, d_head].
            deterministic: If True, disable dropout.
            decode: If True, use cached key/value for autoregressive decoding.
        
        Returns:
            Output tensor of same shape as inputs.
        """
        q, k, v = inputs
        
        # Infer d_head from input
        d_head = self.d_head if self.d_head is not None else q.shape[-1]
        rotary_dim = self._get_rotary_dim(d_head)
        
        # Compute sin/cos for rotary embeddings.
        # Note: Computing inline is fine—JAX's JIT compilation ensures this
        # is only computed once per unique (rotary_dim, max_len) during tracing.
        sin, cos = calculate_sin_cos_rotary(
            rotary_dim=rotary_dim,
            n_ctx=self.max_len
        )
        
        # Handle autoregressive decoding with cached keys/values
        if decode:
            # Initialize or retrieve cache
            is_initialized = self.has_variable('cache', 'cached_key')
            cached_key = self.variable(
                'cache', 'cached_key',
                lambda: jnp.zeros((q.shape[0], q.shape[1], self.max_len, d_head), dtype=self.dtype)
            )
            cached_value = self.variable(
                'cache', 'cached_value',
                lambda: jnp.zeros((q.shape[0], q.shape[1], self.max_len, d_head), dtype=self.dtype)
            )
            cache_index = self.variable(
                'cache', 'cache_index',
                lambda: jnp.array(0, dtype=jnp.int32)
            )
            
            offset = cache_index.value
            seq_len = k.shape[-2]
            
            # Apply rotary embeddings at the correct offset
            q = apply_rotary_embedding(q, sin, cos, rotary_dim, offset=offset)
            k = apply_rotary_embedding(k, sin, cos, rotary_dim, offset=offset)
            
            # FIX #1 & #3: Robust cache update that handles both:
            # - First pass (prefill): may have seq_len > 1
            # - Subsequent passes: typically seq_len = 1
            # Use dynamic_update_slice for proper indexing regardless of seq_len
            indices = (0, 0, offset, 0)  # (batch, heads, seq_pos, d_head)
            cached_key.value = jax.lax.dynamic_update_slice(
                cached_key.value, k, indices
            )
            cached_value.value = jax.lax.dynamic_update_slice(
                cached_value.value, v, indices
            )
            cache_index.value = offset + seq_len
            
            # Retrieve cached k, v up to current position
            k = jax.lax.dynamic_slice(
                cached_key.value,
                (0, 0, 0, 0),
                (k.shape[0], k.shape[1], offset + seq_len, d_head)
            )
            v = jax.lax.dynamic_slice(
                cached_value.value,
                (0, 0, 0, 0),
                (v.shape[0], v.shape[1], offset + seq_len, d_head)
            )
        else:
            # Training/eval mode: apply rotary embeddings
            q = apply_rotary_embedding(q, sin, cos, rotary_dim, offset=0)
            k = apply_rotary_embedding(k, sin, cos, rotary_dim, offset=0)
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(d_head)
        attn_weights = jnp.einsum('...qd,...kd->...qk', q, k) * scale
        
        # Apply causal mask
        seq_len_q, seq_len_k = q.shape[-2], k.shape[-2]
        if decode:
            # In decode mode, q attends to all positions up to and including current
            # For prefill (seq_len_q > 1): need proper causal mask
            # For generation (seq_len_q = 1): can attend to all cached positions
            if seq_len_q == 1:
                causal_mask = jnp.ones((seq_len_q, seq_len_k), dtype=bool)
            else:
                # Account for offset in chunked decode
                query_positions = jnp.arange(seq_len_q) + offset
                key_positions = jnp.arange(seq_len_k)
                causal_mask = query_positions[:, None] >= key_positions[None, :]
        else:
            causal_mask = jnp.tril(jnp.ones((seq_len_q, seq_len_k), dtype=bool))
        
        attn_weights = jnp.where(causal_mask, attn_weights, -1e9)
        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        
        # Apply dropout
        if not deterministic and self.dropout_rate > 0.0:
            dropout_rng = self.make_rng('dropout')
            keep_prob = 1.0 - self.dropout_rate
            dropout_mask = jax.random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)
            attn_weights = jnp.where(dropout_mask, attn_weights / keep_prob, 0.0)
        
        # Compute output
        output = jnp.einsum('...qk,...kd->...qd', attn_weights, v)
        
        return output
