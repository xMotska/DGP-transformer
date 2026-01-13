"""Transformer models using Flax."""

from typing import Callable, Sequence

import gin
import jax
import jax.numpy as jnp
from flax import linen as nn

from layers import CausalConv, DigitEncoding
from attention import DotProductCausalRotaryAttention


class ShiftRight(nn.Module):
    """Shift the input sequence right by one position.
    
    Used for autoregressive models to ensure the model can't see the current
    token when predicting it.
    
    Attributes:
        pad_value: Value to use for padding on the left.
    """
    pad_value: int = 0
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, decode: bool = False) -> jnp.ndarray:
        """Shift input right.
        
        Args:
            x: Input tensor of shape [batch, seq_len, ...].
            decode: If True, don't shift (already handling single tokens).
        
        Returns:
            Right-shifted tensor with same shape.
        """
        if decode:
            # In decode mode, we process one token at a time
            # The caching mechanism handles the history
            return x
        
        # Shift right: prepend pad, drop last
        pad_shape = (x.shape[0], 1) + x.shape[2:]
        pad = jnp.full(pad_shape, self.pad_value, dtype=x.dtype)
        return jnp.concatenate([pad, x[:, :-1]], axis=1)


class FeedForwardBlock(nn.Module):
    """Transformer feed-forward block.
    
    Attributes:
        d_model: Model dimension.
        d_ff: Hidden dimension (typically 4 * d_model).
        dropout_rate: Dropout probability.
        activation: Activation function.
    """
    d_model: int
    d_ff: int
    dropout_rate: float = 0.0
    activation: Callable = nn.relu
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        """Apply feed-forward block.
        
        Args:
            x: Input tensor of shape [batch, seq_len, d_model].
            deterministic: If True, disable dropout.
        
        Returns:
            Output tensor of same shape.
        """
        x = nn.Dense(self.d_ff)(x)
        x = self.activation(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = nn.Dense(self.d_model)(x)
        return x


class RotaryCausalAttention(nn.Module):
    """Multi-head causal attention with rotary position embeddings.
    
    Attributes:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        dropout_rate: Dropout probability.
        fraction_to_rotate: Fraction of head dimension to apply rotary encoding.
        kernel_width: Optional kernel width for convolutional Q/K projections.
        max_len: Maximum sequence length.
        expand_heads: If True, expand dimensions for Q/K/V projections.
    """
    d_model: int
    n_heads: int = 1
    dropout_rate: float = 0.0
    fraction_to_rotate: float = 0.25
    kernel_width: int | None = None
    max_len: int = 2048
    expand_heads: bool = False
    
    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = False,
        decode: bool = False
    ) -> jnp.ndarray:
        """Apply rotary causal attention.
        
        Args:
            x: Input tensor of shape [batch, seq_len, d_model].
            deterministic: If True, disable dropout.
            decode: If True, use cached state for autoregressive decoding.
        
        Returns:
            Output tensor of same shape.
        """
        batch_size, seq_len, _ = x.shape
        d_out = self.d_model * self.n_heads if self.expand_heads else self.d_model
        d_head = d_out // self.n_heads
        
        # Q, K, V projections
        if self.kernel_width is not None:
            q = CausalConv(features=d_out, kernel_size=self.kernel_width)(x, decode=decode)
            k = CausalConv(features=d_out, kernel_size=self.kernel_width)(x, decode=decode)
        else:
            q = nn.Dense(d_out)(x)
            k = nn.Dense(d_out)(x)
        v = nn.Dense(d_out)(x)
        
        # Reshape for multi-head attention: [batch, seq, heads, d_head]
        q = q.reshape(batch_size, seq_len, self.n_heads, d_head)
        k = k.reshape(batch_size, seq_len, self.n_heads, d_head)
        v = v.reshape(batch_size, seq_len, self.n_heads, d_head)
        
        # Transpose to [batch, heads, seq, d_head]
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        
        # Apply rotary attention
        attn = DotProductCausalRotaryAttention(
            num_heads=self.n_heads,
            d_head=d_head,
            fraction_to_rotate=self.fraction_to_rotate,
            dropout_rate=self.dropout_rate,
            max_len=self.max_len,
        )
        out = attn((q, k, v), deterministic=deterministic, decode=decode)
        
        # Transpose back and reshape: [batch, seq, d_out]
        out = jnp.transpose(out, (0, 2, 1, 3))
        out = out.reshape(batch_size, seq_len, d_out)
        
        # Output projection
        out = nn.Dense(self.d_model)(out)
        
        # Additional projection if expanding heads
        if self.expand_heads:
            out = nn.Dense(self.d_model)(out)
        
        return out


class DecoderBlock(nn.Module):
    """Transformer decoder block with pre-norm architecture.
    
    Attributes:
        d_model: Model dimension.
        d_ff: Feed-forward hidden dimension.
        n_heads: Number of attention heads.
        dropout_rate: Dropout probability.
        fraction_to_rotate: Fraction of head dim for rotary encoding.
        kernel_width: Optional kernel width for attention convolutions.
        max_len: Maximum sequence length.
        expand_heads: If True, expand dimensions in attention.
        activation: Activation function for feed-forward block.
    """
    d_model: int
    d_ff: int
    n_heads: int
    dropout_rate: float = 0.0
    fraction_to_rotate: float = 0.25
    kernel_width: int | None = None
    max_len: int = 2048
    expand_heads: bool = False
    activation: Callable = nn.relu
    
    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = False,
        decode: bool = False
    ) -> jnp.ndarray:
        """Apply decoder block.
        
        Args:
            x: Input tensor of shape [batch, seq_len, d_model].
            deterministic: If True, disable dropout.
            decode: If True, use cached state for autoregressive decoding.
        
        Returns:
            Output tensor of same shape.
        """
        # Self-attention with residual
        residual = x
        x = nn.LayerNorm()(x)
        x = RotaryCausalAttention(
            d_model=self.d_model,
            n_heads=self.n_heads,
            dropout_rate=self.dropout_rate,
            fraction_to_rotate=self.fraction_to_rotate,
            kernel_width=self.kernel_width,
            max_len=self.max_len,
            expand_heads=self.expand_heads,
        )(x, deterministic=deterministic, decode=decode)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = residual + x
        
        # Feed-forward with residual
        residual = x
        x = nn.LayerNorm()(x)
        x = FeedForwardBlock(
            d_model=self.d_model,
            d_ff=self.d_ff,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
        )(x, deterministic=deterministic)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = residual + x
        
        return x


@gin.configurable(module='code.models')
class TransformerBody(nn.Module):
    """Transformer decoder body without embedding/output layers.
    
    Attributes:
        d_model: Model dimension.
        d_ff_mul: Multiplier for feed-forward hidden dimension.
        n_layers: Number of decoder blocks.
        n_heads: Number of attention heads.
        max_len: Maximum sequence length.
        dropout_rate: Dropout probability.
        conv_kernel: Kernel size for initial convolution.
        conv_activation: Optional activation after initial convolution.
        fraction_to_rotate: Fraction of head dim for rotary encoding.
        conv_attention_kernel_width: Optional kernel width for attention convs.
        expand_heads: If True, expand dimensions in attention.
        precision: Precision for digit encoding (None to disable).
        digit_encoding: Whether to use digit encoding.
        activation: Activation function for feed-forward blocks.
    """
    d_model: int = 256
    d_ff_mul: int = 2
    n_layers: int = 2
    n_heads: int = 2
    max_len: int = 2048
    dropout_rate: float = 0.1
    conv_kernel: int = 1
    conv_activation: Callable | None = None
    fraction_to_rotate: float = 0.25
    conv_attention_kernel_width: int | None = None
    expand_heads: bool = False
    precision: int | None = None
    digit_encoding: bool = False
    activation: Callable = nn.relu
    
    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = False,
        decode: bool = False
    ) -> jnp.ndarray:
        """Apply transformer body.
        
        Args:
            x: Input tensor of shape [batch, seq_len, d_input].
            deterministic: If True, disable dropout.
            decode: If True, use cached state for autoregressive decoding.
        
        Returns:
            Output tensor of shape [batch, seq_len, d_model].
        """
        # Shift right for causal modeling
        x = ShiftRight()(x, decode=decode)
        
        # Initial dropout
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        
        # Initial convolution
        x = CausalConv(features=self.d_model, kernel_size=self.conv_kernel)(x, decode=decode)
        if self.conv_activation is not None:
            x = self.conv_activation(x)
        
        # Optional digit encoding
        if self.digit_encoding and self.precision is not None:
            x = DigitEncoding(
                max_len=self.max_len,
                precision=self.precision,
            )(x, deterministic=deterministic, decode=decode)
        
        # Decoder blocks
        for _ in range(self.n_layers):
            x = DecoderBlock(
                d_model=self.d_model,
                d_ff=self.d_ff_mul * self.d_model,
                n_heads=self.n_heads,
                dropout_rate=self.dropout_rate,
                fraction_to_rotate=self.fraction_to_rotate,
                kernel_width=self.conv_attention_kernel_width,
                max_len=self.max_len,
                expand_heads=self.expand_heads,
                activation=self.activation,
            )(x, deterministic=deterministic, decode=decode)
        
        # Final layer norm
        x = nn.LayerNorm()(x)
        
        return x


@gin.configurable(module='code.models')
class ConvTransformerLM(nn.Module):
    """Convolutional Transformer language model.
    
    Performs autoregressive language modeling with:
    - Input: Token IDs of shape [batch, seq_len]
    - Output: Logits of shape [batch, seq_len, vocab_size]
    
    Attributes:
        vocab_size: Size of the vocabulary.
        d_model: Model dimension.
        d_ff_mul: Multiplier for feed-forward hidden dimension.
        n_layers: Number of decoder blocks.
        n_heads: Number of attention heads.
        max_len: Maximum sequence length.
        dropout_rate: Dropout probability.
        conv_kernel: Kernel size for initial convolution.
        conv_activation: Optional activation after initial convolution.
        conv_attention_kernel_width: Optional kernel width for attention convs.
        expand_heads: If True, expand dimensions in attention.
        precision: Precision for digit encoding.
        digit_encoding: Whether to use digit encoding.
        activation: Activation function for feed-forward blocks.
    """
    vocab_size: int
    d_model: int = 256
    d_ff_mul: int = 2
    n_layers: int = 2
    n_heads: int = 2
    max_len: int = 2048
    dropout_rate: float = 0.1
    conv_kernel: int = 1
    conv_activation: Callable | None = None
    conv_attention_kernel_width: int | None = None
    expand_heads: bool = False
    precision: int | None = None
    digit_encoding: bool = True
    activation: Callable = nn.relu
    
    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = False,
        decode: bool = False
    ) -> jnp.ndarray:
        """Apply language model.
        
        Args:
            x: Token IDs of shape [batch, seq_len].
            deterministic: If True, disable dropout.
            decode: If True, use cached state for autoregressive decoding.
        
        Returns:
            Logits of shape [batch, seq_len, vocab_size].
        """
        # Shift right for causal modeling
        x = ShiftRight(pad_value=0)(x, decode=decode)
        
        # Token embedding
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.d_model)(x)
        
        # Dropout after embedding
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        
        # Initial convolution
        x = CausalConv(features=self.d_model, kernel_size=self.conv_kernel)(x, decode=decode)
        if self.conv_activation is not None:
            x = self.conv_activation(x)
        
        # Optional digit encoding
        if self.digit_encoding and self.precision is not None:
            x = DigitEncoding(
                max_len=self.max_len,
                precision=self.precision,
            )(x, deterministic=deterministic, decode=decode)
        
        # Decoder blocks
        for _ in range(self.n_layers):
            x = DecoderBlock(
                d_model=self.d_model,
                d_ff=self.d_ff_mul * self.d_model,
                n_heads=self.n_heads,
                dropout_rate=self.dropout_rate,
                kernel_width=self.conv_attention_kernel_width,
                max_len=self.max_len,
                expand_heads=self.expand_heads,
                activation=self.activation,
            )(x, deterministic=deterministic, decode=decode)
        
        # Final layer norm
        x = nn.LayerNorm()(x)
        
        # Output projection to vocab
        x = nn.Dense(self.vocab_size)(x)
        
        return x
