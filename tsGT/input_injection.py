"""Input injection layers for auxiliary inputs (e.g., time features)."""

from typing import Sequence

import jax.numpy as jnp
from flax import linen as nn


class InjectInputs(nn.Module):
    """Inject discrete auxiliary inputs (e.g., time) into the model.
    
    Takes context embeddings and auxiliary inputs, embeds the inputs,
    and adds them to the context.
    
    Attributes:
        input_vocab_sizes: Vocab sizes for each auxiliary input.
            If None, auxiliary inputs are ignored.
            Example: [30, 7, 24, 512] for day-of-month, day-of-week, hour, series_id.
            Use None for individual entries to skip that input.
        d_emb: Embedding dimension.
    """
    input_vocab_sizes: Sequence[int | None] | None
    d_emb: int
    
    @nn.compact
    def __call__(
        self, 
        context_emb: jnp.ndarray, 
        inputs: jnp.ndarray
    ) -> jnp.ndarray:
        """Inject auxiliary inputs into context embeddings.
        
        Args:
            context_emb: Context embeddings of shape (batch, seq_len, d_emb).
            inputs: Auxiliary inputs of shape (batch, seq_len, n_inputs).
        
        Returns:
            Combined embeddings of shape (batch, seq_len, d_emb).
        """
        if self.input_vocab_sizes is None:
            # Ignore auxiliary inputs, just return context
            return context_emb
        
        # Embed each auxiliary input
        input_embs = []
        for i, vocab_size in enumerate(self.input_vocab_sizes):
            if vocab_size is not None:
                # Get the i-th input and embed it
                inp = inputs[..., i].astype(jnp.int32)
                emb = nn.Embed(
                    num_embeddings=vocab_size, 
                    features=self.d_emb,
                    name=f'input_embed_{i}'
                )(inp)
                input_embs.append(emb)
        
        if len(input_embs) == 0:
            # All inputs were None, just return context
            return context_emb
        
        # Stack and sum embeddings
        # input_embs: list of (batch, seq_len, d_emb)
        stacked = jnp.stack(input_embs, axis=-2)  # (batch, seq_len, n_inputs, d_emb)
        summed = jnp.sum(stacked, axis=-2)  # (batch, seq_len, d_emb)
        input_emb = nn.LayerNorm(name='input_ln')(summed)
        
        # Add to context and normalize
        combined = context_emb + input_emb
        return nn.LayerNorm(name='combined_ln')(combined)


def create_inject_inputs(
    input_vocab_sizes: Sequence[int | None] | None,
    d_emb: int
) -> InjectInputs:
    """Factory function to create InjectInputs layer.
    
    Args:
        input_vocab_sizes: Vocab sizes for each auxiliary input.
        d_emb: Embedding dimension.
    
    Returns:
        InjectInputs module.
    """
    return InjectInputs(input_vocab_sizes=input_vocab_sizes, d_emb=d_emb)


# Functional version for use outside nn.Module
def inject_inputs_fn(
    context_emb: jnp.ndarray,
    inputs: jnp.ndarray,
    input_embeddings: list[jnp.ndarray],
    input_vocab_sizes: Sequence[int | None] | None,
) -> jnp.ndarray:
    """Functional version of input injection.
    
    Args:
        context_emb: Context embeddings of shape (batch, seq_len, d_emb).
        inputs: Auxiliary inputs of shape (batch, seq_len, n_inputs).
        input_embeddings: List of embedding weight matrices.
        input_vocab_sizes: Vocab sizes (to know which inputs to skip).
    
    Returns:
        Combined embeddings.
    """
    if input_vocab_sizes is None:
        return context_emb
    
    input_embs = []
    emb_idx = 0
    for i, vocab_size in enumerate(input_vocab_sizes):
        if vocab_size is not None:
            inp = inputs[..., i].astype(jnp.int32)
            emb = input_embeddings[emb_idx][inp]
            input_embs.append(emb)
            emb_idx += 1
    
    if len(input_embs) == 0:
        return context_emb
    
    stacked = jnp.stack(input_embs, axis=-2)
    summed = jnp.sum(stacked, axis=-2)
    
    return context_emb + summed
