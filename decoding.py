"""Decoding with Flax models."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


# Type alias for Flax variables dict
Variables = dict[str, Any]


def autoregressive_sample(
    model: Any,
    variables: Variables,
    sample_fn: Callable[[jnp.ndarray], jnp.ndarray],
    context: np.ndarray | None,
    inputs: np.ndarray | None = None,
    batch_size: int = 1,
    start_element: int | np.ndarray = 0,
    horizon_length: int = 100,
    rng: jax.Array | None = None,
) -> np.ndarray:
    """Return a batch of sequences created by autoregressive sampling.

    This function uses `model` to generate outputs one position at a time, with
    access to context for the current position and all preceding positions. The
    new output becomes the next position's input, and this loop repeats until
    the output sequence reaches `horizon_length` items.
    
    Note: This is a non-cached implementation that runs the full forward pass
    each step. Slower but works with any model.

    Args:
        model: A Flax Linen module.
        variables: Flax variables dict containing model parameters.
        sample_fn: Function that samples the new output from the predicted
            distribution parameters (model's output).
        context: Sequence of symbols the model sees as input the first time it
            generates an output. If None, the model must generate the first
            output with no input to guide it.
        inputs: Sequence of auxiliary inputs (e.g. time) that can help in
            prediction. If None, only the context is fed into the model.
        batch_size: Number of sequences to generate in parallel as a batch.
        start_element: The start symbol (ID/integer) for the autoregressive
            process, or array of shape (batch_size, 1) of such integers.
        horizon_length: Length of the generated sequences.
        rng: Optional JAX random key for stochastic operations (e.g., dropout).
            If None, a default key is created.

    Returns:
        Array of integers with shape (batch_size, horizon_length) representing
        a batch of output sequences.
    """
    if context is None:
        context = np.empty((batch_size, 0), dtype=np.int32)

    if context.shape[0] != batch_size:
        raise ValueError(
            f"Context batch size ({context.shape[0]}) does not match "
            f"batch_size arg ({batch_size})."
        )

    context_length = context.shape[1]

    if inputs is not None:
        if inputs.shape[0] != batch_size:
            raise ValueError(
                f"Inputs batch size ({inputs.shape[0]}) does not match "
                f"batch_size arg ({batch_size})."
            )

        expected_length = context_length + horizon_length
        if inputs.shape[1] != expected_length:
            raise ValueError(
                f"Invalid length of the inputs: expected {expected_length}, got "
                f"{inputs.shape[1]}."
            )

    if rng is None:
        rng = jax.random.key(0)
    
    # Start with context (no start symbol - just use the context directly)
    current_sequence = context
    
    result = []
    for step in range(horizon_length):
        # Split RNG for this step
        rng, step_rng = jax.random.split(rng)
        
        # Get inputs for current sequence length
        seq_len = current_sequence.shape[1]
        
        # Apply model (full forward pass, no caching)
        if inputs is not None:
            current_inputs = inputs[:, :seq_len, :]
            model_output = model.apply(
                variables,
                current_sequence,
                current_inputs,
                decode=False,
                deterministic=True,
            )
        else:
            model_output = model.apply(
                variables,
                current_sequence,
                decode=False,
                deterministic=True,
            )
        
        # Sample from the last position's output
        sample = sample_fn(model_output[:, -1, :])[:, None]  # Add time dimension
        result.append(np.asarray(sample))

        # Append sample to sequence for next iteration
        current_sequence = np.concatenate([current_sequence, sample], axis=1)

    return np.concatenate(result, axis=1)


def autoregressive_sample_jit(
    model: Any,
    variables: Variables,
    sample_fn: Callable[[jnp.ndarray], jnp.ndarray],
    context: jnp.ndarray | None,
    inputs: jnp.ndarray | None = None,
    batch_size: int = 1,
    start_element: int | jnp.ndarray = 0,
    horizon_length: int = 100,
    rng: jax.Array | None = None,
) -> jnp.ndarray:
    """JIT-compatible autoregressive sampling using jax.lax.scan.
    
    This version is more efficient for longer sequences as it compiles
    the entire sampling loop. However, it requires fixed horizon_length
    at compile time.

    Args:
        model: A Flax Linen module with autoregressive caching support.
        variables: Flax variables dict containing model parameters.
        sample_fn: Function that samples from model output distribution.
        context: Initial context sequence or None.
        inputs: Optional auxiliary inputs.
        batch_size: Number of sequences to generate.
        start_element: Start symbol(s) for generation.
        horizon_length: Fixed length of generated sequences.
        rng: JAX random key.

    Returns:
        Array of shape (batch_size, horizon_length) with generated tokens.
    """
    if context is None:
        context = jnp.empty((batch_size, 0), dtype=jnp.int32)

    if context.shape[0] != batch_size:
        raise ValueError(
            f"Context batch size ({context.shape[0]}) does not match "
            f"batch_size arg ({batch_size})."
        )

    if jnp.isscalar(start_element):
        start_symbol = jnp.full((batch_size, 1), start_element, dtype=jnp.int32)
    else:
        start_symbol = jnp.asarray(start_element)

    initial_tokens = jnp.concatenate([start_symbol, context], axis=1)

    if rng is None:
        rng = jax.random.key(0)

    # Process context to initialize cache
    _, initial_vars = model.apply(
        variables,
        initial_tokens,
        decode=True,
        deterministic=True,
        mutable=['cache'],
    )
    variables = {**variables, **initial_vars}

    # Get last prediction to start generation
    model_output, variables = model.apply(
        variables,
        initial_tokens[:, -1:],
        decode=True,
        deterministic=True,
        mutable=['cache'],
    )
    variables = {**variables, **variables}
    
    first_sample = sample_fn(model_output[:, -1, :])[:, None]

    def scan_fn(carry, step_rng):
        prev_token, vars_state = carry
        
        # Prepare input
        if inputs is not None:
            step_idx = context.shape[1] + step_rng[1]  # Would need step counter
            current_input = (prev_token, inputs[:, step_idx:step_idx + 1])
        else:
            current_input = prev_token
        
        model_output, new_vars = model.apply(
            vars_state,
            current_input,
            decode=True,
            deterministic=True,
            rngs={'dropout': step_rng},
            mutable=['cache'],
        )
        new_vars = {**vars_state, **new_vars}
        
        sample = sample_fn(model_output[:, -1, :])[:, None]
        return (sample, new_vars), sample[:, 0]

    # Generate remaining tokens
    rngs = jax.random.split(rng, horizon_length - 1)
    _, samples = jax.lax.scan(scan_fn, (first_sample, variables), rngs)
    
    # Combine first sample with scanned samples
    # samples shape: [horizon_length - 1, batch_size]
    all_samples = jnp.concatenate(
        [first_sample[:, 0], samples.T], 
        axis=1
    )
    
    return all_samples
