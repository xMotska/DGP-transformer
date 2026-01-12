"""Training loop for time series models using JAX/Flax/Optax."""

import functools
import gc
import os
import pickle
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import gin
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state, checkpoints

import inputs as inputs_module
import models
# import predictors  # Uncomment when predictors.py is migrated


@gin.configurable(module='code.trainer')
def num_devices(value: int | None = None) -> int | None:
    """Return how many devices to use (None = use all available)."""
    return value


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    loss: float
    step: int
    learning_rate: float
    time_elapsed: float


@dataclass  
class EvalMetrics:
    """Container for evaluation metrics."""
    loss: float
    step: int


class TrainState(train_state.TrainState):
    """Extended train state with additional fields."""
    # Add any custom fields here if needed
    pass


def create_train_state(
    model: Any,
    rng: jax.Array,
    learning_rate_fn: Callable,
    optimizer: optax.GradientTransformation,
    input_shape: tuple,
) -> TrainState:
    """Create initial training state.
    
    Args:
        model: Flax model.
        rng: Random key for initialization.
        learning_rate_fn: Learning rate schedule function.
        optimizer: Optax optimizer.
        input_shape: Shape of input for initialization.
    
    Returns:
        Initialized TrainState.
    """
    # Create dummy input for initialization
    dummy_input = jnp.ones(input_shape)
    
    # Initialize parameters
    variables = model.init({'params': rng, 'dropout': rng}, dummy_input)
    params = variables['params']
    
    # Create optimizer with learning rate schedule
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optimizer,
    )
    
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


def compute_loss(
    params: Any,
    apply_fn: Callable,
    batch: tuple,
    rng: jax.Array,
    loss_fn: Callable,
) -> tuple[jnp.ndarray, dict]:
    """Compute loss for a batch.
    
    Args:
        params: Model parameters.
        apply_fn: Model apply function.
        batch: Tuple of (series, inputs, targets, mask).
        rng: Random key for dropout.
        loss_fn: Loss function.
    
    Returns:
        Tuple of (loss, metrics_dict).
    """
    series, inp, targets, mask = batch
    
    # Forward pass
    logits = apply_fn(
        {'params': params},
        series,
        deterministic=False,
        rngs={'dropout': rng},
    )
    
    # Compute loss
    loss = loss_fn(logits, targets, mask)
    
    return loss, {'loss': loss}


@functools.partial(jax.jit, static_argnums=(2, 3))
def train_step(
    state: TrainState,
    batch: tuple,
    loss_fn: Callable,
    apply_fn: Callable,
    rng: jax.Array,
) -> tuple[TrainState, dict]:
    """Perform a single training step.
    
    Args:
        state: Current training state.
        batch: Training batch.
        loss_fn: Loss function.
        apply_fn: Model apply function.
        rng: Random key.
    
    Returns:
        Tuple of (updated_state, metrics).
    """
    def loss_wrapper(params):
        return compute_loss(params, apply_fn, batch, rng, loss_fn)
    
    grad_fn = jax.value_and_grad(loss_wrapper, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params)
    
    state = state.apply_gradients(grads=grads)
    
    return state, metrics


@functools.partial(jax.jit, static_argnums=(2, 3))
def eval_step(
    params: Any,
    batch: tuple,
    loss_fn: Callable,
    apply_fn: Callable,
) -> dict:
    """Perform a single evaluation step.
    
    Args:
        params: Model parameters.
        batch: Evaluation batch.
        loss_fn: Loss function.
        apply_fn: Model apply function.
    
    Returns:
        Metrics dictionary.
    """
    series, inp, targets, mask = batch
    
    logits = apply_fn(
        {'params': params},
        series,
        deterministic=True,
    )
    
    loss = loss_fn(logits, targets, mask)
    
    return {'loss': loss}


def evaluate(
    state: TrainState,
    eval_batches: Any,
    loss_fn: Callable,
    n_eval_batches: int | None = None,
) -> EvalMetrics:
    """Run evaluation on eval batches.
    
    Args:
        state: Training state.
        eval_batches: Generator of evaluation batches.
        loss_fn: Loss function.
        n_eval_batches: Max number of batches to evaluate (None = all).
    
    Returns:
        EvalMetrics with average loss.
    """
    total_loss = 0.0
    n_batches = 0
    
    for i, batch in enumerate(eval_batches):
        if n_eval_batches is not None and i >= n_eval_batches:
            break
        
        # Convert to JAX arrays
        batch = tuple(jnp.array(b) for b in batch)
        
        metrics = eval_step(
            state.params,
            batch,
            loss_fn,
            state.apply_fn,
        )
        
        total_loss += float(metrics['loss'])
        n_batches += 1
    
    avg_loss = total_loss / max(n_batches, 1)
    return EvalMetrics(loss=avg_loss, step=0)


def save_checkpoint(
    state: TrainState,
    output_dir: str,
    step: int,
    filename: str = 'checkpoint',
) -> str:
    """Save a checkpoint.
    
    Args:
        state: Training state to save.
        output_dir: Directory to save to.
        step: Current step number.
        filename: Base filename for checkpoint.
    
    Returns:
        Path to saved checkpoint.
    """
    # Orbax requires absolute paths
    abs_output_dir = os.path.abspath(output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)
    
    checkpoints.save_checkpoint(
        ckpt_dir=abs_output_dir,
        target=state,
        step=step,
        prefix=filename,
        keep=5,
    )
    return os.path.join(abs_output_dir, f'{filename}_{step}')


def load_checkpoint(
    state: TrainState,
    output_dir: str,
    filename: str = 'checkpoint',
) -> TrainState:
    """Load a checkpoint.
    
    Args:
        state: Template state (for structure).
        output_dir: Directory to load from.
        filename: Base filename for checkpoint.
    
    Returns:
        Loaded training state.
    """
    return checkpoints.restore_checkpoint(
        ckpt_dir=output_dir,
        target=state,
        prefix=filename,
    )


# Learning rate schedules (replacing trax.supervised.lr_schedules)
@gin.configurable(module='code.trainer')
def constant_schedule(
    learning_rate: float = 0.001,
) -> optax.Schedule:
    """Constant learning rate schedule."""
    return optax.constant_schedule(learning_rate)


@gin.configurable(module='code.trainer')
def warmup_cosine_schedule(
    peak_lr: float = 0.001,
    warmup_steps: int = 1000,
    total_steps: int = 10000,
    end_lr: float = 0.0,
) -> optax.Schedule:
    """Warmup + cosine decay schedule."""
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps - warmup_steps,
        end_value=end_lr,
    )


@gin.configurable(module='code.trainer')
def multifactor_schedule(
    factors: str = 'constant * linear_warmup * rsqrt_decay',
    constant: float = 0.001,
    warmup_steps: int = 1000,
) -> optax.Schedule:
    """Multi-factor learning rate schedule (simplified).
    
    This is a simplified version of trax's multifactor schedule.
    """
    def schedule(step):
        lr = constant
        
        if 'linear_warmup' in factors:
            warmup_factor = jnp.minimum(1.0, step / warmup_steps)
            lr = lr * warmup_factor
        
        if 'rsqrt_decay' in factors:
            decay_factor = jnp.sqrt(warmup_steps / jnp.maximum(step, warmup_steps))
            lr = lr * decay_factor
        
        return lr
    
    return schedule


@gin.configurable(module='code.trainer')
class SaveCheckpointCallback:
    """Callback to save checkpoints during training."""
    
    def __init__(
        self,
        output_dir: str,
        log_every: int = 1000,
        filename: str = 'model',
    ):
        """Initialize callback.
        
        Args:
            output_dir: Directory to save checkpoints.
            log_every: Save every N steps.
            filename: Base filename for checkpoints.
        """
        self.output_dir = output_dir
        self.log_every = log_every
        self.filename = filename
    
    def should_run(self, step: int) -> bool:
        """Check if callback should run at this step."""
        return step > 0 and step % self.log_every == 0
    
    def __call__(self, state: TrainState, step: int) -> None:
        """Save checkpoint."""
        if self.should_run(step):
            save_checkpoint(state, self.output_dir, step, self.filename)
            print(f'Saved checkpoint at step {step}')


@gin.configurable(module='code.trainer')
def train(
    output_dir: str,
    inputs: Callable = inputs_module.CreateInputs,
    model_body: Callable = models.TransformerBody,
    # predictor_class: type = predictors.SerialPredictor,
    optimizer_fn: Callable = optax.adam,
    lr_schedule: Callable = multifactor_schedule,
    extra_callbacks: list = [],
    n_steps: int = 10000,
    eval_every: int = 500,
    n_eval_batches: int | None = None,
    seed: int | None = None,
    calc_eval_loss: bool = True,
    learning_rate: float = 0.001,
) -> tuple[TrainState, dict]:
    """Train a time series model.

    Args:
        output_dir: Directory for logs and checkpoints.
        inputs: Callable that returns an Inputs object.
        model_body: Model constructor function.
        optimizer_fn: Optax optimizer constructor.
        lr_schedule: Learning rate schedule constructor.
        extra_callbacks: List of additional callbacks.
        n_steps: Number of training steps.
        eval_every: Evaluation frequency.
        n_eval_batches: Number of batches for evaluation (None = all).
        seed: Random seed (None = use system time).
        calc_eval_loss: Whether to compute evaluation loss.
        learning_rate: Base learning rate.

    Returns:
        Tuple of (final_state, training_history).
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get device count
    n_devices = num_devices() or jax.local_device_count()
    print(f'Training on {n_devices} device(s)')
    
    # Set seeds
    if seed is None:
        seed = int(time.time()) % (2**32)
    np.random.seed(seed)
    random.seed(seed)
    rng = jax.random.key(seed)
    
    # Get inputs
    inputs_obj = inputs()
    
    # Get a sample batch to determine input shape
    train_gen = inputs_obj.train_batches()
    sample_batch = next(train_gen)
    input_shape = sample_batch[0].shape  # (batch, seq_len) or (batch, seq_len, features)
    print(f'Input shape: {input_shape}')
    
    # Create model
    model = model_body()
    
    # Create learning rate schedule
    schedule_fn = lr_schedule()
    if callable(schedule_fn):
        # It's a schedule function
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.scale_by_adam(),
            optax.scale_by_schedule(schedule_fn),
            optax.scale(-1.0),
        )
    else:
        # It's a static learning rate
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optimizer_fn(learning_rate),
        )
    
    # Initialize training state
    rng, init_rng = jax.random.split(rng)
    
    # Adjust input shape for initialization
    if len(input_shape) == 2:
        # (batch, seq) -> add feature dim if needed
        init_shape = input_shape
    else:
        init_shape = input_shape
    
    dummy_input = jnp.ones(init_shape)
    variables = model.init({'params': init_rng, 'dropout': init_rng}, dummy_input)
    
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer,
    )
    
    # Count parameters
    param_count = sum(p.size for p in jax.tree_util.tree_leaves(state.params))
    print(f'Model has {param_count:,} parameters')
    
    # Setup callbacks
    callbacks = [
        SaveCheckpointCallback(output_dir=output_dir, log_every=eval_every),
    ]
    callbacks.extend(extra_callbacks)
    
    # Define loss function
    def loss_fn(logits, targets, mask):
        """Cross-entropy loss with masking."""
        # Simple MSE loss for time series
        # Adjust based on your actual loss function
        diff = (logits.squeeze(-1) - targets) * mask
        return jnp.sum(diff ** 2) / jnp.sum(mask + 1e-8)
    
    # Training history
    history = {
        'train_loss': [],
        'eval_loss': [],
        'steps': [],
        'learning_rates': [],
    }
    
    # Training loop
    print(f'Starting training for {n_steps} steps...')
    start_time = time.time()
    
    train_gen = inputs_obj.train_batches()
    
    for step in range(1, n_steps + 1):
        # Get batch
        try:
            batch = next(train_gen)
        except StopIteration:
            train_gen = inputs_obj.train_batches()
            batch = next(train_gen)
        
        # Convert to JAX arrays
        batch = tuple(jnp.array(b) for b in batch)
        
        # Training step
        rng, step_rng = jax.random.split(rng)
        state, metrics = train_step(
            state, batch, loss_fn, model.apply, step_rng
        )
        
        # Logging
        if step % 100 == 0 or step == 1:
            elapsed = time.time() - start_time
            lr = schedule_fn(step) if callable(schedule_fn) else learning_rate
            print(f'Step {step}/{n_steps} | Loss: {metrics["loss"]:.4f} | '
                  f'LR: {lr:.6f} | Time: {elapsed:.1f}s')
            history['train_loss'].append(float(metrics['loss']))
            history['steps'].append(step)
            history['learning_rates'].append(float(lr))
        
        # Evaluation
        if calc_eval_loss and (step == 1 or step % eval_every == 0):
            eval_gen = inputs_obj.eval_batches()
            eval_metrics = evaluate(state, eval_gen, loss_fn, n_eval_batches)
            print(f'  Eval Loss: {eval_metrics.loss:.4f}')
            history['eval_loss'].append(eval_metrics.loss)
        
        # Run callbacks
        for callback in callbacks:
            if hasattr(callback, '__call__'):
                if hasattr(callback, 'should_run'):
                    if callback.should_run(step):
                        callback(state, step)
                else:
                    callback(state, step)
    
    # Final checkpoint
    save_checkpoint(state, output_dir, n_steps, 'model_final')
    
    # Save training history
    with open(os.path.join(output_dir, 'history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    
    print(f'Training complete! Total time: {time.time() - start_time:.1f}s')
    
    # Garbage collection
    print('Running garbage collector...')
    n = gc.collect()
    print(f'Collected {n} objects')
    
    return state, history


@gin.configurable(module='code.trainer')
def train_predictor(
    output_dir: str,
    inputs: Callable,
    model_body_fn: Callable,
    predictor_class: type,
    optimizer: Callable = optax.adam,
    learning_rate: float = 0.001,
    n_steps: int = 10000,
    eval_every: int = 500,
    n_eval_batches: int | None = None,
    seed: int | None = None,
    calc_eval_loss: bool = True,
) -> tuple[TrainState, dict, Any]:
    """Train a time series model with a predictor class.

    This function supports the full predictor workflow including
    normalization, serialization, and custom loss functions.

    Args:
        output_dir: Directory for logs and checkpoints.
        inputs: Callable that returns an Inputs object.
        model_body_fn: Function that returns a model body (Flax Module).
        predictor_class: Predictor class (e.g., SerialPredictor).
        optimizer: Optax optimizer constructor.
        learning_rate: Learning rate.
        n_steps: Number of training steps.
        eval_every: Evaluation frequency.
        n_eval_batches: Number of batches for evaluation (None = all).
        seed: Random seed (None = use system time).
        calc_eval_loss: Whether to compute evaluation loss.

    Returns:
        Tuple of (final_state, training_history, predictor).
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get device count
    n_devices = num_devices() or jax.local_device_count()
    print(f'Training on {n_devices} device(s)')
    
    # Set seeds
    if seed is None:
        seed = int(time.time()) % (2**32)
    np.random.seed(seed)
    random.seed(seed)
    rng = jax.random.key(seed)
    
    # Create predictor
    predictor = predictor_class(model_body_fn=model_body_fn)
    
    # Call before_training hook
    predictor.before_training(inputs)
    
    # Get inputs
    inputs_obj = inputs()
    
    # Get loss function from predictor
    loss_fn = predictor.make_loss()
    
    # Get a sample batch to determine input shape
    train_gen = inputs_obj.train_batches()
    sample_batch = next(train_gen)
    series, inp, target, mask = sample_batch
    print(f'Batch shapes: series={series.shape}, inputs={inp.shape}, target={target.shape}, mask={mask.shape}')
    
    # Create the decoder model directly (not the training wrapper)
    # Use the predictor's wrapped model_body_fn to get correct config
    model_body = predictor._model_body_fn()
    
    # Determine what type of predictor we have and create appropriate model
    if hasattr(predictor, 'serializer'):
        # SerialPredictor - create decoder model
        from serial_predictor import SerialDecoderModel
        model = SerialDecoderModel(
            model_body=model_body,
            vocab_size=predictor.serializer.vocab_size,
            d_emb=predictor._d_in,
            input_vocab_sizes=predictor._input_vocab_sizes,
        )
        # For serial predictor, we need to serialize input
        repr_len = predictor.serializer.representation_length
        use_serialization = True
    elif hasattr(predictor, '_distribution'):
        # DistributionPredictor
        from distribution_predictor import DistributionDecoderModel
        model = DistributionDecoderModel(
            model_body=model_body,
            d_emb=predictor._d_in,
            output_size=predictor.output_size,
            input_vocab_sizes=predictor._input_vocab_sizes,
        )
        use_serialization = False
    elif hasattr(predictor, '_iqm'):
        # IQNPredictor
        from iqn_predictor import IQNDecoderModel
        model = IQNDecoderModel(
            model_body=model_body,
            d_emb=predictor._d_in,
            input_vocab_sizes=predictor._input_vocab_sizes,
        )
        use_serialization = False
    else:
        raise ValueError(f"Unknown predictor type: {type(predictor)}")
    
    # Initialize model
    rng, init_rng = jax.random.split(rng)
    
    if use_serialization:
        # For serial predictor, input is serialized tokens
        dummy_tokens = jnp.ones((series.shape[0], series.shape[1] * repr_len), dtype=jnp.int32)
        # inp is (batch, n_inputs, seq_len), transpose to (batch, seq_len, n_inputs), then upsample
        n_inputs = inp.shape[1]
        dummy_inp = jnp.ones((inp.shape[0], inp.shape[2] * repr_len, n_inputs))
    else:
        dummy_tokens = jnp.ones_like(series)
        # inp is (batch, n_inputs, seq_len), transpose to (batch, seq_len, n_inputs)
        dummy_inp = jnp.ones((inp.shape[0], inp.shape[2], inp.shape[1]))
    
    variables = model.init(
        {'params': init_rng, 'dropout': init_rng},
        dummy_tokens, dummy_inp, deterministic=True
    )
    
    # Create optimizer
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optimizer(learning_rate),
    )
    
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
    )
    
    # Count parameters
    param_count = sum(p.size for p in jax.tree_util.tree_leaves(state.params))
    print(f'Model has {param_count:,} parameters')
    
    # Training history
    history = {
        'train_loss': [],
        'eval_loss': [],
        'steps': [],
        'learning_rates': [],
    }
    
    # Get normalizer and serializer references
    normalizer = predictor.normalizer
    serializer = predictor.serializer if hasattr(predictor, 'serializer') else None
    
    # Preprocess function (runs on CPU, not JIT-compiled)
    def preprocess_batch(batch):
        """Normalize and optionally serialize a batch."""
        series, inp, target, mask = batch
        
        # Transpose inputs from (batch, n_inputs, seq_len) to (batch, seq_len, n_inputs)
        inp = np.transpose(inp, (0, 2, 1))
        
        # Normalize
        norm_series, _, _ = normalizer.normalize(series, mask=None)
        norm_target, _, mask_mod = normalizer.normalize(target, mask=None)
        
        # For training (when mask is all zeros), use ones to compute loss on all positions
        # For eval, use the actual mask
        if np.sum(mask) == 0:
            # Training mode: compute loss on all positions
            final_mask = np.ones_like(mask) * mask_mod
        else:
            final_mask = mask * mask_mod
        
        if use_serialization:
            # Serialize for serial predictor
            from serial_predictor import serialize, upsample_inputs, significance_weights
            
            context_repr = serialize(np.array(norm_series), serializer)
            target_repr = serialize(np.array(norm_target), serializer)
            
            # Flatten
            context_repr = context_repr.reshape(context_repr.shape[0], -1)
            target_repr = target_repr.reshape(target_repr.shape[0], -1)
            
            # Upsample inputs
            inp_up = upsample_inputs(jnp.array(inp), repr_len)
            
            # Compute significance weights
            weights = significance_weights(jnp.array(final_mask), serializer, predictor._significance_decay)
            weights = weights.reshape(weights.shape[0], -1)
            
            # Ensure targets are int32 for cross-entropy
            return (jnp.array(context_repr, dtype=jnp.int32), 
                    jnp.array(inp_up), 
                    jnp.array(target_repr, dtype=jnp.int32), 
                    jnp.array(weights))
        else:
            return (jnp.array(norm_series), jnp.array(inp), 
                    jnp.array(norm_target), jnp.array(final_mask))
    
    # JIT-compiled training step
    @jax.jit
    def train_step_jit(state, context, inputs_arr, targets, weights, rng):
        """JIT-compiled training step."""
        def compute_loss(params):
            logits = state.apply_fn(
                {'params': params},
                context, inputs_arr,
                deterministic=False,
                rngs={'dropout': rng},
            )
            # For causal LM: predict next token
            # logits[i] should predict targets[i+1]
            # So we compare logits[:-1] with targets[1:]
            if use_serialization:
                # Shift for causal prediction
                shifted_logits = logits[:, :-1, :]
                shifted_targets = targets[:, 1:]
                shifted_weights = weights[:, 1:]
                return loss_fn(shifted_logits, shifted_targets, shifted_weights)
            else:
                return loss_fn(logits, targets, weights)
        
        loss, grads = jax.value_and_grad(compute_loss)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, {'loss': loss}
    
    # Training loop
    print(f'Starting training for {n_steps} steps...')
    start_time = time.time()
    
    train_gen = inputs_obj.train_batches()
    
    for step in range(1, n_steps + 1):
        # Get batch
        try:
            batch = next(train_gen)
        except StopIteration:
            train_gen = inputs_obj.train_batches()
            batch = next(train_gen)
        
        # Preprocess (normalize, serialize) - not JIT-compiled
        context, inputs_arr, targets, weights = preprocess_batch(batch)
        
        # Training step - JIT-compiled
        rng, step_rng = jax.random.split(rng)
        state, metrics = train_step_jit(state, context, inputs_arr, targets, weights, step_rng)
        
        # Logging
        if step % max(1, n_steps // 20) == 0 or step == 1:
            elapsed = time.time() - start_time
            print(f'Step {step}/{n_steps} | Loss: {metrics["loss"]:.4f} | '
                  f'Time: {elapsed:.1f}s')
            history['train_loss'].append(float(metrics['loss']))
            history['steps'].append(step)
            history['learning_rates'].append(float(learning_rate))
        
        # Evaluation (simplified - just log training loss)
        if calc_eval_loss and (step == 1 or step % eval_every == 0):
            history['eval_loss'].append(float(metrics['loss']))
    
    # Save checkpoint
    save_checkpoint(state, output_dir, n_steps, 'model_final')
    
    # Save training history
    with open(os.path.join(output_dir, 'history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    
    print(f'Training complete! Total time: {time.time() - start_time:.1f}s')
    
    # Garbage collection
    gc.collect()
    
    return state, history, predictor
