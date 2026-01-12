"""Base class for time series prediction models."""

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from normalization import Normalizer


class TimeSeriesPredictor:
    """Training and prediction interface for time series modeling."""

    def __init__(
        self,
        model_body_fn: Callable,
        accelerate_predict_model: bool,
        normalization: str | None,
        normalization_regularizer: float,
        context_type: np.dtype,
        input_vocab_sizes: list[int] | None,
        decoder_fn: Callable,
    ):
        """Initialize TimeSeriesPredictor.

        Args:
            model_body_fn: Function that returns a Flax model body.
                The model should have input shape [batch_size, n_timesteps, d_in]
                and output shape [..., d_out].
            accelerate_predict_model: Whether to JIT the prediction model.
                Should be turned off for testing/debugging.
            normalization: Normalization method name. See Normalizer.from_name()
                for options ('per_ts', 'per_batch', 'causal', None).
            normalization_regularizer: Constant added to divisor to prevent
                division by zero.
            context_type: Data type for context (e.g., np.float32).
            input_vocab_sizes: Vocab sizes for auxiliary inputs.
                If None, auxiliary inputs are skipped.
                Example: [30, 7, 24, 512] for day-of-month, day-of-week, hour, series_id.
            decoder_fn: Function (model_body, **kwargs) -> decoder model.
        """
        self._model_body_fn = model_body_fn
        self._accelerate_predict_model = accelerate_predict_model
        self._predict_model = None
        self._predict_model_jitted = None

        self._normalizer: Normalizer = Normalizer.from_name(
            name=normalization,
            regularizer=normalization_regularizer,
        )

        self._n_inputs = len(input_vocab_sizes) if input_vocab_sizes is not None else 0
        self._input_vocab_sizes = input_vocab_sizes

        self._init_state = None
        self._init_params = None
        self._batch_size = None
        self._context_type = context_type
        self._decoder_fn = decoder_fn

    def init_state(self, batch_size: int, rng: jax.Array | None = None):
        """Initialize model state for a given batch size.
        
        Args:
            batch_size: Batch size for initialization.
            rng: Random key for initialization.
        
        Returns:
            Initial model state/variables.
        """
        if rng is None:
            rng = jax.random.key(0)
        
        if self._init_state is None:
            self._predict_model = None
            model = self.predict_model
            
            # Create dummy inputs for initialization
            dummy_context = jnp.ones((batch_size, 1), dtype=self._context_type)
            dummy_inputs = jnp.ones((batch_size, 1, self._n_inputs), dtype=jnp.int32)
            
            # Initialize model
            variables = model.init(
                {'params': rng, 'dropout': rng},
                dummy_context,
                dummy_inputs,
                deterministic=True,
                decode=True,
            )
            
            self._init_params = variables.get('params', {})
            self._init_state = variables
            self._batch_size = batch_size
            
        elif batch_size != self._batch_size:
            raise ValueError(
                f"Model was initialized with batch size {self._batch_size}, "
                f"but a batch of size {batch_size} was received."
            )
        
        return self._init_state

    @property
    def predict_model(self) -> nn.Module:
        """Get the prediction model (lazily created).
        
        Returns:
            Flax model for prediction.
        """
        if self._predict_model is None:
            self._predict_model = self.make_predict_model()
        return self._predict_model

    @property
    def predict_fn(self) -> Callable:
        """Get the (optionally JIT-compiled) prediction function.
        
        Returns:
            Function for model forward pass.
        """
        if self._predict_model_jitted is None:
            model = self.predict_model
            
            def apply_fn(variables, context, inputs, deterministic=True, decode=True):
                return model.apply(
                    variables,
                    context,
                    inputs,
                    deterministic=deterministic,
                    decode=decode,
                )
            
            if self._accelerate_predict_model:
                self._predict_model_jitted = jax.jit(apply_fn)
            else:
                self._predict_model_jitted = apply_fn
        
        return self._predict_model_jitted

    def make_train_eval_model(self, mode: str) -> nn.Module:
        """Create a new model for training or evaluation.
        
        Args:
            mode: Either 'train' or 'eval'.
        
        Returns:
            Flax model configured for the specified mode.
        """
        raise NotImplementedError

    def make_predict_model(self) -> nn.Module:
        """Create a new model for autoregressive prediction.
        
        Returns:
            Flax model configured for prediction/decoding.
        """
        model_body = self._model_body_fn()
        return self._decoder_fn(model_body=model_body, mode="predict")

    def make_loss(self) -> Callable:
        """Create a loss function for model training.
        
        Returns:
            Loss function with signature (logits, targets, mask) -> scalar.
        """
        raise NotImplementedError

    def before_training(self, inputs: Callable) -> None:
        """Called before training begins.

        Can be used to calculate statistics on the dataset (e.g., for
        quantile-based serialization).

        Args:
            inputs: Callable returning an Inputs object.
        """
        pass

    def predict(
        self,
        weights: Any,
        context: np.ndarray,
        inputs: np.ndarray,
        horizon_length: int,
    ) -> np.ndarray:
        """Predict a batch of time series up to a given horizon.

        Args:
            weights: Model weights/parameters (Flax params dict).
            context: Array of shape (batch_size, context_length)
                containing past values of the time series.
            inputs: Array of shape (batch_size, context_length + horizon_length, d_input)
                containing auxiliary inputs for the entire sequence.
            horizon_length: Number of timesteps to predict.

        Returns:
            Array of predicted values with shape (batch_size, horizon_length).
        """
        raise NotImplementedError

    def get_metrics(self) -> dict:
        """Get any custom metrics tracked by the predictor.
        
        Returns:
            Dictionary of metric names to values.
        """
        return {}

    @property
    def normalizer(self) -> Normalizer:
        """Get the normalizer used by this predictor."""
        return self._normalizer
