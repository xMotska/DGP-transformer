"""Data input pipelines for training and evaluation."""

from collections.abc import Callable, Generator
from dataclasses import dataclass
from functools import partial
from typing import Any

import gin
import numpy as np

from datasets import Dataset
from index_streams import create_eval_index_stream, create_train_index_stream


# Type aliases
BatchType = list[np.ndarray]  # [series, inputs, targets, masks]
StreamFn = Callable[[], Generator[BatchType, None, None]]


@dataclass
class Inputs:
    """Container for train and eval data streams.
    
    Replacement for trax.data.inputs.Inputs.
    
    Attributes:
        train_stream: Callable that returns a generator of training batches.
        eval_stream: Callable that returns a generator of evaluation batches.
    """
    train_stream: StreamFn
    eval_stream: StreamFn
    
    def train_batches(self) -> Generator[BatchType, None, None]:
        """Get training batch generator."""
        return self.train_stream()
    
    def eval_batches(self) -> Generator[BatchType, None, None]:
        """Get evaluation batch generator."""
        return self.eval_stream()


def shuffle_decorator(
    slice_stream: Generator,
    context_length: int,
    rng: np.random.Generator | None = None
) -> Generator:
    """Decorator that shuffles the context portion of each sample.
    
    Args:
        slice_stream: Generator yielding (series, inp, target, mask) tuples.
        context_length: Number of elements to shuffle at the start.
        rng: NumPy random generator. If None, uses default.
    
    Yields:
        Shuffled (series, inp, target, mask) tuples.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    def shuffle_fn(
        series: np.ndarray,
        inp: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        perm = rng.permutation(context_length)
        
        def permute_prefix(arr: np.ndarray) -> np.ndarray:
            pre = arr[:len(perm)][perm]
            return np.concatenate([pre, arr[len(perm):]])
        
        return (permute_prefix(series), permute_prefix(inp), 
                permute_prefix(target), mask)
    
    return (shuffle_fn(*sl) for sl in slice_stream)


def slice_stream(
    index_stream: Generator,
    dataset: np.ndarray,
    inputs_fn: Callable[[int, int, int], np.ndarray],
    eval_horizon: int
) -> Generator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
    """Generate slices from dataset based on index stream.
    
    Args:
        index_stream: Generator of (series_idx, start, end) triples.
        dataset: Array of shape (n_series, n_timesteps).
        inputs_fn: Function (series_idx, start, end) -> array of shape 
            (n_inputs, end - start).
        eval_horizon: Number of last timesteps to mask for evaluation.
    
    Yields:
        Tuples of (series, inp, target, mask).
    """
    for sample in index_stream:
        series_idx, slice_start, slice_stop = sample
        series = dataset[series_idx][slice_start:slice_stop]
        inp = inputs_fn(series_idx, slice_start, slice_stop)
        mask = np.zeros_like(series)
        if eval_horizon > 0:
            mask[-eval_horizon:] = 1
        yield (series, inp, series, mask)


def minibatch_stream(
    slice_stream: Generator,
    batch_size: int,
    drop_remainder: bool = False
) -> Generator[BatchType, None, None]:
    """Aggregate samples from slice_stream into batches.
    
    Args:
        slice_stream: Generator of (series, inp, target, mask) tuples.
        batch_size: Number of samples per batch.
        drop_remainder: If True, drop the last incomplete batch.
    
    Yields:
        Lists of [series, inputs, targets, masks] arrays, each with 
        shape (batch_size, ...).
    """
    batch: list[list[np.ndarray]] = [[], [], [], []]
    
    for sample in slice_stream:
        for batch_comp, slice_comp in zip(batch, sample):
            batch_comp.append(slice_comp)

        if len(batch[0]) == batch_size:
            yield [np.stack(batch_comp) for batch_comp in batch]
            batch = [[], [], [], []]

    # Output the last, partially complete batch
    if len(batch[0]) > 0 and not drop_remainder:
        padding = batch_size - len(batch[0])
        yield [
            np.pad(
                np.stack(batch_comp),
                ((0, padding),) + ((0, 0),) * (np.stack(batch_comp).ndim - 1),
                constant_values=0
            )
            for batch_comp in batch
        ]


@gin.configurable(module='code.inputs')
def CreateInputs(
    dataset: Dataset,
    batch_size: int = gin.REQUIRED,
    series_length: int = gin.REQUIRED,
    weighted_sampling: bool = gin.REQUIRED,
    full_eval: bool = False,
    shuffle: bool = False,
    rng: np.random.Generator | None = None,
) -> Inputs:
    """Create training and evaluation input streams.
    
    Args:
        dataset: Dataset object containing train/eval data.
        batch_size: Number of samples per batch.
        series_length: Length of each time series slice.
        weighted_sampling: If True, sample training data weighted by values.
        full_eval: If True, iterate through all eval data once; 
            if False, sample randomly.
        shuffle: If True, shuffle the context portion of eval samples.
        rng: NumPy random generator for reproducibility.
    
    Returns:
        Inputs object with train_stream and eval_stream callables.
    """
    assert dataset.train_data.shape[1] >= series_length, (
        f'Series length ({series_length}) should be <= '
        f'train dataset length ({dataset.train_data.shape[1]}).'
    )

    assert dataset.eval_data.shape[1] >= series_length, (
        f'Series length ({series_length}) should be <= '
        f'eval dataset length ({dataset.eval_data.shape[1]}).'
    )

    train_data = dataset.train_data
    eval_data = dataset.eval_data
    eval_horizon = dataset.eval_horizon
    covariates = dataset.covariates

    def make_train_stream() -> Generator[BatchType, None, None]:
        """Create a fresh training stream."""
        train_index_stream = create_train_index_stream(
            train_data, series_length, weighted_sampling, rng
        )
        train_slice = slice_stream(
            train_index_stream, train_data, covariates, 0
        )
        return minibatch_stream(train_slice, batch_size)

    def make_eval_stream() -> Generator[BatchType, None, None]:
        """Create a fresh evaluation stream."""
        eval_index_stream = create_eval_index_stream(
            eval_data, series_length, full_eval, rng
        )
        eval_slice = slice_stream(
            eval_index_stream, eval_data, covariates, eval_horizon
        )
        
        if shuffle:
            context_length = series_length - eval_horizon
            eval_slice = shuffle_decorator(eval_slice, context_length, rng)
        
        return minibatch_stream(eval_slice, batch_size)

    return Inputs(
        train_stream=make_train_stream,
        eval_stream=make_eval_stream,
    )
