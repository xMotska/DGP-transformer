"""Index stream generators for training and evaluation data sampling."""

from collections.abc import Generator

import numpy as np


IndexTuple = tuple[int, int, int | None]


def create_full_eval_index_stream(
    dataset: np.ndarray,
    series_length: int
) -> Generator[IndexTuple, None, None]:
    """Pass through the dataset once, in order.
    
    Args:
        dataset: Array of shape (n_series, n_timesteps).
        series_length: Length of each series slice.
    
    Yields:
        Tuples of (series_idx, slice_start, slice_stop).
    """
    for series_idx in range(len(dataset)):
        slice_start = -series_length
        slice_stop = None
        yield (series_idx, slice_start, slice_stop)


def create_random_eval_index_stream(
    dataset: np.ndarray,
    series_length: int,
    rng: np.random.Generator | None = None
) -> Generator[IndexTuple, None, None]:
    """Sample batches indefinitely.
    
    Args:
        dataset: Array of shape (n_series, n_timesteps).
        series_length: Length of each series slice.
        rng: NumPy random generator. If None, uses default.
    
    Yields:
        Tuples of (series_idx, slice_start, slice_stop).
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n_series = len(dataset)
    while True:
        series_idx = rng.integers(0, n_series)
        slice_start = -series_length
        slice_stop = None
        yield (series_idx, slice_start, slice_stop)


def create_eval_index_stream(
    dataset: np.ndarray,
    series_length: int,
    full_eval: bool,
    rng: np.random.Generator | None = None
) -> Generator[IndexTuple, None, None]:
    """Create an evaluation index stream.
    
    Args:
        dataset: Array of shape (n_series, n_timesteps).
        series_length: Length of each series slice.
        full_eval: If True, iterate once in order; if False, sample randomly.
        rng: NumPy random generator for random sampling.
    
    Yields:
        Tuples of (series_idx, slice_start, slice_stop).
    """
    if full_eval:
        stream = create_full_eval_index_stream(dataset, series_length)
    else:
        stream = create_random_eval_index_stream(dataset, series_length, rng)

    yield from stream


def get_weights(dataset: np.ndarray, series_length: int) -> np.ndarray:
    """Compute sampling weights based on data values.
    
    Weights are proportional to the rolling sum of values, with zeros
    at the beginning (before first non-zero) excluded.
    
    Args:
        dataset: Array of shape (n_series, n_timesteps).
        series_length: Length of each series slice.
    
    Returns:
        Weight array of shape (n_series, n_timesteps - series_length + 1).
    """
    n_series = dataset.shape[0]
    first_nonzeros = (dataset != 0).argmax(axis=1)
    
    weights = dataset.copy()
    for i in range(1, series_length):
        weights[:, :-i] += dataset[:, i:]
    weights = weights / series_length + 1
    
    for i, j in zip(range(n_series), first_nonzeros):
        weights[i, :j] = 0
    
    return weights[:, :(-series_length + 1)]


def create_weighted_train_index_stream(
    dataset: np.ndarray,
    series_length: int,
    rng: np.random.Generator | None = None
) -> Generator[tuple[int, int, int], None, None]:
    """Sample training indices weighted by data values.
    
    Args:
        dataset: Array of shape (n_series, n_timesteps).
        series_length: Length of each series slice.
        rng: NumPy random generator. If None, uses default.
    
    Yields:
        Tuples of (series_id, slice_start, slice_stop).
    """
    if rng is None:
        rng = np.random.default_rng()
    
    weights = get_weights(dataset, series_length)
    cum_weights = np.cumsum(weights.flatten())
    ts_len = weights.shape[1]
    
    # Sample indices indefinitely
    while True:
        x = rng.uniform(0, cum_weights[-1])
        idx = np.searchsorted(cum_weights, x)
        series_id = idx // ts_len
        slice_start = idx % ts_len
        slice_stop = slice_start + series_length
        yield series_id, slice_start, slice_stop


def create_uniform_train_index_stream(
    dataset: np.ndarray,
    series_length: int,
    rng: np.random.Generator | None = None
) -> Generator[tuple[int, int, int], None, None]:
    """Sample training indices uniformly at random.
    
    Args:
        dataset: Array of shape (n_series, n_timesteps).
        series_length: Length of each series slice.
        rng: NumPy random generator. If None, uses default.
    
    Yields:
        Tuples of (series_idx, slice_start, slice_stop).
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n_series = len(dataset)
    max_start = dataset.shape[1] - series_length
    
    while True:
        series_idx = rng.integers(0, n_series)
        slice_start = rng.integers(0, max_start)
        slice_stop = slice_start + series_length
        yield (series_idx, slice_start, slice_stop)


def create_train_index_stream(
    dataset: np.ndarray,
    series_length: int,
    weighted_sampling: bool,
    rng: np.random.Generator | None = None
) -> Generator[tuple[int, int, int], None, None]:
    """Create a training index stream.
    
    Args:
        dataset: Array of shape (n_series, n_timesteps).
        series_length: Positive int, length of each slice.
        weighted_sampling: If True, sample proportional to sum of values;
            if False, sample uniformly.
        rng: NumPy random generator. If None, uses default.
    
    Yields:
        Tuples of (series_idx, slice_start, slice_stop).
    """
    if weighted_sampling:
        stream = create_weighted_train_index_stream(dataset, series_length, rng)
    else:
        stream = create_uniform_train_index_stream(dataset, series_length, rng)

    yield from stream
