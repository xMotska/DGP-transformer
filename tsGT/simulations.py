"""Simulation utilities for generating predictions."""

import itertools
from typing import Any, Callable, Generator

import numpy as np
import pandas as pd
import tqdm

from inputs import Inputs


def predict_batches(
    predictor: Any,
    weights: Any,
    batches: list,
    n_samples: int
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Generate predictions for all batches.
    
    Args:
        predictor: Predictor object with a predict() method.
        weights: Model weights/parameters.
        batches: List of (series, inputs, targets, mask) tuples.
        n_samples: Number of prediction samples to generate per series.
    
    Yields:
        Tuples of (predictions, ground_truth) with shapes
        (1, n_samples, horizon) and (1, horizon).
    """
    for batch in tqdm.tqdm(list(batches)):
        dataset_batch_size = batch[0].shape[0]  # usually 16
        
        # Process each element in the batch individually
        for start in range(dataset_batch_size):
            subbatch = tuple(item[start:(start + 1)] for item in batch)
            # One ground-truth series. mask.shape [1, series].
            ground_truth_subbatch, _, _, mask = subbatch
            
            # If the mask is all 0s, there is nothing to generate
            # (happens e.g., when the batch is partially completed).
            if np.sum(mask) == 0:
                break

            # Repeat each element `n_samples` times (to make that many predictions).
            repeated_subbatch = tuple(
                np.repeat(item, repeats=n_samples, axis=0) for item in subbatch
            )

            pred_subsubbatches = []
            # Split `n_samples` into batches of size `dataset_batch_size`.
            for substart in range(0, n_samples, dataset_batch_size):
                substop = substart + dataset_batch_size
                subsubbatch = tuple(
                    item[substart:substop] for item in repeated_subbatch
                )
                pred_subsubbatch = predict_batch(predictor, weights, subsubbatch)
                pred_subsubbatches.append(pred_subsubbatch)
            
            pred_subbatch = np.concatenate(pred_subsubbatches)

            horizon_length = pred_subbatch.shape[-1]
            ground_truth_subbatch = ground_truth_subbatch[:, -horizon_length:]
            pred_subbatch = np.reshape(
                pred_subbatch, (1, n_samples, horizon_length)
            )

            yield (pred_subbatch, ground_truth_subbatch)


def predict_batch(
    predictor: Any,
    weights: Any,
    batch: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    """Perform prediction on a single batch.
    
    Args:
        predictor: Predictor object with a predict() method.
        weights: Model weights/parameters.
        batch: Tuple of (series, inputs, targets, mask).
    
    Returns:
        Predictions array of shape (batch_size, horizon_length).
    """
    series, inputs, targets, mask = batch

    assert np.all((mask == 0) | (mask == 1)), "Mask must be binary"
    
    horizon_lengths = np.sum(mask, axis=1)
    assert horizon_lengths.min() == horizon_lengths.max(), (
        "All slices in a batch should have the same horizon length."
    )
    horizon_length = int(np.mean(horizon_lengths))
    _, total_length = mask.shape
    context_length = total_length - horizon_length
    context = series[:, :context_length]
    
    # Transpose inputs from (batch, n_inputs, seq_len) to (batch, seq_len, n_inputs)
    if inputs.ndim == 3 and inputs.shape[1] != inputs.shape[2]:
        # Heuristic: if shape is (batch, small, large), transpose
        if inputs.shape[1] < inputs.shape[2]:
            inputs = np.transpose(inputs, (0, 2, 1))
    
    # For autoregressive prediction, we need inputs for the full horizon
    # The predictor will handle slicing/upsampling internally

    pred = predictor.predict(
        weights=weights,
        context=context,
        horizon_length=horizon_length,
        inputs=inputs,  # Full inputs (context + horizon)
    )
    return pred[:, :horizon_length]


def simulate(
    inputs_fn: Callable[[], Inputs],
    weights: Any,
    predictor: Any,
    dataset: Any,
    n_samples: int,
) -> pd.DataFrame:
    """Run simulation to generate predictions for evaluation.
    
    Args:
        inputs_fn: Callable that returns an Inputs object.
        weights: Model weights/parameters.
        predictor: Predictor object with predict() and optionally
            before_training() methods.
        dataset: Dataset object with eval_horizon, eval_start, eval_end.
        n_samples: Number of prediction samples per time series.
    
    Returns:
        DataFrame with columns:
        - ts_id: Time series index
        - pred_id: Prediction sample index
        - pred_0, pred_1, ...: Predicted values at each horizon step
        - gt_0, gt_1, ...: Ground truth values at each horizon step
        - eval_start, eval_end: Evaluation window timestamps
        - horizon: Horizon length
    """
    horizon = dataset.eval_horizon
    
    # Optional: call before_training hook if predictor has it
    if hasattr(predictor, 'before_training'):
        predictor.before_training(inputs_fn)
    
    # Get inputs object
    inputs = inputs_fn()
    
    # Get evaluation batches
    # Note: In the migrated version, eval_batches() returns a generator
    eval_stream = inputs.eval_batches()
    batches = list(eval_stream)

    # Generate predictions
    # Each element from prediction_generator corresponds to n_samples 
    # for a single time series.
    # prediction_generator yields tuples of (predictions, ground_truth)
    # with shapes (1, n_samples, horizon) and (1, horizon).
    prediction_generator = predict_batches(
        predictor, weights, batches, n_samples
    )

    # Get predictions for all time series
    outcome = list(prediction_generator)
    
    # Number of time series
    N = len(outcome)
    
    # Split into predictions and ground truth
    # preds: (N, 1, n_samples, horizon), gts: (N, 1, horizon)
    preds, gts = zip(*outcome)
    
    # Process predictions: (N, n_samples, horizon) -> (N * n_samples, horizon)
    preds = np.squeeze(preds)
    preds = np.reshape(preds, (-1, horizon))
    
    # Process ground truth: repeat to match predictions
    gts = np.repeat(gts, n_samples, axis=1)  # (N, n_samples, horizon)
    gts = np.concatenate(gts, axis=0)  # (N * n_samples, horizon)
    
    # Combine predictions and ground truth
    outcome = np.concatenate([preds, gts], axis=1)  # (N * n_samples, 2 * horizon)
    
    # Create time series and prediction IDs
    ts_ids, pred_ids = range(N), range(n_samples)
    ts_pred_ids = np.array(list(itertools.product(ts_ids, pred_ids)))
    
    # Concatenate IDs with predictions
    outcome = np.concatenate(
        [ts_pred_ids, outcome], axis=1
    )  # (N * n_samples, 2 + 2 * horizon)
    
    # Create DataFrame
    column_names = (
        ["ts_id", "pred_id"]
        + [f"pred_{i}" for i in range(horizon)]
        + [f"gt_{i}" for i in range(horizon)]
    )
    simulations_df = pd.DataFrame(outcome, columns=column_names)
    
    # Add metadata
    simulations_df["eval_start"] = dataset.eval_start
    simulations_df["eval_end"] = dataset.eval_end
    simulations_df["horizon"] = horizon
    
    return simulations_df.reset_index()
