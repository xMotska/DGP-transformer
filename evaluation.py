"""Evaluation utilities for time series predictions."""

import os
import pickle
import time
from collections.abc import Callable, Sequence

import gin
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy.stats as sps
import seaborn as sns
import tqdm
from matplotlib.gridspec import GridSpec

# Axis indices for array operations
SERIES_AXIS = -3
PREDS_AXIS = -2
HORIZON_AXIS = -1
WINDOWS_AXIS = 0

# Legacy aliases for backward compatibility
series_axis = SERIES_AXIS
preds_axis = PREDS_AXIS
horizon_axis = HORIZON_AXIS
windows_axis = WINDOWS_AXIS


def violin_stats(
    X: np.ndarray,
    method: Callable,
    points: int = 100,
    quantiles: np.ndarray | None = None
) -> list[dict]:
    """
    Return a list of dictionaries of data for drawing violin plots.
    
    See the Returns section below to view the required keys of the dictionary.
    Users can skip this function and pass a user-defined set of dictionaries
    with the same keys to `~.axes.Axes.violinplot` instead of using Matplotlib
    to do the calculations.
    
    Parameters
    ----------
    X : array-like
        Sample data that will be used to produce the gaussian kernel density
        estimates. Must have 2 or fewer dimensions.
    method : callable
        The method used to calculate the kernel density estimate for each
        column of data. When called via `method(v, coords)`, it should
        return a vector of the values of the KDE evaluated at the values
        specified in coords.
    points : int, default=100
        Defines the number of points to evaluate each of the gaussian kernel
        density estimates at.
    quantiles : array-like, default=None
        Defines (if not None) a list of floats in interval [0, 1] for each
        column of data, which represents the quantiles that will be rendered
        for that column of data. Must have 2 or fewer dimensions. 1D array
        will be treated as a singleton list containing them.
    
    Returns
    -------
    vpstats : list of dict
        A list of dictionaries containing the results for each column of data.
        The dictionaries contain at least the following:
        - coords: A list of scalars containing the coordinates this particular
          kernel density estimate was evaluated at.
        - vals: A list of scalars containing the values of the kernel density
          estimate at each of the coordinates given in `coords`.
        - mean: The mean value for this column of data.
        - median: The median value for this column of data.
        - min: The minimum value for this column of data.
        - max: The maximum value for this column of data.
        - quantiles: The quantile values for this column of data.
    """
    # List of dictionaries describing each of the violins.
    vpstats = []

    # Want X to be a list of data sequences
    X = matplotlib.cbook._reshape_2D(X, "X")

    # Want quantiles to be as the same shape as data sequences
    if quantiles is not None and len(quantiles) != 0:
        quantiles = matplotlib.cbook._reshape_2D(quantiles, "quantiles")
    # Else, mock quantiles if is none or empty
    else:
        quantiles = [[]] * np.shape(X)[0]

    # quantiles should have the same size as dataset
    if np.shape(X)[:1] != np.shape(quantiles)[:1]:
        raise ValueError(
            "List of violinplot statistics and quantiles values"
            " must have the same length"
        )

    ci = 0.995
    # Zip x and quantiles
    for x, q in zip(X, quantiles):
        # Dictionary of results for this distribution
        stats = {}

        # Calculate basic stats for the distribution
        min_val = np.min(x)
        max_val = np.max(x)
        quantile_val = np.percentile(x, 100 * q)

        # Evaluate the kernel density estimate
        q_05 = np.quantile(x, 1 - ci)
        q_95 = np.quantile(x, ci)
        coords = np.linspace(q_05, q_95, points)
        stats["vals"] = method(x, coords)
        stats["coords"] = coords

        # Store additional statistics for this distribution
        stats["mean"] = np.mean(x)
        stats["median"] = np.median(x)
        stats["min"] = min_val
        stats["max"] = max_val
        stats["quantiles"] = np.atleast_1d(quantile_val)

        # Append to output
        vpstats.append(stats)

    return vpstats


# Monkey-patch matplotlib's violin_stats
matplotlib.cbook.violin_stats = violin_stats


def plot_trajectory(
    tsi: np.ndarray,
    gt: np.ndarray,
    window: int,
    series: int,
    output_dir: str,
    ci: float = 0.95,
    suffix: str = ""
) -> None:
    """Plot a single trajectory with violin plots.
    
    Args:
        tsi: Predictions of shape [predictions, horizon].
        gt: Ground truth of shape [horizon].
        window: Window index.
        series: Series index.
        output_dir: Output directory for saving plots.
        ci: Confidence interval level.
        suffix: Filename suffix.
    """
    H = tsi.shape[-1]
    x_labels = list(range(H))

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot the ground truth data.
    ax.plot(gt, color="red", ls="--")
    # Get the median of the prediction.
    pred_mean = np.median(tsi, axis=0)  # [horizon, ]
    # Plot the median of the prediction.
    ax.plot(pred_mean, color="green")
    # Shade out the [q(1-ci), q(ci)] area.
    lower = np.quantile(tsi, q=1 - ci, axis=0)
    upper = np.quantile(tsi, q=ci, axis=0)
    ax.fill_between(x=range(H), y1=lower, y2=upper, alpha=0.3)

    means = np.mean(tsi, axis=0)

    v = ax.violinplot(
        dataset=tsi,
        positions=list(range(H)),
        widths=0.7,
        showmeans=False,
        showextrema=False,
        showmedians=False,
        points=100,
    )
    # Plot mean tick.
    for mu, p in zip(means, range(H)):
        ax.plot([p - 0.15, p], [mu] * 2, c="black")
    # Only keep one half of the violin plot.
    for b in v["bodies"]:
        # Get the center.
        m = np.mean(b.get_paths()[0].vertices[:, 0])
        # Plot the right half: modify the paths to not go further left than m.
        # To plot the left half set `-np.inf, m`.
        b.get_paths()[0].vertices[:, 0] = np.clip(
            b.get_paths()[0].vertices[:, 0], m, np.inf
        )
        b.set_color("blue")
        b.set_alpha(0.8)
    ax.set_ylabel(f"{int(series)}")
    
    trajectories_dir = os.path.join(output_dir, "trajectories")
    if not os.path.exists(trajectories_dir):
        os.makedirs(trajectories_dir, exist_ok=True)
    plt.savefig(
        os.path.join(trajectories_dir, f"trajectory_{suffix}_{window}_{series}.png"),
        bbox_inches="tight",
    )
    plt.close()


def plot_trajectories(
    preds: np.ndarray,
    targets: np.ndarray,
    draw_series_ids: Sequence[int] | None,
    horizon: int,
    output_dir: str,
    suffix: str = ""
) -> None:
    """Plot multiple trajectories with violin plots.
    
    Args:
        preds: Predictions of shape [windows, series, predictions, horizon].
        targets: Ground truth of shape [windows, series, 1, horizon].
        draw_series_ids: Series IDs to draw. None=all, []=none.
        horizon: Prediction horizon.
        output_dir: Output directory.
        suffix: Filename suffix.
    """
    style = {
        "axes.labelsize": 20,
        "legend.fontsize": 14,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "axes.titlesize": 20,
        "font.serif": "Times New Roman",
    }
    plt.rcParams.update(style)

    num_windows = preds.shape[WINDOWS_AXIS]
    num_series = preds.shape[SERIES_AXIS]

    if draw_series_ids is None:
        draw_series_ids = range(num_series)

    for window in tqdm.tqdm(range(num_windows)):
        for series in draw_series_ids:
            tsi = preds[window, series, :, :horizon]
            gt = targets[window, series, 0, :horizon]
            plot_trajectory(
                tsi, gt, window, series, output_dir, suffix=suffix
            )


def plot_trajectories_histograms(
    preds: np.ndarray,
    targets: np.ndarray,
    draw_series_ids: Sequence[int] | None,
    horizon: int,
    var_level: float,
    output_dir: str
) -> None:
    """Plot trajectories with individual histograms.
    
    Args:
        preds: Predictions of shape [windows, series, predictions, horizon].
        targets: Ground truth of shape [windows, series, 1, horizon].
        draw_series_ids: Series IDs to draw.
        horizon: Prediction horizon.
        var_level: VaR level for plotting.
        output_dir: Output directory.
    """
    style = {
        "axes.labelsize": 20,
        "legend.fontsize": 14,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "axes.titlesize": 20,
        "font.serif": "Times New Roman",
    }
    plt.rcParams.update(style)

    num_windows = preds.shape[WINDOWS_AXIS]
    num_series = preds.shape[SERIES_AXIS]

    if draw_series_ids is None:
        draw_series_ids = range(num_series)

    histograms_dir = os.path.join(output_dir, "histograms")
    if not os.path.exists(histograms_dir):
        os.makedirs(histograms_dir, exist_ok=True)

    for window in tqdm.tqdm(range(num_windows)):
        for series in draw_series_ids:
            tsi = preds[window, series, :, :horizon]
            gt = targets[window, series, 0, :horizon]
            
            fig, axes = plt.subplots(1, horizon, figsize=(3 * horizon, 3))
            for h in range(horizon):
                ax = axes[h] if horizon > 1 else axes
                ax.hist(tsi[:, h], bins=30, alpha=0.7, density=True)
                ax.axvline(gt[h], color='red', linestyle='--', label='Ground Truth')
                ax.axvline(np.quantile(tsi[:, h], var_level), color='blue', 
                          linestyle=':', label=f'VaR {var_level}')
                ax.set_title(f'h={h}')
                if h == 0:
                    ax.legend()
            
            plt.tight_layout()
            plt.savefig(
                os.path.join(histograms_dir, f"histogram_{window}_{series}.png"),
                bbox_inches="tight",
            )
            plt.close()


def reduce(
    data: np.ndarray,
    reduce_functions: Sequence[Callable],
    axis: int
) -> np.ndarray:
    """Apply multiple reduction functions along an axis.
    
    Args:
        data: Input array.
        reduce_functions: List of functions to apply.
        axis: Axis to reduce over.
    
    Returns:
        Stacked results of shape [..., len(reduce_functions), ...].
    """
    results = [fn(data, axis=axis) for fn in reduce_functions]
    return np.stack(results, axis=axis)


# Metric functions
def mean(preds: np.ndarray, targets: np.ndarray, **kwargs) -> np.ndarray:
    """Compute mean of predictions."""
    return np.mean(preds, axis=PREDS_AXIS, keepdims=True)


def std(preds: np.ndarray, targets: np.ndarray, **kwargs) -> np.ndarray:
    """Compute standard deviation of predictions."""
    return np.std(preds, axis=PREDS_AXIS, keepdims=True)


def mse(preds: np.ndarray, targets: np.ndarray, **kwargs) -> np.ndarray:
    """Compute mean squared error."""
    return np.mean((preds - targets) ** 2, axis=PREDS_AXIS, keepdims=True)


def mad(preds: np.ndarray, targets: np.ndarray, **kwargs) -> np.ndarray:
    """Compute mean absolute deviation."""
    return np.mean(np.abs(preds - targets), axis=PREDS_AXIS, keepdims=True)


def quantile_loss(
    preds: np.ndarray,
    targets: np.ndarray,
    alphas: Sequence[float] = (0.5, 0.9),
    **kwargs
) -> np.ndarray:
    """Compute quantile loss (pinball loss).
    
    Args:
        preds: Predictions of shape [..., predictions, horizon].
        targets: Ground truth of shape [..., 1, horizon].
        alphas: Quantile levels.
    
    Returns:
        Quantile losses of shape [..., len(alphas), horizon].
    """
    losses = []
    for alpha in alphas:
        q = np.quantile(preds, alpha, axis=PREDS_AXIS, keepdims=True)
        error = targets - q
        loss = np.where(error >= 0, alpha * error, (alpha - 1) * error)
        losses.append(loss.squeeze(axis=PREDS_AXIS))
    return np.stack(losses, axis=PREDS_AXIS)


def crps(preds: np.ndarray, targets: np.ndarray, **kwargs) -> np.ndarray:
    """Compute Continuous Ranked Probability Score.
    
    CRPS measures the quality of probabilistic forecasts.
    
    Args:
        preds: Predictions of shape [..., n_samples, horizon].
        targets: Ground truth of shape [..., 1, horizon].
    
    Returns:
        CRPS values of shape [..., 1, horizon].
    """
    # Sort predictions along the samples axis
    sorted_preds = np.sort(preds, axis=PREDS_AXIS)
    n = preds.shape[PREDS_AXIS]
    
    # Compute empirical CDF values: [1/n, 2/n, ..., 1]
    ecdf = np.arange(1, n + 1) / n
    
    # Reshape ecdf to broadcast correctly
    # For 3D input [series, samples, horizon], we need ecdf shape [1, samples, 1]
    ecdf_shape = [1] * preds.ndim
    ecdf_shape[PREDS_AXIS] = n
    ecdf = ecdf.reshape(ecdf_shape)
    
    # Indicator function: 1 if sorted_pred <= target, else 0
    indicator = (sorted_preds <= targets).astype(float)
    
    # CRPS = integral of (F(x) - 1[x >= y])^2 dx
    # Approximate using trapezoidal rule
    diff = np.diff(sorted_preds, axis=PREDS_AXIS)
    
    # Compute integrand (need to slice off last element to match diff shape)
    ecdf_sliced = np.take(ecdf, np.arange(n - 1), axis=PREDS_AXIS)
    indicator_sliced = np.take(indicator, np.arange(n - 1), axis=PREDS_AXIS)
    integrand = (ecdf_sliced - indicator_sliced) ** 2
    
    crps_val = np.sum(integrand * diff, axis=PREDS_AXIS, keepdims=True)
    return crps_val


def compute_metrics(
    metrics: Sequence[Callable],
    metric_names: Sequence[str],
    sim_data: list[tuple[np.ndarray, np.ndarray]],
    train_data: np.ndarray,
    reduce_series: Callable | None = None,
    reduce_horizon: tuple[Callable, ...] | None = None,
    normalize: str | None = None,
    alphas: Sequence[float] = (0.5, 0.75, 0.9, 0.95, 0.99),
) -> dict:
    """Compute all metrics for simulation data.
    
    Args:
        metrics: List of metric functions.
        metric_names: Names for each metric.
        sim_data: List of (predictions, targets) tuples.
        train_data: Training data for normalization.
        reduce_series: Function to reduce over series axis.
        reduce_horizon: Tuple of functions to reduce over horizon axis.
        normalize: Normalization method ('mean', 'std', or None).
        alphas: Quantile levels for quantile-based metrics.
    
    Returns:
        Dictionary mapping metric names to computed values.
    """
    results = {}
    
    for metric_fn, name in zip(metrics, metric_names):
        metric_values = []
        for preds, targets in sim_data:
            # Normalize if requested
            if normalize == 'mean':
                scale = np.mean(np.abs(train_data), axis=-1, keepdims=True)
                preds = preds / (scale + 1e-8)
                targets = targets / (scale + 1e-8)
            elif normalize == 'std':
                scale = np.std(train_data, axis=-1, keepdims=True)
                preds = preds / (scale + 1e-8)
                targets = targets / (scale + 1e-8)
            
            # Compute metric
            if name == 'quantile_loss':
                value = metric_fn(preds, targets, alphas=alphas)
            else:
                value = metric_fn(preds, targets)
            metric_values.append(value)
        
        # Stack across windows
        stacked = np.stack(metric_values, axis=WINDOWS_AXIS)
        
        # Apply reductions
        if reduce_series is not None:
            stacked = reduce_series(stacked)
        if reduce_horizon is not None:
            reduced_horizons = []
            for fn in reduce_horizon:
                reduced_horizons.append(fn(stacked))
            stacked = np.stack(reduced_horizons, axis=-1)
        
        results[name] = stacked
    
    return results


def compute_backtest_metrics(
    quantiles: np.ndarray,
    targets: np.ndarray,
    alphas: Sequence[float],
    output_dir: str
) -> dict:
    """Compute backtesting metrics (POF test p-values).
    
    Args:
        quantiles: Predicted quantiles of shape [windows, series, quantiles, horizon].
        targets: Ground truth of shape [windows, series, 1, horizon].
        alphas: Quantile levels.
        output_dir: Output directory for plots.
    
    Returns:
        Dictionary with p-values.
    """
    # Proportion of failures test
    n_windows = quantiles.shape[WINDOWS_AXIS]
    n_series = quantiles.shape[SERIES_AXIS]
    n_alphas = len(alphas)
    n_horizon = quantiles.shape[HORIZON_AXIS]
    
    p_values = np.zeros((n_series, n_alphas, n_horizon))
    
    for s in range(n_series):
        for a_idx, alpha in enumerate(alphas):
            for h in range(n_horizon):
                # Count exceedances
                q = quantiles[:, s, a_idx, h]
                t = targets[:, s, 0, h]
                exceedances = np.sum(t > q)
                
                # Binomial test
                if n_windows > 0:
                    p_val = sps.binomtest(
                        exceedances, n_windows, 1 - alpha
                    ).pvalue
                    p_values[s, a_idx, h] = p_val
    
    return {"pof_p_values": p_values}


def probe_p_values(
    p_values: np.ndarray,
    probe_at: tuple[int, ...],
    threshold: float = 0.05
) -> dict:
    """Compute fraction of p-values above threshold at probe points.
    
    Args:
        p_values: P-values of shape [series, alphas, horizon].
        probe_at: Horizon indices to probe.
        threshold: Significance threshold.
    
    Returns:
        Dictionary with probe results.
    """
    results = {}
    for h_idx in probe_at:
        p_at_h = p_values[:, :, h_idx]
        frac_above = np.mean(p_at_h > threshold)
        results[f"frac_p_above_{threshold}_at_h{h_idx}"] = frac_above
    return results


def reduce_data_and_compute_quantiles(
    sim_data: list[tuple[np.ndarray, np.ndarray]],
    reduce_series: Callable | None,
    reduce_horizon: tuple[Callable, ...] | None,
    alphas: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce data and compute quantiles.
    
    Args:
        sim_data: List of (predictions, targets) tuples.
        reduce_series: Series reduction function.
        reduce_horizon: Horizon reduction functions.
        alphas: Quantile levels.
    
    Returns:
        Tuple of (predictions, targets, quantiles) arrays.
    """
    preds_list = []
    targets_list = []
    quantiles_list = []
    
    for preds, targets in sim_data:
        preds_list.append(preds)
        targets_list.append(targets)
        
        # Compute quantiles
        quantile_functions = [lambda x, q=alpha: np.quantile(x, q, axis=PREDS_AXIS) 
                            for alpha in alphas]
        preds_quantiles = np.stack(
            [fn(preds) for fn in quantile_functions], 
            axis=PREDS_AXIS
        )
        quantiles_list.append(preds_quantiles)
    
    return np.stack(preds_list), np.stack(targets_list), np.stack(quantiles_list)


def process_simulation_data(
    dataframes: list[pd.DataFrame],
    average_preds: bool = False
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    """Process simulation dataframes into numpy arrays.
    
    Args:
        dataframes: List of simulation result dataframes.
        average_preds: Whether to average predictions.
    
    Returns:
        Tuple of (simulation data list, horizon).
    """
    horizon = len([col for col in dataframes[0].columns if "pred" in col]) - 1
    N = len(dataframes[0]["ts_id"].unique())
    H = horizon
    P = len(dataframes[0]["pred_id"].unique())

    def extract_preds_targets(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        df_numpy = df.to_numpy()
        # df.columns = ['Unnamed: 0', 'ts_id', 'pred_id', 'pred_0', ..., 'pred_H',
        # 'gt_0', ..., 'gt_H', 'eval_start', 'eval_end', 'horizon']
        preds = df_numpy[:, 3 : -(H + 3)].reshape((N, P, H))
        preds = preds.astype(np.float64)
        targets = df_numpy[:, -(H + 3) : -3].reshape((N, P, H))[:, :1, :]
        targets = targets.astype(np.float64)

        if average_preds:
            preds = np.mean(preds, axis=PREDS_AXIS, keepdims=True)

        return preds, targets

    sim_data = [extract_preds_targets(df) for df in dataframes]
    return sim_data, horizon


@gin.configurable(module="code.evaluation")
def evaluate(
    train_data: np.ndarray,
    sim_dataframes: list[pd.DataFrame],
    metrics: tuple[Callable, ...] = (mean, std, mse, mad, quantile_loss, crps),
    metric_names: tuple[str, ...] = (
        "mean",
        "std",
        "mse",
        "mad",
        "quantile_loss",
        "crps",
    ),
    reduce_series: Callable | str | None = None,
    reduce_horizon: tuple[Callable, ...] | None = None,
    alphas: tuple[float, ...] = (0.5, 0.75, 0.9, 0.95, 0.99),
    save_preds: bool = False,
    output_dir: str = "./out",
    normalize: str | None = None,
    average_preds: bool = False,
) -> None:
    """Compute metrics, save them to a file, and draw plots.

    Shapes:
        preds: (windows, series, predictions, horizon)
        targets: (windows, series, 1, horizon)
        non-quantile metrics: (windows, series, 1, horizon)
        quantile metrics: (windows, series, 2, horizon)
        p_values: (series, 2, horizon)

    Args:
        train_data: Training data for normalization.
        sim_dataframes: List of simulation result dataframes.
        metrics: Tuple of metric functions.
        metric_names: Names for each metric.
        reduce_series: Series reduction function (or string for gin).
        reduce_horizon: Horizon reduction functions.
        alphas: Quantile levels.
        save_preds: Whether to save predictions (can be very large!).
        output_dir: Output directory.
        normalize: Normalization method.
        average_preds: Whether to average predictions.
    """
    # Handle gin string evaluation
    if isinstance(reduce_series, str):
        reduce_series = eval(reduce_series)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    start = time.time()
    sim_data, horizon = process_simulation_data(
        sim_dataframes,
        average_preds=average_preds,
    )
    print(f"Finished processing {len(sim_dataframes)} experiments in {time.time() - start:.2f} sec.")

    # Metrics and predictions
    start = time.time()

    if reduce_horizon is not None:
        reduce_horizon = tuple(
            [lambda x, idx=i: x[..., idx] for i in range(horizon)]
        ) + reduce_horizon

    print("Computing metrics and predictions...")
    results = compute_metrics(
        metrics=metrics,
        metric_names=metric_names,
        sim_data=sim_data,
        train_data=train_data,
        reduce_series=reduce_series,
        reduce_horizon=reduce_horizon,
        normalize=normalize,
        alphas=alphas,
    )
    end = time.time()
    print(f"Finished computing metrics in {end - start:.2f} sec.")

    preds_arr, targets_arr, quantiles_arr = reduce_data_and_compute_quantiles(
        sim_data=sim_data,
        reduce_series=reduce_series,
        reduce_horizon=reduce_horizon,
        alphas=alphas,
    )

    # Backtest metrics (p-values and plots)
    start = time.time()
    print("Plotting p-value strips...")
    p_values = compute_backtest_metrics(
        quantiles=quantiles_arr,
        targets=targets_arr,
        alphas=alphas,
        output_dir=output_dir,
    )
    results.update(p_values)

    # Compute fraction of p-values above 5% probed at short, mid, and long horizons
    print("Computing statistics for p_values...")
    freq_p_values = probe_p_values(
        results["pof_p_values"],
        probe_at=(0, horizon // 2, horizon - 1),
        threshold=0.05,
    )
    results.update(freq_p_values)

    print("Saving the resulting dict...")
    results_dir = os.path.join(output_dir, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    filename = os.path.join(results_dir, "results.pkl")

    results["preds_quantiles"] = quantiles_arr
    if save_preds:
        print("Saving the predictions...")
        results["preds"] = preds_arr
        results["targets"] = targets_arr

    with open(filename, "wb") as file:
        pickle.dump(results, file, protocol=4)

    end = time.time()
    print(f"Finished saving in {end - start:.2f} sec.")

    # Identify series that did not pass backtest
    pof_predictions = results["pof_p_values"]
    frequency_of_bad_p_values = np.mean(
        pof_predictions < 0.05, axis=HORIZON_AXIS, keepdims=True
    )
    num_series = pof_predictions.shape[0]  # SERIES is first after reduction
    num_quantiles = pof_predictions.shape[1]
    
    # Create a priority queue with the worst p_values
    queue = []
    for series in range(num_series):
        for idx in range(num_quantiles):
            queue.append((-frequency_of_bad_p_values[series, idx, 0], series))
    queue.sort()
    
    # Get top 5 worst and best series
    bad_indices = []
    for p, v in queue:
        if v not in bad_indices:
            bad_indices.append(v)
        if len(bad_indices) >= 5:
            break

    good_indices = []
    for p, v in reversed(queue):
        if v not in good_indices:
            good_indices.append(v)
        if len(good_indices) >= 5:
            break

    # Plotting
    start = time.time()
    print("Plotting trajectories with violin plots...")
    plot_trajectories(
        preds=preds_arr,
        targets=targets_arr,
        draw_series_ids=bad_indices,
        horizon=horizon,
        output_dir=output_dir,
        suffix="bad",
    )

    plot_trajectories(
        preds=preds_arr,
        targets=targets_arr,
        draw_series_ids=good_indices,
        horizon=horizon,
        output_dir=output_dir,
        suffix="good",
    )
    end = time.time()
    print(f"Finished plotting trajectories in {end - start:.2f} sec.")

    # Plotting trajectories with separate histograms
    start = time.time()
    print("Plotting trajectories with individual histograms...")
    plot_trajectories_histograms(
        preds=preds_arr,
        targets=targets_arr,
        draw_series_ids=bad_indices[:1],
        horizon=horizon,
        var_level=alphas[-1],
        output_dir=output_dir,
    )
    end = time.time()
    print(f"Finished plotting histograms in {end - start:.2f} sec.")
