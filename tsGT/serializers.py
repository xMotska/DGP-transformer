"""Serialization of the elements of Gym spaces into discrete sequences."""

import copy
from abc import ABC, abstractmethod

from absl import logging
import gin
import gymnasium as gym
import jax.numpy as jnp
import numpy as np
from sklearn import preprocessing as skp


class SpaceSerializer(ABC):
    """Abstract base class for space serializers.
    
    Replacement for trax.rl.space_serializer.SpaceSerializer.
    
    Attributes:
        space_type: The gym space type this serializer handles.
    """
    
    space_type = None
    
    def __init__(self, space: gym.Space, vocab_size: int):
        """Initialize the serializer.
        
        Args:
            space: The gym space to serialize.
            vocab_size: Number of distinct tokens in the vocabulary.
        """
        self._space = space
        self._vocab_size = vocab_size
    
    @abstractmethod
    def serialize(self, data: np.ndarray) -> np.ndarray:
        """Serialize data from the space into token sequences.
        
        Args:
            data: Array of values from the space.
        
        Returns:
            Array of token sequences.
        """
        pass
    
    @abstractmethod
    def deserialize(self, representation: np.ndarray) -> np.ndarray:
        """Deserialize token sequences back to space values.
        
        Args:
            representation: Array of token sequences.
        
        Returns:
            Array of values in the space.
        """
        pass
    
    @property
    @abstractmethod
    def representation_length(self) -> int:
        """Length of the token sequence for one element."""
        pass
    
    @property
    def vocab_size(self) -> int:
        """Size of the token vocabulary."""
        return self._vocab_size
    
    @property
    def space(self) -> gym.Space:
        """The gym space being serialized."""
        return self._space


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def inv_sigmoid(s: np.ndarray) -> np.ndarray:
    """Inverse sigmoid (logit) function."""
    # Clip to avoid numerical issues
    s = np.clip(s, 1e-7, 1 - 1e-7)
    return np.log(s / (1 - s))


@gin.configurable(denylist=["space", "vocab_size"], module="code.serializers")
class BoxSpaceSerializer(SpaceSerializer):
    """Serializer for gym.spaces.Box.

    Assumes that the space is bounded. Internally rescales it to the [0, 1]
    interval and uses a fixed-precision encoding.
    """

    space_type = gym.spaces.Box

    def __init__(
        self,
        space: gym.spaces.Box,
        vocab_size: int,
        precision: int = 2,
        max_range: tuple[float, float] = (-100.0, 100.0),
        first_digit_mode: str = "uniform",
        quantile_fit_n_points: int = 1024**2,
        clip_or_squash: str = "clip",
    ):
        """Initialize BoxSpaceSerializer.

        Args:
            space: Gym Box space.
            vocab_size: Number of distinct tokens.
            precision: Number of tokens to use to encode each number.
            max_range: Maximum representable range (min, max).
            first_digit_mode: How to encode the first digit. Available modes
                are 'uniform' and 'quantile'.
            quantile_fit_n_points: Number of datapoints to fit the histogram
                on when first_digit_mode == 'quantile'.
            clip_or_squash: Method for handling out-of-range values.
                'clip' clips to bounds, 'squash' uses sigmoid.
        """
        self._precision = precision
        self._first_digit_mode = first_digit_mode
        self._quantile_fit_n_points = quantile_fit_n_points

        assert space.shape == (), f"Expected scalar Box space, got shape {space.shape}"

        # Some gym envs have unreasonably high bounds for observations.
        # We clip so we can represent them.
        bounded_space = copy.copy(space)
        min_low, max_high = max_range
        bounded_space.low = np.maximum(space.low, min_low)
        bounded_space.high = np.minimum(space.high, max_high)
        
        if not np.allclose(bounded_space.low, space.low) or not np.allclose(
            bounded_space.high, space.high
        ):
            logging.warning(
                "Space limits %s, %s out of bounds %s. Clipping to %s, %s.",
                str(space.low),
                str(space.high),
                str(max_range),
                str(bounded_space.low),
                str(bounded_space.high),
            )

        self._discretizer = None

        super().__init__(bounded_space, vocab_size)
        
        assert clip_or_squash in ("clip", "squash"), (
            f"clip_or_squash must be 'clip' or 'squash', got {clip_or_squash}"
        )
        self.clip_or_squash = clip_or_squash

    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        """Preprocess data to [0, 1] range.
        
        Args:
            data: Raw data from the space.
        
        Returns:
            Normalized data in [0, 1].
        """
        array = np.asarray(data)
        # Normalize to [0, 1]
        array = (array - self._space.low) / (self._space.high - self._space.low)
        
        if self.clip_or_squash == "clip":
            array = np.clip(array, 0, 1)
        elif self.clip_or_squash == "squash":
            # Apply sigmoid-based squashing
            scale = 5 + 5 * (self._space.low < 0)
            offset = 5 * (self._space.low < 0)
            array = array * scale - offset
            array = sigmoid(array)
            # Adjust based on space bounds
            array = np.where(
                self._space.low < 0,
                array,
                (array - 0.5) * 2
            )
        
        return array

    def _postprocess(self, data: np.ndarray) -> np.ndarray:
        """Postprocess data from [0, 1] back to original space.
        
        Args:
            data: Normalized data in [0, 1].
        
        Returns:
            Data in original space.
        """
        array = np.asarray(data)
        
        if self.clip_or_squash == "squash":
            # Reverse the sigmoid squashing
            array = np.where(
                self._space.low < 0,
                array,
                array / 2 + 0.5
            )
            array = inv_sigmoid(array)
            scale = 5 + 5 * (self._space.low < 0)
            offset = 0.5 * (self._space.low < 0)
            array = array / scale + offset

        # Denormalize from [0, 1] to original space
        array = array * (self._space.high - self._space.low) + self._space.low
        return array

    def fit(self, input_stream):
        """Fit the serializer on data (for quantile mode).

        Args:
            input_stream: Generator of arrays of shape (batch_size, length).
        """
        if self._first_digit_mode != "quantile":
            return

        n_points_left = self._quantile_fit_n_points
        point_batches = []
        
        while n_points_left > 0:
            series = next(input_stream)
            series = self._preprocess(series)
            point_batches.append(np.reshape(series, -1))
            n_points_left -= series.size
        
        points = np.concatenate(point_batches)
        points = np.reshape(points, (-1, 1))

        self._discretizer = skp.KBinsDiscretizer(
            strategy="quantile", encode="ordinal", n_bins=self._vocab_size
        )
        self._discretizer.fit(points)
        # Ensure edge bins cover full [0, 1] range
        self._discretizer.bin_edges_[0][0] = 0.0
        self._discretizer.bin_edges_[0][-1] = 1.0

    def serialize(self, data: np.ndarray) -> np.ndarray:
        """Serialize data to token sequences.
        
        Args:
            data: Array of values from the space.
        
        Returns:
            Array of token sequences with shape (batch_size, precision).
        """
        array = self._preprocess(data)
        batch_size = array.shape[0]
        digits = []
        current_range = 1.0
        
        for digit_index in range(-1, -self._precision - 1, -1):
            if digit_index == -1 and self._first_digit_mode == "quantile":
                edges = np.array(self._discretizer.bin_edges_[0])
                # Manual searchsorted for JAX compatibility
                digit = np.searchsorted(edges[1:-1], array, side="right")
                lower_bound = edges[digit]
                current_range = edges[digit + 1] - edges[digit]
            else:
                threshold = current_range / self._vocab_size
                digit = np.array(array / threshold).astype(np.int32)
                # Handle corner case of x ~= high
                digit = np.where(digit == self._vocab_size, digit - 1, digit)
                lower_bound = digit * threshold
                current_range = current_range / self._vocab_size

            digits.append(digit)
            array = array - lower_bound

        digits = np.stack(digits, axis=-1)
        return np.reshape(digits, (batch_size, -1))

    def deserialize(self, representation: np.ndarray) -> np.ndarray:
        """Deserialize token sequences back to space values.
        
        Args:
            representation: Array of token sequences.
        
        Returns:
            Array of values in the space.
        """
        digits = np.asarray(representation)
        batch_size = digits.shape[0]
        digits = np.reshape(digits, (batch_size, self._precision))
        array = np.zeros(digits.shape[:-1])
        current_range = 1.0
        
        for digit_index_in_seq in range(self._precision):
            if digit_index_in_seq == 0 and self._first_digit_mode == "quantile":
                digit = digits[..., 0]
                edges = np.array(self._discretizer.bin_edges_[0])
                array = edges[digit]
                current_range = edges[digit + 1] - edges[digit]
            else:
                array = array + (
                    current_range / self._vocab_size * digits[..., digit_index_in_seq]
                )
                current_range = current_range / self._vocab_size
        
        array = np.reshape(array, (batch_size,) + self._space.shape)
        return self._postprocess(array)

    @property
    def representation_length(self) -> int:
        """Length of token sequence for one element."""
        return self._precision * int(np.prod(self._space.low.shape))

    @property
    def significance_map(self) -> np.ndarray:
        """Map showing significance level of each position.
        
        Returns:
            Array where each element indicates the digit index (0 = most significant).
        """
        return np.reshape(
            np.broadcast_to(
                np.arange(self._precision),
                self._space.shape + (self._precision,),
            ),
            -1,
        )
