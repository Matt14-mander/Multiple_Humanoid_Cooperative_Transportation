"""Backend-neutral PAINT-style intent estimation contracts.

PyTorch-dependent components live in ``network``, ``losses`` and ``trainer``
so importing this package remains possible in lightweight deployment tools.
"""

from .dataset import IntentDataset, split_by_episode
from .history import IntentHistoryBuffer
from .normalization import NormalizationStats
from .runtime import CallableIntentBackend, IntentEstimatorRuntime
from .spec import IntentEstimatorSpec

__all__ = [
    "CallableIntentBackend",
    "IntentDataset",
    "IntentEstimatorRuntime",
    "IntentEstimatorSpec",
    "IntentHistoryBuffer",
    "NormalizationStats",
    "split_by_episode",
]

