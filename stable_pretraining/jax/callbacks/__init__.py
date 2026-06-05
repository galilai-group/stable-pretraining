"""Evaluation/training callbacks for the JAX backend.

All callbacks subclass :class:`stable_pretraining.jax.trainer.Callback`, which
exposes the same hook names as Lightning's ``pl.Callback`` — so the ordering
rules documented in ``AGENTS.md`` carry over.
"""

from ..trainer import Callback
from .earlystop import EarlyStopping
from .knn import OnlineKNN, knn_predict
from .lidar import LiDAR, lidar
from .probe import OnlineProbe
from .queue import OnlineQueue
from .rankme import RankMe, rankme
from .teacher_student import TeacherStudentCallback
from .writer import OnlineWriter

__all__ = [
    "Callback",
    "OnlineProbe",
    "OnlineKNN",
    "knn_predict",
    "RankMe",
    "rankme",
    "LiDAR",
    "lidar",
    "OnlineQueue",
    "OnlineWriter",
    "EarlyStopping",
    "TeacherStudentCallback",
]
