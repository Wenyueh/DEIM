from ._embed import embed, encoders
from ._slice.abstract import Slicer
from ._slice.domino import DominoSlicer
from ._slice.seq_domino import SeqDominoSlicer
from ._slice.domino_noise_transition import DominoErrorSlicer
from ._slice.spotlight import SpotlightSlicer
from ._slice.barlow import BarlowSlicer
from ._slice.Gyhat_domino import DominoSlicerGaussianYhat
from ._slice.GG_domino import DominoSlicerAllGaussian
from ._slice.GG_domino_NT import DominoSlicerAllGaussianNT
from ._slice.distance_domino import DistanceDominoSlicer
from ._slice.Ghat_noise_transition_domino import DominoSlicerGaussianYhatNT
from ._slice.multiaccuracy import MultiaccuracySlicer
from ._slice.abstract import Slicer
from ._describe.generate import generate_candidate_descriptions
from ._describe import describe
from .gui import explore

__all__ = [
    "DominoSlicer",
    "DominoErrorSlicer",
    "DominoSlicerGaussianYhat",
    "DominoSlicerGaussianYhatNT",
    "DominoSlicerAllGaussian",
    "DominoSlicerAllGaussianNT",
    "SeqDominoSlicer",
    "DistanceDominoSlicer",
    "SpotlightSlicer",
    "BarlowSlicer",
    "MultiaccuracySlicer",
    "Slicer",
    "embed",
    "encoders",
    "explore",
    "describe",
    "generate_candidate_descriptions",
]
