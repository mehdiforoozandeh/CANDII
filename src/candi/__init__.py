"""CANDI — confidence-aware neural denoising imputer.

Only the training-path symbols are exported. A bare `import candi` must not pull in the eval
instrument or the prep pipeline, both of which are heavy and neither of which the model needs.
"""

__version__ = "0.1.0"

from candi.config import EncoderConfig
from candi.encoder import V2Encoder
from candi.model import CandiModel, build_model, forward_full, nb_nll

__all__ = [
    "__version__",
    "EncoderConfig",
    "V2Encoder",
    "CandiModel",
    "build_model",
    "forward_full",
    "nb_nll",
]
