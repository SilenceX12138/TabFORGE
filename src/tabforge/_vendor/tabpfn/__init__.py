from importlib.metadata import version

from tabforge._vendor.tabpfn.classifier import TabPFNClassifier
from tabforge._vendor.tabpfn.misc.debug_versions import display_debug_info
from tabforge._vendor.tabpfn.model_loading import (
    load_fitted_tabpfn_model,
    save_fitted_tabpfn_model,
)
from tabforge._vendor.tabpfn.regressor import TabPFNRegressor

try:
    __version__ = version(__name__)
except ImportError:
    __version__ = "unknown"

__all__ = [
    "TabPFNClassifier",
    "TabPFNRegressor",
    "__version__",
    "display_debug_info",
    "load_fitted_tabpfn_model",
    "save_fitted_tabpfn_model",
]
