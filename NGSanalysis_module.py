# Backward-compatible alias: this module now lives in core/NGSanalysis_module.py.
# Lets older notebooks keep using "import TagmentationAnalysis.NGSanalysis_module" or "import NGSanalysis_module".
import sys
try:
    from .core import NGSanalysis_module as _module
except ImportError:
    from core import NGSanalysis_module as _module
sys.modules[__name__] = _module
