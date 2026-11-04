# Backward-compatible alias: this module now lives in core/postprocessHelpers.py.
# Lets older notebooks keep using "import TagmentationAnalysis.postprocessHelpers" or "import postprocessHelpers".
import sys
try:
    from .core import postprocessHelpers as _module
except ImportError:
    from core import postprocessHelpers as _module
sys.modules[__name__] = _module
