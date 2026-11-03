# Backward-compatible alias: this module now lives in tagmentation/postprocessHelpers.py.
# Lets older notebooks keep using "import TagmentationAnalysis.postprocessHelpers" or "import postprocessHelpers".
import sys
try:
    from .tagmentation import postprocessHelpers as _module
except ImportError:
    from tagmentation import postprocessHelpers as _module
sys.modules[__name__] = _module
