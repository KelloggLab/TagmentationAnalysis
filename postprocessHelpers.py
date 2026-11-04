# Backward-compatible alias: this module now lives in seqsuite/postprocessHelpers.py.
# Lets older notebooks keep using "import TagmentationAnalysis.postprocessHelpers" or "import postprocessHelpers".
import sys
try:
    from .seqsuite import postprocessHelpers as _module
except ImportError:
    from seqsuite import postprocessHelpers as _module
sys.modules[__name__] = _module
