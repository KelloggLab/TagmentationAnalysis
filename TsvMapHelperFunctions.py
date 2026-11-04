# Backward-compatible alias: this module now lives in seqsuite/TsvMapHelperFunctions.py.
# Lets older notebooks keep using "import TagmentationAnalysis.TsvMapHelperFunctions" or "import TsvMapHelperFunctions".
import sys
try:
    from .seqsuite import TsvMapHelperFunctions as _module
except ImportError:
    from seqsuite import TsvMapHelperFunctions as _module
sys.modules[__name__] = _module
