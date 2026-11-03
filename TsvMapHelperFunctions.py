# Backward-compatible alias: this module now lives in tagmentation/TsvMapHelperFunctions.py.
# Lets older notebooks keep using "import TagmentationAnalysis.TsvMapHelperFunctions" or "import TsvMapHelperFunctions".
import sys
try:
    from .tagmentation import TsvMapHelperFunctions as _module
except ImportError:
    from tagmentation import TsvMapHelperFunctions as _module
sys.modules[__name__] = _module
