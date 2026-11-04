# Backward-compatible alias: this module now lives in seqsuite/NGSanalysis_module.py.
# Lets older notebooks keep using "import TagmentationAnalysis.NGSanalysis_module" or "import NGSanalysis_module".
import sys
try:
    from .seqsuite import NGSanalysis_module as _module
except ImportError:
    from seqsuite import NGSanalysis_module as _module
sys.modules[__name__] = _module
