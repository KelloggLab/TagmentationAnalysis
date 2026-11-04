# Backward-compatible alias: this package is now `core`.
# Lets older notebooks keep using "from tagmentation import postprocessHelpers".
import sys

import core

sys.modules[__name__] = core
