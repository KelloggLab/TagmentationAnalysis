# Backward-compatible alias: this package is now `seqsuite`.
# Lets older notebooks keep using "from tagmentation import postprocessHelpers".
import sys

import seqsuite

sys.modules[__name__] = seqsuite
