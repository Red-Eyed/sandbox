from collections.abc import Callable

from torch import nn

# Runs calibration data through the prepared model once, populating its observers.
# Typed as nn.Module, not the concrete GraphModule prepare_pt2e() returns, since the
# callback only ever calls it and never touches GraphModule-specific methods.
CalibrationFn = Callable[[nn.Module], None]
