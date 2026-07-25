"""Mathematical constants used throughout the package."""

import math
import numpy as np

# Euler's number (for base conversion and default base)
E = math.e

# Unit ball volumes for dimensions 1-3
# V_1 = 2.0 (line segment: [-1, 1])
# V_2 = pi (circle with radius 1)
# V_3 = 4*pi/3 (sphere with radius 1)
UNIT_BALL_VOLUMES = np.array([2.0, math.pi, 4.0 * math.pi / 3.0])

# Precomputed log of unit ball volumes for efficiency
LOG_UNIT_BALL_VOLUMES = np.log(UNIT_BALL_VOLUMES)
