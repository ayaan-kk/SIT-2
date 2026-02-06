"""IRBS (Interleaved Randomized Block Scheduling) module for SIT.

Provides:
- protocol: experiment execution with interleaved randomization
- estimators: treatment effect estimation and confidence intervals
- drift_bias_demo: demonstration of drift-induced bias and IRBS correction
"""

from . import protocol
from . import estimators
from . import drift_bias_demo
