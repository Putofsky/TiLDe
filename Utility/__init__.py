"""Reusable scientific utilities for the TiLDe project.

Submodules are intentionally *not* imported eagerly.  This keeps lightweight
workflows (preprocessing, plots and time-delay estimation) usable on machines
where optional machine-learning dependencies such as PyTorch or ``sktime`` are
not installed.

Examples
--------
Import only the module required by the current workflow::

    from Utility import Pretraitement
    from Utility import time_delay_interpolation
"""

__all__ = [
    "IdSep",
    "Inference",
    "LensNN",
    "Plot",
    "Pretraitement",
    "RF",
    "time_delay_interpolation",
    "time_delay_pspline",
]
