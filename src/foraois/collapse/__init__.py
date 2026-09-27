"""
foraois.collapse -- the collapse-barrier prescription delta_c(M, z), kept separate from the dark-matter model
(which acts on P(k) and sigma(M) inside ``CosmoData``) and from the first-crossing and tree algorithms.

The barrier is chosen by the config keys ``barrier`` (``fixed`` or ``linear``) and ``barrier_parameter``; see
``barrier.py`` for the definitions and for which algorithms support which barrier. ``delta_c(M, z, cosmo_data)`` is
the interface tree-building code depends on.
"""

from .barrier import barrier_is_flat, barrier_settings, delta_c, require_flat_barrier

__all__ = ["barrier_is_flat", "barrier_settings", "delta_c", "require_flat_barrier"]
