"""
foraois.collapse -- collapse-barrier physics, isolated from the
excursion-set/tree-building machinery (pch_trees.py, first_crossing.py).

`delta_c(M, z, model, cosmo_data)` (barrier.py) is the only public
interface tree-building code should depend on.

Per-model modules:
    cdm.py   -- delta_c_cdm: constant delta_col(z), mass-independent.
    wdm.py   -- delta_c_wdm: same constant barrier as CDM (whether this is
                actually sufficient without a WDM-specific rate refit is
                an open question -- see ROADMAP.md).
    fdm.py   -- delta_c_fdm: NOT the real physics -- falls back to the
                constant CDM barrier as a known-inadequate placeholder,
                and warns every time, rather than silently returning a
                wrong answer as though it were correct. The real
                mass-dependent barrier is still open research (ROADMAP.md).
    sidm.py  -- delta_c_sidm: not implemented at all -- no barrier has
                even been scoped (ROADMAP.md).
"""

from .barrier import delta_c

__all__ = ["delta_c"]
