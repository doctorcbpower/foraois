"""
Characterisation of PCH08 main-progenitor histories against the Zhang & Hui generator at low M_res / high z.
Diagnostic only: it reports numbers and does not decide whether the behaviour is an implementation issue or a
regime limitation. See docs/PCH08_HIGH_Z_DIAGNOSTIC.md.

Uses only foraois (numba backend). Masses are in Msun/h because that is the generator's unit; only ratios are shown.
"""
import numpy as np
from pathlib import Path

from foraois import cosmo_utils, ZhangHuiMergerTree, PCHMergerTree
from foraois.utils import io

CONFIG = Path(__file__).resolve().parents[1] / "config" / "menon_power_2024.yml"
rp = io.get_params(str(CONFIG))
cd = cosmo_utils.CosmoData(rp, redshift=[0.0])
GENS = {"PCH08": PCHMergerTree(cd, rp), "ZH": ZhangHuiMergerTree(cd, rp, model="cdm")}
N = 60


def forest(name, M0, z0, z_target, M_res, dz=0.05):
    out = GENS[name].build_forest_numba(M0_array=np.full(N, M0), z0=z0, z_max=z_target + 1.0, M_res=M_res, dz=dz)
    mass, z_steps, smooth, merged = out[0], out[1], out[2], out[3]
    j = int(np.argmin(abs(z_steps[1:] - z_target)))          # mass_history[:, j] is the mass at z_steps[j+1]
    r = mass[:, j] / M0
    return r, float(np.mean(merged > 0)), float(merged.sum() / max(smooth.sum(), 1e-300))


def line(name, M0, z0, zt, M_res, dz=0.05):
    r, fm, ms = forest(name, M0, z0, zt, M_res, dz)
    return f"{name:5s} M(z={zt})/M0 median {np.median(r):.3f} [16,84 = {np.percentile(r,16):.3f}, {np.percentile(r,84):.3f}];  steps with a resolved merger {fm*100:5.1f}%;  merged/smooth mass {ms:.2g}"


print("A. standard anchor z0 = 0, M0 = 1e12, M_res = 1e4 (Msun/h)")
for zt in (1.0, 2.0):
    for g in GENS:
        print("  ", line(g, 1e12, 0.0, zt, 1e4))
print("\nB. anchor z0 = 5, M(z=10)/M0 for several halo masses, M_res = 1e4")
for M0 in (3e10, 3e11, 3e12, 3e13):
    for g in GENS:
        print(f"   M0={M0:.0e}  ", line(g, M0, 5.0, 10.0, 1e4))
print("\nC. dependence on M_res at z0 = 0, M0 = 1e12, z = 1")
for M_res in (1e4, 1e8, 1e10):
    for g in GENS:
        print(f"   M_res={M_res:.0e}  ", line(g, 1e12, 0.0, 1.0, M_res))
print("\nD. dependence on dz (PCH08, z0 = 5, M0 = 3e10, M_res = 1e4)")
for dz in (0.02, 0.05, 0.1, 0.2):
    print(f"   dz={dz}  ", line("PCH08", 3e10, 5.0, 10.0, 1e4, dz))
