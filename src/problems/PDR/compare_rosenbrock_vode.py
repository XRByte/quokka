#!/usr/bin/env python3
"""Compare the qoukka-jaff Rosenbrock PDR profile against the source VODE profile.

Both files are pdr_State/profile.txt written by testPDR.cpp (composite AMR grid,
sorted by depth x). Columns:
  x Tgas Erad0 Erad1 Erad2 H Hp e H2 H2p He Hep C Cp CO HCOp O Si Sip CH OH H3p Op
(NOTE: the header labels 3 Erad columns but nGroups=1 -> only Erad0 is real; the
 remaining "Erad1 Erad2" labels are cosmetic. We key on x + the named species.)

Usage: compare_rosenbrock_vode.py [rosenbrock_profile] [vode_profile]
Defaults to the two worktree paths.
"""
import sys
import numpy as np

ROS = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/anish/External/programming/research/quokka/worktrees/qoukka-jaff/tests/pdr_State/profile.txt"
VODE = sys.argv[2] if len(sys.argv) > 2 else \
    "/home/anish/External/programming/research/quokka/worktrees/photoionization/tests/pdr_State/profile.txt"


def load(path):
    with open(path) as f:
        header = f.readline().split()
    data = np.loadtxt(path, skiprows=1)
    return header, data


hr, ros = load(ROS)
hv, vode = load(VODE)

# Align on x. The two runs use the same grid, but AMR cell counts can differ slightly;
# interpolate VODE onto the Rosenbrock x-grid for a pointwise comparison.
xr, xv = ros[:, 0], vode[:, 0]
print(f"Rosenbrock rows: {len(xr)}   VODE rows: {len(xv)}")
print(f"x range  ROS [{xr.min():.3e}, {xr.max():.3e}]   VODE [{xv.min():.3e}, {xv.max():.3e}]")

# columns to compare: Tgas (idx 1) + all species (idx 5..22). skip Erad label mess.
cols = {"Tgas": 1}
species = hr[5:5 + 18]
for i, name in enumerate(species):
    cols[name] = 5 + i


def relerr(a, b):
    denom = np.where(np.abs(b) > 0, np.abs(b), 1.0)
    return np.abs(a - b) / denom


print("\n%-8s  %10s  %10s  %10s" % ("field", "max_relerr", "med_relerr", "@x(maxerr)"))
print("-" * 46)
worst = []
for name, c in cols.items():
    vi = np.interp(xr, xv, vode[:, c])
    re = relerr(ros[:, c], vi)
    # ignore floor species (both ~1e-40) where relerr is noise
    mask = np.abs(vi) > 1e-30
    if mask.sum() == 0:
        continue
    rem = re[mask]
    imax = np.argmax(rem)
    xatmax = xr[mask][imax]
    print("%-8s  %10.3e  %10.3e  %10.3e" % (name, rem.max(), np.median(rem), xatmax))
    worst.append((rem.max(), name))

worst.sort(reverse=True)
print("\nlargest deviations (Rosenbrock vs VODE):")
for v, n in worst[:6]:
    print(f"  {n:8s} max_relerr = {v:.2%}")

# surface temperature (first non-floor cell) sanity
print(f"\nsurface Tgas:  Rosenbrock {ros[0,1]:.3f} K   VODE {np.interp(xr[0],xv,vode[:,1]):.3f} K")
print(f"deepest  Tgas: Rosenbrock {ros[-1,1]:.3f} K   VODE {np.interp(xr[-1],xv,vode[:,1]):.3f} K")
