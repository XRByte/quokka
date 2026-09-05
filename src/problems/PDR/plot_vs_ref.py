#!/usr/bin/env python3
"""Overlay the live-radiation PDR profile against the PseudoPDR reference.

Reads tests/pdr_State/profile.txt (this run) and tests/pdr_State_chi1_n1000/profile.txt
(reference, chi=1 n=1000) and writes tests/pdr_State/pdr_vs_ref.png: temperature and the
H / H2 / C / C+ / CO abundances vs Av. Run from anywhere; paths resolve to the repo tests/.
"""

import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# tests/ dir (repo_root/tests), independent of cwd
TESTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tests"
)
INPUTS = os.path.join(TESTS, "..", "inputs", "PDR.toml")


def _toml_ninit(path: str) -> float:
    """Read pdr.ninit from the input TOML so the plot's density always matches the run."""
    import re

    in_pdr = False
    with open(path) as fh:
        for line in fh:
            s = line.split("#", 1)[0].strip()
            if s.startswith("[") and s.endswith("]"):
                in_pdr = s == "[pdr]"
            elif in_pdr:
                m = re.match(r"ninit\s*=\s*([0-9.eE+-]+)", s)
                if m:
                    return float(m.group(1))
    raise ValueError(f"pdr.ninit not found in {path}")


# Run density n_H (cm^-3): first CLI arg overrides, else read from the input TOML so the
# plot stays consistent with the run. It sets the simulation Av (=x*n_H/N_H_per_Av), the
# abundance normalization (n_i/n_H), and the matching reference pdr_State_chi<CHI>_n<NH>.
NH = float(sys.argv[1]) if len(sys.argv) > 1 else _toml_ninit(INPUTS)
# FUV field strength chi (Draine units); second CLI arg, default 1. Selects the matching
# reference directory pdr_State_chi<CHI>_n<NH>. The reference dirs are named with chi in
# {0.1, 1, 10} and n in {10, 100, 1000}: integers print without a decimal (1, 10), 0.1 as "0.1".
CHI = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0


def _chi_label(chi: float) -> str:
    """Format chi to match the reference dir naming: 1.0 -> '1', 10.0 -> '10', 0.1 -> '0.1'."""
    return str(int(chi)) if chi == int(chi) else ("%g" % chi)


CHI_LABEL = _chi_label(CHI)
NH_PER_AV = 1.87e21  # column N_H per unit Av (cm^-2)
# Av path-length factor. Keep at 1.0: the PseudoPDR reference already writes its Av as
# iso_factor*N_H/1.87e21 (the same doubled Av it burns with), so its iso_factor=2 self-
# cancels in the plotted comparison -- multiplying our Av here would double-count it.
ISO_FACTOR = 1.0

SIM = os.path.join(TESTS, "pdr_State", "profile.txt")
REF = os.path.join(TESTS, f"pdr_State_chi{CHI_LABEL}_n{int(NH)}", "profile.txt")
OUT = os.path.join(TESTS, "pdr_State", f"pdr_vs_ref_chi{CHI_LABEL}_n{int(NH)}.png")
if not os.path.exists(REF):
    raise SystemExit(
        f"reference not found: {REF}\n"
        f"  available: chi in {{0.1, 1, 10}}, n in {{10, 100, 1000}} "
        f"(dirs pdr_State_chi<chi>_n<n>/profile.txt under tests/)"
    )

# column indices (0-based) into each profile's data row
#   sim: x Tgas Erad0 H Hp e H2 H2p He Hep C Cp CO ...  (T=1, species start at 3)
#   ref: NH Av Tgas H Hp e H2 H2p He Hep C Cp CO ...    (T=2, species start at 3)
SIM_T, REF_T = 1, 2
SPECIES = {"H": 3, "H2": 6, "C": 10, "Cp": 11, "CO": 12}  # same offsets in both files
COLORS = {"H": "k", "H2": "g", "C": "orange", "Cp": "purple", "CO": "m"}


def main() -> int:
    a = np.loadtxt(SIM, skiprows=1)
    a = a[np.argsort(a[:, 0])]
    Av = ISO_FACTOR * a[:, 0] * NH / NH_PER_AV  # x -> N_H (=x*n_H) -> Av, x ISO_FACTOR
    r = np.loadtxt(REF, skiprows=1)
    Avr = r[:, 1]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(Av, a[:, SIM_T], "b-", label="sim")
    ax[0].plot(Avr, r[:, REF_T], "b--", label="ref")
    ax[0].set(
        xscale="log",
        xlabel="Av",
        ylabel="T [K]",
        ylim=(0, 80),
        title=f"Temperature  chi={CHI_LABEL} n={int(NH)}  (min Av={Av.min():.1e})",
    )
    ax[0].legend()

    for s, c in SPECIES.items():
        ax[1].plot(Av, a[:, c] / NH, "-", c=COLORS[s], label=s)
        ax[1].plot(Avr, r[:, c] / NH, "--", c=COLORS[s])
    ax[1].set(
        xscale="log",
        yscale="log",
        xlabel="Av",
        ylabel="x_i",
        ylim=(1e-8, 2),
        title=f"Abundances chi={CHI_LABEL} n={int(NH)} (solid=sim, dashed=ref)",
    )
    ax[1].legend(ncol=2)

    fig.tight_layout()
    fig.savefig(OUT, dpi=110)
    print(
        f"wrote {OUT}  (chi={CHI_LABEL}, n={int(NH)}, rows={len(a)}, min Av={Av.min():.2e}, max Av={Av.max():.1f})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
