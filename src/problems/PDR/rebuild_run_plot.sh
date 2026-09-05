#!/usr/bin/env bash
# Rebuild the PDR target, run it to equilibrium at a chosen FUV field strength chi and
# density n_H, then overlay the result on the matching PseudoPDR reference.
#
# Usage: rebuild_run_plot.sh [chi] [n_H] [stop_time]
#   chi         FUV field strength in Draine units. Default 1. The incident band-0 energy
#               density is set to Erad_inc_0 = 8.94e-14 * chi (8.94e-14 = Draine chi=1).
#               Reference data exists for chi in {0.1, 1, 10}.
#   n_H         gas number density (cm^-3). Default: pdr.ninit from inputs/PDR.toml.
#               Reference data exists for n_H in {10, 100, 1000}. Lx auto-derives from n_H.
#   stop_time   sim end time (s). Omit to use inputs/PDR.toml's [pdr] stop_time.
# Env:
#   OMP_NUM_THREADS   burn threads (default 16). MPI is NOT used: the shielding-column
#                     sweep in computeAfterTimestep accumulates across the composite grid
#                     on a single rank, so scale with OpenMP only.
#   SKIP_BUILD=1      skip the build step (just run + plot)
#
# Writes tests/pdr_State/profile.txt (this run) and
# tests/pdr_State/pdr_vs_ref_chi<chi>_n<n>.png (overlay vs reference).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"

CHI="${1:-1}"
NH="${2:-}"          # empty -> read pdr.ninit from the toml (below)
STOP_TIME="${3:-}"   # empty -> use toml's [pdr] stop_time

cd "$ROOT"

# Incident band-0 energy density for this chi (8.94e-14 erg cm^-3 = Draine chi=1).
ERAD0="$(awk -v c="$CHI" 'BEGIN{printf "%.6e", 8.94e-14 * c}')"

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
	echo ">>> build PDR"
	quokka build -d 1d PDR -j 8 --root "$ROOT"
fi

BIN="$(find "$ROOT/build/1d" -name PDR -type f -executable | head -1)"
if [[ -z "$BIN" ]]; then
	echo "ERROR: PDR binary not found under build/1d (configure first: quokka config -d 1d)" >&2
	exit 1
fi

# Build the runtime override list: chi (via Erad_inc_0), optional n_H, optional stop_time.
ARGS=(pdr.Erad_inc_0="$ERAD0")
[[ -n "$NH" ]] && ARGS+=(pdr.ninit="$NH")
[[ -n "$STOP_TIME" ]] && ARGS+=(pdr.stop_time="$STOP_TIME")

echo ">>> run PDR  chi=$CHI (Erad_inc_0=$ERAD0)  n_H=${NH:-<toml>}  stop_time=${STOP_TIME:-<toml>}"
echo "    (log: tests/pdr_eq.log)"
cd "$ROOT/tests"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}" "$BIN" ../inputs/PDR.toml \
	"${ARGS[@]}" amrex.abort_on_unused_inputs=0 2>&1 |
	grep -aE "Coarse STEP|Finished|Abort|NAN|Lx =" >pdr_eq.log || true
tail -1 pdr_eq.log

# Resolve n_H actually used for the plot (arg wins, else pdr.ninit from the toml).
NH_PLOT="$NH"
if [[ -z "$NH_PLOT" ]]; then
	NH_PLOT="$(awk -F= '/^\s*ninit\s*=/{gsub(/[^0-9.eE+-]/,"",$2); print $2; exit}' "$ROOT/inputs/PDR.toml")"
fi

echo ">>> plot vs reference (chi=$CHI, n_H=$NH_PLOT)"
python3 "$HERE/plot_vs_ref.py" "$NH_PLOT" "$CHI"
