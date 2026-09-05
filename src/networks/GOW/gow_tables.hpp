#ifndef GOW_TABLES_HPP
#define GOW_TABLES_HPP

// Runtime-loaded interpolation tables for the GOW network (CO cooling + dust
// photo-electric psi_gd).  The tables are read from an HDF5 file at network-init
// time (see actual_network_data.cpp) into quokka::DataTable objects and exposed
// to GPU device code through a managed-memory registry, mirroring the tabulated
// EOS registry in src/cooling/EOSTabulatedRegistry.hpp.
//
// The interpolation itself is done by quokka::DataTable (src/util/DataTable.hpp):
// uniform grids with per-axis log spacing, per-output log/linear transform, and
// built-in analytic partial derivatives.  See gen_gow_tables.py for the table
// layout (groups "l0", "co", "dust").

#include "AMReX_Extension.H"
#include "AMReX_GpuQualifiers.H"

#include "util/DataTable.hpp"

namespace gow_tables
{

// Output indices into the 3-output CO DataTable (group "co").
constexpr int CO_LLTE_IDX = 0;	// CO LTE cooling coefficient
constexpr int CO_NHALF_IDX = 1; // CO n_1/2
constexpr int CO_ALPHA_IDX = 2; // CO alpha exponent

// Output indices into the 2-output C/CO self-shielding DataTables ("c_shield",
// "co_shield"): shielding of the species by H2, and its own self-shielding.
constexpr int SHIELD_H2_IDX = 0;   // shielding factor from H2
constexpr int SHIELD_SELF_IDX = 1; // self-shielding factor

// GPU-friendly bundle of the const table views for one execution context.
struct GowGpuConstTables {
	quokka::DataTableGpuConst<1, 1> l0;	  // L0(T)
	quokka::DataTableGpuConst<2, 3> co;	  // [LLTE, nhalf, alpha](T, Neff)
	quokka::DataTableGpuConst<2, 1> dust;	  // psi_gd(Tg, nH)
	quokka::DataTableGpuConst<1, 2> c_shield;  // [by-H2, self](N) for atomic C
	quokka::DataTableGpuConst<1, 2> co_shield; // [by-H2, self](N) for CO
};

struct GowTabulatedRegistry {
	GowGpuConstTables host;
	GowGpuConstTables device;
};

// Managed-memory registry pointer; set once by registerGowTables() at init.
extern AMREX_GPU_MANAGED GowTabulatedRegistry *g_gow_registry; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

[[nodiscard]] inline AMREX_GPU_HOST_DEVICE auto getGowRegistry() -> GowTabulatedRegistry * { return g_gow_registry; }

// Returns the device- or host-appropriate table bundle for the current context.
[[nodiscard]] AMREX_FORCE_INLINE AMREX_GPU_HOST_DEVICE auto getGowTables() -> GowGpuConstTables const &
{
	auto *reg = getGowRegistry();
	AMREX_ASSERT(reg != nullptr);
	AMREX_IF_ON_DEVICE((return reg->device;))
	AMREX_IF_ON_HOST((return reg->host;))
}

// Load the GOW cooling tables from HDF5 (host) and populate the managed registry.
// Idempotent: safe to call more than once (re-reads and overwrites).
void registerGowTables();

} // namespace gow_tables

#endif // GOW_TABLES_HPP
