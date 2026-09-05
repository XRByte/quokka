// Runtime loader + managed registry for the GOW network interpolation tables
// (CO cooling + dust photo-electric psi_gd).  Kept out of the JAFF-generated
// actual_network_data.cpp so it survives network regeneration.
//
// The tables are read from gow_cooling_tables.hdf5 (see gen_gow_tables.py) into
// quokka::DataTable objects and exposed to GPU device code through a managed
// registry, mirroring src/cooling/EOSTabulatedRegistry.hpp.  Registration is
// scheduled via amrex::ExecOnInitialize so it runs during amrex::Initialize(),
// with no dependency on the generated network init path.

#include <string>

#include "AMReX.H"
#include "AMReX_Arena.H"
#include "AMReX_FileSystem.H"
#include "AMReX_Print.H"

#include "util/DataTable.hpp"

#include <gow_tables.hpp>

namespace gow_tables
{

// Managed-memory registry pointer (declared extern in gow_tables.hpp).
AMREX_GPU_MANAGED GowTabulatedRegistry *g_gow_registry = nullptr; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

namespace
{
// Persistent owners of the device/host table memory (must outlive all kernels).
quokka::DataTable<1, 1> s_l0_table;	    // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
quokka::DataTable<2, 3> s_co_table;	    // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
quokka::DataTable<2, 1> s_dust_table;	    // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
quokka::DataTable<1, 2> s_c_shield_table;   // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
quokka::DataTable<1, 2> s_co_shield_table;  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

// Assemble a GowGpuConstTables bundle from the DataTables.
auto makeConstTables(bool device) -> GowGpuConstTables
{
	if (device) {
		return GowGpuConstTables{.l0 = s_l0_table.const_tables(),
					 .co = s_co_table.const_tables(),
					 .dust = s_dust_table.const_tables(),
					 .c_shield = s_c_shield_table.const_tables(),
					 .co_shield = s_co_shield_table.const_tables()};
	}
	return GowGpuConstTables{.l0 = s_l0_table.const_tables_host(),
				 .co = s_co_table.const_tables_host(),
				 .dust = s_dust_table.const_tables_host(),
				 .c_shield = s_c_shield_table.const_tables_host(),
				 .co_shield = s_co_shield_table.const_tables_host()};
}
} // namespace

void registerGowTables()
{
	const std::string filename = "gow_cooling_tables.hdf5";
	amrex::Print() << "Initializing GOW cooling tables from " << filename << ".\n";
	if (!amrex::FileSystem::Exists(filename)) {
		amrex::Abort("GOW cooling table file '" + filename + "' does not exist (expected in the run directory)!");
	}

	using quokka::TransformType;
	// Physical values stored; interpolate in log space, apply per-output transform on return.
	s_l0_table = quokka::DataTable<1, 1>::H5Reader(filename, "l0", {TransformType::log});
	s_co_table = quokka::DataTable<2, 3>::H5Reader(filename, "co", {TransformType::log, TransformType::log, TransformType::linear});
	s_dust_table = quokka::DataTable<2, 1>::H5Reader(filename, "dust", {TransformType::log});
	// C/CO self-shielding factors span many decades (1 -> ~1e-200); interpolate in log space.
	s_c_shield_table = quokka::DataTable<1, 2>::H5Reader(filename, "c_shield", {TransformType::log, TransformType::log});
	s_co_shield_table = quokka::DataTable<1, 2>::H5Reader(filename, "co_shield", {TransformType::log, TransformType::log});

	if (g_gow_registry == nullptr) {
		auto *mem = amrex::The_Managed_Arena()->alloc(sizeof(GowTabulatedRegistry));
		g_gow_registry = new (mem) GowTabulatedRegistry{}; // NOLINT(cppcoreguidelines-owning-memory)
	}
	g_gow_registry->host = makeConstTables(/*device=*/false);
	g_gow_registry->device = makeConstTables(/*device=*/true);
}

// NOTE: registerGowTables() must be called explicitly from problem_main() after
// amrex::Initialize() (so MPI is up for H5Reader's IO-processor read + broadcast).
// It is deliberately NOT scheduled via a static initializer + ExecOnInitialize:
// pushing onto amrex's namespace-scope The_Initialize_Function_Stack during static
// init hits the static-initialization-order fiasco (the stack may be constructed
// after this TU), which segfaults before main(). Mirror EOSTabulatedRegistry, which
// is likewise populated by an explicit runtime call, not at static-init time.

} // namespace gow_tables
