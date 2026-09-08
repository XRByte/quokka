//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testPDR.cpp
/// \brief 1-D photodissociation region (PDR) with LIVE radiation transport and
///        the Gong, Ostriker & Wolfire (2017) chemistry + thermal network
///        (gow17, 18 species).
///
///        Unlike PseudoPDR (an Av-sweep of independent one-zone cells), this is a
///        real spatial slab: gas at fixed density n_H fills the box, an external
///        FUV field is injected at the illuminated (lo-x) face, and the three
///        chemistry-active radiation bands ([0, 11.2, 13.6, inf] eV) are evolved
///        by the M1 solver. Continuum (dust) + line absorption are baked into the
///        network radiation terms (there is no Av/chi global and no transport
///        opacity: ComputePlanck/FluxMeanOpacity == 0). Photons are consumed
///        locally by the chemistry burn, which reads the per-cell photon field
///        (Erad -> n_gamma -> burn_t::rn) and writes the attenuated field back.
///
///        Self-shielding: the network's H2/CO/C self-shielding factors read the
///        per-cell shielding columns network_rp::ncol_{H2,CO,C,H}. These are the
///        REAL cumulative columns Sum(n_species * dx) integrated from the
///        illuminated face to each cell (no isotropic path-length factor -- this
///        is one-sided normal incidence). Because the columns require a
///        face->interior ordered sweep and the network reads them from globals,
///        the burn is done serially per cell in computeAfterTimestep (the
///        built-in parallel photochem burner is left disabled): each cell's
///        columns are assigned to network_rp::ncol_* immediately before its burn.
///
///        Gas dynamics are frozen (only chemistry + temperature evolve): hydro is
///        enabled so the burner can read/write the hydro state, but momenta are
///        zeroed and the density is pinned after every step.
///
#include "util/BC.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "AMReX.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"
#include "AMReX_Scan.H"

#include "QuokkaSimulation.hpp"
#include "SimulationData.hpp"
#include "hydro/hydro_system.hpp"
#include "physics_info.hpp"
#include "radiation/photochemistry.hpp"
#include "radiation/radiation_system.hpp"
#include "util/fextract.hpp"

#include "actual_eos_data.H"
#include "burn_type.H"
#include "eos.H"
#include "extern_parameters.H"
#include "network.H"

#include <gow_tables.hpp>

using amrex::Real;
struct PDRTest {
}; // dummy type to allow compile-time polymorphism via template specialization

template <> struct quokka::EOS_Traits<PDRTest> {
	static constexpr double mean_molecular_weight = 1.0;
	static constexpr double gamma = 5. / 3.;
};

template <> struct Physics_Traits<PDRTest> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;		     // gas state accessed by the burner (kept frozen at runtime)
	static constexpr bool is_radiation_enabled = true;	     // evolve the FUV photon field
	static constexpr int numMassScalars = NumSpec;		     // 18 chemical species
	static constexpr int numPassiveScalars = numMassScalars + 0; // only mass scalars
	static constexpr int nGroups = 1;			     // single chemistry-active FUV band (must equal NumChemBands)
};

template <> struct RadSystem_Traits<PDRTest> {
	// Reduced speed of light (RSLA). The GOW network is generated with rsl = "c_hat"
	// (jaffgen.toml), i.e. it expects a reduced c: the steady-state photon number
	// density scales ~1/c_hat, keeping the photoreaction rate (~c_hat * n_gamma)
	// approximately invariant while easing the stiff in-burn photon-absorption term.
	static constexpr double c_hat_over_c = 1.0e-5;
	// Small positive floor: the M1 ConservedToPrimitive step requires E_r > 0 in every
	// (incl. dark/ghost) cell, so the field is never initialized or injected at exactly 0.
	static constexpr double Erad_floor = C::a_rad * 1.0e-8;
	static constexpr int beta_order = 0; // frozen gas: no O(v/c) radiation-pressure work term
	// Band boundaries are in eV (jaff's native unit for radiation band edges; see the GOW
	// registry entry CHEM_BANDS), so a boundary times energy_unit is the photon energy in erg:
	// energy_unit = ev2erg.
	static constexpr double energy_unit = C::ev2erg;
	// Single chemistry band (nGroups == 1): use the single-group opacity path. Transport
	// opacity is zero -- continuum + line absorption are baked into the network radiation
	// terms and applied in the chemistry burn (ComputePlanckOpacity returns g_kappa_dust = 0).
	static constexpr OpacityModel opacity_model = OpacityModel::single_group;
	static constexpr auto ChemBands() { return ChemBandsHeader_; }
	// photon-number spectrum index within the band (jaff network.radiation.power_law_index = 0)
	static constexpr auto ChemBandsPowerLawIndex() { return ChemBandsPowerLawIndex_; }
};

template <> struct SimulationData<PDRTest> {
	amrex::Real small_temp{};
	amrex::Real small_dens{};
	amrex::Real temperature{}; // initial gas temperature (K)
	amrex::Real ninit{};	   // fixed gas number density n_H (cm^-3)
	amrex::Real metallicity{}; // scales metal initial abundances

	std::array<amrex::Real, NumSpec> rel_abundance{}; // initial abundance relative to n_init
	amrex::Real t_prev_ = 0.0;			  // previous sim time (for burn dt)

	// Static face-anchored refinement: level `lev` tags cells with x < refine_frac[lev]*Lx
	// (fractions telescope: refine_frac[0] > refine_frac[1] > ...), so the finest level sits
	// at the illuminated face and the mesh coarsens outward. Consumed by refineGrid and by
	// the AMR-aware shielding-column sweep in computeAfterTimestep.
	// One fraction per refine level (index = level being refined). Default geometric
	// telescoping (ratio ~0.464) reaches Av ~ 1e-3 at the deepest boundary; override any
	// entry via refine_frac_<l> in the input. Sized for up to 16 levels.
	std::array<amrex::Real, 16> refine_frac{{0.1, 0.0464, 0.0215, 0.01, 4.64e-3, 2.15e-3, 1.0e-3, 4.64e-4, 2.15e-4, 1.0e-4, 4.64e-5, 2.15e-5,
						 1.0e-5, 4.64e-6, 2.15e-6, 1.0e-6}};
};

// Incident radiation energy density per band (erg cm^-3) injected at the lo-x face,
// and the pinned uniform gas state used to freeze the slab. AMREX_GPU_MANAGED so the
// device-side custom BC can read them; set in preCalculateInitialConditions.
namespace
{
AMREX_GPU_MANAGED amrex::Real Erad_floor_val = C::a_rad * 1.0e-8; // NOLINT
AMREX_GPU_MANAGED amrex::Real g_Erad_inc[3] = {0.0, 0.0, 0.0}; // NOLINT
AMREX_GPU_MANAGED amrex::Real g_rho0 = 0.0;		       // NOLINT
AMREX_GPU_MANAGED amrex::Real g_Egas0 = 0.0;		       // NOLINT
AMREX_GPU_MANAGED amrex::Real g_xn0[NumSpec] = {};	       // NOLINT (surface number densities)
AMREX_GPU_MANAGED amrex::Real g_kappa_dust = 0.0;	       // NOLINT (transport dust opacity; couples chem band to gas energy -> keep 0)
} // namespace

// Reconstruct the gas temperature from the hydro state (rho, internal energy,
// species number densities) by inverting the EOS.
AMREX_INLINE auto compute_Tgas(Real rho, Real Eint, const std::array<Real, NumSpec> &numdens) -> Real
{
	burn_t state;
	for (int n = 0; n < NumSpec; ++n) {
		state.xn[n] = numdens[n];
	}
	state.rho = rho;
	state.e = Eint / rho;
	eos(eos_input_re, state);
	return state.T;
}

template <> void QuokkaSimulation<PDRTest>::preCalculateInitialConditions()
{
	// initialize microphysics routines
	init_extern_parameters();

	amrex::ParmParse const pp("pdr");
	userData_.small_temp = 1e1;
	pp.query("small_temp", userData_.small_temp);
	userData_.small_dens = 1e-60;
	pp.query("small_dens", userData_.small_dens);

	userData_.temperature = 1.0e2; // K, initial (T solved self-consistently)
	pp.query("temperature", userData_.temperature);

	userData_.ninit = 1000.0; // fixed n_H
	pp.query("ninit", userData_.ninit);

	// face-anchored refinement fractions (telescoping); one per refine level
	for (int l = 0; l < static_cast<int>(userData_.refine_frac.size()); ++l) {
		pp.query(("refine_frac_" + std::to_string(l)).c_str(), userData_.refine_frac[l]);
	}

	userData_.metallicity = network_rp::metallicity;

	// Initial abundances relative to n_init (Gong+17 single-zone ICs, identical to
	// the GOW one-zone/PseudoPDR tests), in xn[] index order 0=H,1=H+,2=e-,3=H2,
	// 4=H2+,5=He,6=He+,7=C,8=C+,9=CO,10=HCO+,11=O,12=Si,13=Si+,14=CH,15=OH,16=H3+,17=O+.
	userData_.rel_abundance = {
	    1.0e0,	  // H
	    1.0e-4,	  // H+
	    2.60283e-4,	  // e-
	    1.0e-10,	  // H2
	    1.0e-40,	  // H2+
	    1.0e-1,	  // He
	    1.450654e-08, // He+
	    1.0e-40,	  // C
	    1.6e-4,	  // C+
	    1.0e-7,	  // CO
	    1.0e-40,	  // HCO+
	    3.2e-4,	  // O
	    1.7e-6,	  // Si
	    1.0e-40,	  // Si+
	    1.0e-40,	  // CH
	    1.0e-40,	  // OH
	    2.681411e-07, // H3+
	    1.0e-40,	  // O+
	};

	// Incident FUV field, per band (erg cm^-3). Single chemistry band (nGroups == 1) is
	// the 6-13.6 eV FUV band: put the Draine (chi=1) energy density in band 0. Overridable
	// from the input (Erad_inc_0) -- the incident-spectrum calibration knob.
	const amrex::Real u_FUV_draine = 8.94e-14; // Draine (1978) 6-13.6 eV energy density, chi=1
	g_Erad_inc[0] = u_FUV_draine;
	g_Erad_inc[1] = 0.0;
	g_Erad_inc[2] = 0.0;
	pp.query("Erad_inc_0", g_Erad_inc[0]);
	pp.query("Erad_inc_1", g_Erad_inc[1]);
	pp.query("Erad_inc_2", g_Erad_inc[2]);

	// RSLA normalization: with a reduced c_hat the steady-state photon number density
	// scales ~1/c_hat, so the photoreaction rate (~c_hat*n_gamma) stays physical only if
	// the injected energy density is boosted by c/c_hat. Apply that boost to every band.
	const amrex::Real rsl_boost = 1.0 / RadSystem_Traits<PDRTest>::c_hat_over_c;
	for (double &inc : g_Erad_inc) {
		inc *= rsl_boost;
	}

	// Surface (uniform) gas state, pinned every step to keep the slab frozen.
	burn_t state;
	const Real met = userData_.metallicity;
	Real rhotot = 0.0_rt;
	for (int n = 0; n < NumSpec; ++n) {
		const bool is_metal = (n == 8) || (n == 11) || (n == 12);
		const Real nd = userData_.rel_abundance[n] * userData_.ninit * (is_metal ? met : 1.0);
		g_xn0[n] = nd;
		state.xn[n] = nd;
		rhotot += nd * spmasses[n];
	}
	state.T = userData_.temperature;
	state.rho = rhotot;
	eos(eos_input_rt, state);
	g_rho0 = rhotot;
	g_Egas0 = state.e * rhotot;

	eos_init(userData_.small_temp, userData_.small_dens);
	network_init();
}

// No transport opacity: continuum + line absorption are baked into the network
// radiation terms and applied in the chemistry burn, so the single-group transport
// opacity is g_kappa_dust (= 0). Flux- and energy-mean opacities default to the Planck value.
template <> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<PDRTest>::ComputePlanckOpacity(const double /*rho*/, const double /*Tgas*/) -> Real
{
	return g_kappa_dust;
}

template <> AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto RadSystem<PDRTest>::ComputeFluxMeanOpacity(const double rho, const double Tgas) -> Real
{
	return ComputePlanckOpacity(rho, Tgas);
}

// One-sided illumination: a fixed incident radiation field enters at the lo-x face
// (ext_dir), streams in freely (F = c*E, forward-peaked), and leaves at hi-x
// (foextrap). The ghost gas state is the pinned uniform slab state.
template <>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
AMRSimulation<PDRTest>::setCustomBoundaryConditions(const amrex::IntVect &iv, amrex::Array4<amrex::Real> const &consVar, int /*dcomp*/, int /*numcomp*/,
						    amrex::GeometryData const & /*geom*/, const amrex::Real /*time*/, const amrex::BCRec *bcr, int /*bcomp*/,
						    int /*orig_comp*/)
{
	if (bcr->lo(0) != quokka::BCType::ext_dir) {
		return;
	}
	auto const [i, j, k] = iv.dim3();
	if (i >= 0) {
		return; // only the lo-x ghost cells are custom
	}

	constexpr int nRadVars = Physics_NumVars::numRadVarsPerGroup;
	for (int g = 0; g < Physics_Traits<PDRTest>::nGroups; ++g) {
		const amrex::Real E_inc = amrex::max(g_Erad_inc[g], Erad_floor_val); // strictly positive
		consVar(i, j, k, RadSystem<PDRTest>::radEnergy_index + nRadVars * g) = E_inc;
		// forward free-streaming inflow for illuminated bands; dark bands (at the floor) carry no flux
		consVar(i, j, k, RadSystem<PDRTest>::x1RadFlux_index + nRadVars * g) = C::c_light * g_Erad_inc[g];
		consVar(i, j, k, RadSystem<PDRTest>::x2RadFlux_index + nRadVars * g) = 0.0;
		consVar(i, j, k, RadSystem<PDRTest>::x3RadFlux_index + nRadVars * g) = 0.0;
	}

	// pinned uniform gas state in the ghost cells
	consVar(i, j, k, RadSystem<PDRTest>::gasEnergy_index) = g_Egas0;
	consVar(i, j, k, RadSystem<PDRTest>::gasDensity_index) = g_rho0;
	consVar(i, j, k, RadSystem<PDRTest>::gasInternalEnergy_index) = g_Egas0;
	consVar(i, j, k, RadSystem<PDRTest>::x1GasMomentum_index) = 0.0;
	consVar(i, j, k, RadSystem<PDRTest>::x2GasMomentum_index) = 0.0;
	consVar(i, j, k, RadSystem<PDRTest>::x3GasMomentum_index) = 0.0;
	for (int nn = 0; nn < NumSpec; ++nn) {
		consVar(i, j, k, HydroSystem<PDRTest>::scalar0_index + nn) = g_xn0[nn] * spmasses[nn];
	}
}

template <> void QuokkaSimulation<PDRTest>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	const Real rho0 = g_rho0;
	const Real Egas0 = g_Egas0;
	std::array<Real, NumSpec> numdens{};
	for (int n = 0; n < NumSpec; ++n) {
		numdens[n] = g_xn0[n];
	}

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		// gas: uniform slab, at rest
		state_cc(i, j, k, HydroSystem<PDRTest>::energy_index) = Egas0;
		state_cc(i, j, k, HydroSystem<PDRTest>::internalEnergy_index) = Egas0;
		state_cc(i, j, k, HydroSystem<PDRTest>::density_index) = rho0;
		state_cc(i, j, k, HydroSystem<PDRTest>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<PDRTest>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<PDRTest>::x3Momentum_index) = 0.0;
		for (int nn = 0; nn < NumSpec; ++nn) {
			state_cc(i, j, k, HydroSystem<PDRTest>::scalar0_index + nn) = numdens[nn] * spmasses[nn];
		}
		// radiation: start at the floor (strictly positive); the field builds up from the boundary inflow
		for (int g = 0; g < Physics_Traits<PDRTest>::nGroups; ++g) {
			state_cc(i, j, k, RadSystem<PDRTest>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g) = Erad_floor_val;
			state_cc(i, j, k, RadSystem<PDRTest>::x1RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0;
			state_cc(i, j, k, RadSystem<PDRTest>::x2RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0;
			state_cc(i, j, k, RadSystem<PDRTest>::x3RadFlux_index + Physics_NumVars::numRadVarsPerGroup * g) = 0.0;
		}
	});
}

// Static face-anchored refinement: tag cells on level `lev` whose center lies within
// refine_frac[lev] of the domain length from the illuminated (lo-x) face. The fractions
// telescope, so the mesh is finest at the face and coarsens outward.
template <> void QuokkaSimulation<PDRTest>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	const Real xlo = geom[lev].ProbLoArray()[0];
	const Real Lx = geom[0].ProbHiArray()[0] - geom[0].ProbLoArray()[0];
	const Real dx = geom[lev].CellSizeArray()[0];
	const Real x_refine = xlo + userData_.refine_frac[lev] * Lx;

	for (amrex::MFIter mfi(state_new_cc_[lev]); mfi.isValid(); ++mfi) {
		const amrex::Box &box = mfi.validbox();
		auto const &tag = tags.array(mfi);
		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const Real x = xlo + (static_cast<Real>(i) + 0.5) * dx;
			if (x < x_refine) {
				tag(i, j, k) = amrex::TagBox::SET;
			}
		});
	}
}

// Component layout of the per-cell shielding-column scratch MultiFab (Phase 1 output).
namespace ncol_comp
{
constexpr int H2 = 0;
constexpr int CO = 1;
constexpr int C = 2;
constexpr int H = 3;
constexpr int n = 4;
} // namespace ncol_comp

// AMR-aware chemistry burn, split into two phases so the burn can be parallelized:
//   Phase 1 (deterministic device prefix-sum): compute each cell's cumulative shielding
//     columns from the PRE-burn species, walking the composite grid face->interior
//     (finest->coarsest, covered cells skipped via makeFineMask). Each level is an exclusive
//     scan (amrex::Scan, bit-reproducible) plus a serial inter-level carry. Because the
//     columns come from the pre-burn state, they do not depend on the order cells are burned
//     -> Phase 2 is order-independent and parallelizable. Stored in a 4-component scratch MultiFab.
//   Phase 2 (burn): each cell reads its own columns and burns. Serial today, delivering
//     the columns through the network global network_rp::ncol_*. Once the GOW RHS reads
//     state.aux[] instead, define PDR_PARALLEL_NCOL_AUX (and NumAux>=4) to switch to the
//     ParallelFor path, which hands each cell its columns via burn_t::aux -> thread/GPU
//     safe, no shared global.
// The built-in parallel photochem burner is disabled ([photochemistry] enabled=0).
template <> void QuokkaSimulation<PDRTest>::computeAfterTimestep()
{
	// advance chemistry by the same dt the simulation just took
	const Real t_now = tNew_[0];
	const Real dt = t_now - userData_.t_prev_;
	userData_.t_prev_ = t_now;
	if (dt <= 0.0) {
		return;
	}

	constexpr int nBands = Physics_Traits<PDRTest>::nGroups;
	constexpr int nRadVars = Physics_NumVars::numRadVarsPerGroup;
	std::array<Real, nBands> quanta{};
	for (int g = 0; g < nBands; ++g) {
		quanta[g] = RadSystem<PDRTest>::GetChemBandQuanta(g);
	}
	const Real c_hat = RadSystem_Traits<PDRTest>::c_hat_over_c * C::c_light;
	const int finest = finestLevel();

	// per-level covered-cell masks (uncovered=1, covered=0), reused by both phases
	amrex::Vector<amrex::iMultiFab> mask(finest + 1);
	for (int lev = 0; lev < finest; ++lev) {
		mask[lev] = amrex::makeFineMask(state_new_cc_[lev], state_new_cc_[lev + 1], amrex::IntVect(0), refRatio(lev), geom[lev].periodicity(),
						/*uncovered=*/1, /*covered=*/0);
	}
	// ---- Phase 1: cumulative shielding columns (cm^-2) per cell, from PRE-burn species,
	// walked face->interior over the composite grid (deterministic device prefix-sum; no chemistry).
	amrex::Vector<amrex::MultiFab> ncol_mf(finest + 1);
	for (int lev = 0; lev <= finest; ++lev) {
		ncol_mf[lev].define(boxArray(lev), DistributionMap(lev), ncol_comp::n, 0);
		ncol_mf[lev].setVal(0.0);
	}
	// Deterministic device prefix-sum (amrex::Scan -> CUB/rocPRIM on GPU, serial on CPU):
	// walk levels finest->coarsest with an inter-level carry; within a level exclusive-scan
	// n_s*dx along x. Column c at cell i:  N_c(i) = carry_c + exclScan_c(i) + 0.5*contrib_c(i).
	// Keeps the state on the device (no host access -> no managed-memory dependency) and is
	// bit-reproducible run-to-run. Assumes ONE box per level (1-D: max_grid_size >= n_cell);
	// covered cells contribute 0, so each physical location is counted once on the composite.
	{
		// column index -> species number density at cell i (H column = H-nucleus sum)
		auto column_ndens = [] AMREX_GPU_DEVICE(amrex::Array4<Real const> const &s, int i, int c) -> Real {
			auto nsp = [&](int sp) { return s(i, 0, 0, HydroSystem<PDRTest>::scalar0_index + sp) / spmasses[sp]; };
			switch (c) {
				case ncol_comp::H2:
					return nsp(3);
				case ncol_comp::CO:
					return nsp(9);
				case ncol_comp::C:
					return nsp(7);
				default: // H nuclei: H + H+ + 2 H2 + 2 H2+ + HCO+ + CH + OH + 3 H3+
					return nsp(0) + nsp(1) + 2.0 * nsp(3) + 2.0 * nsp(4) + nsp(10) + nsp(14) + nsp(15) + 3.0 * nsp(16);
			}
		};

		// scratch sized to the largest level (level 0), reused across levels and columns
		int maxN = 0;
		for (int lev = 0; lev <= finest; ++lev) {
			AMREX_ALWAYS_ASSERT_WITH_MESSAGE(boxArray(lev).size() == 1,
							 "PDR Phase-1 scan assumes one box per level (set max_grid_size >= n_cell)");
			maxN = std::max(maxN, static_cast<int>(boxArray(lev).numPts()));
		}
		amrex::Gpu::DeviceVector<Real> contrib(maxN);
		amrex::Gpu::DeviceVector<Real> excl(maxN);
		Real *pc = contrib.dataPtr();
		Real *pe = excl.dataPtr();

		std::array<Real, ncol_comp::n> carry{}; // carry_in per column; grows face->interior
		for (int lev = finest; lev >= 0; --lev) {
			const Real dxlev = geom[lev].CellSizeArray()[0];
			const int has_mask = (lev < finest) ? 1 : 0;
			for (amrex::MFIter iter(state_new_cc_[lev]); iter.isValid(); ++iter) {
				const amrex::Box &bx = iter.validbox();
				const int ilo = bx.smallEnd(0);
				const int n = bx.length(0);
				auto const &state = state_new_cc_[lev].const_array(iter);
				auto const &ncol = ncol_mf[lev].array(iter);
				amrex::Array4<int const> m = has_mask ? mask[lev].const_array(iter) : amrex::Array4<int const>{};

				for (int c = 0; c < ncol_comp::n; ++c) {
					// 1) per-cell contribution n_c*dx (covered cells contribute 0)
					amrex::ParallelFor(n, [=] AMREX_GPU_DEVICE(int t) noexcept {
						const int i = ilo + t;
						const bool covered = (has_mask != 0) && (m(i, 0, 0) == 0);
						pc[t] = covered ? 0.0 : column_ndens(state, i, c) * dxlev;
					});
					// 2) exclusive prefix sum of pc -> pe, plus the level total for the inter-level carry.
					const Real carry_c = carry[c];
					Real total = 0.0;
#ifdef AMREX_USE_GPU
					// GPU: amrex::Scan (CUB/rocPRIM) accumulates in the element type -> correct.
					total = amrex::Scan::ExclusiveSum(n, pc, pe, amrex::Scan::retSum);
#else
					// CPU: amrex::Scan::ExclusiveSum's host path is std::exclusive_scan(in,in+n,out,0),
					// whose accumulator takes the type of the literal `0` (int); the ~1e18 column
					// contributions overflow it to INT_MIN and the prefix comes back garbage. Build the
					// exclusive prefix explicitly in Real instead (1-D single box, deterministic).
					{
						Real running = 0.0;
						for (int t = 0; t < n; ++t) {
							pe[t] = running;
							running += pc[t];
						}
						total = running;
					}
#endif
					// 3) N_c(i) = carry + exclScan + half-self cell
					amrex::ParallelFor(n, [=] AMREX_GPU_DEVICE(int t) noexcept { ncol(ilo + t, 0, 0, c) = carry_c + pe[t] + 0.5 * pc[t]; });
					carry[c] += total; // inter-level carry: serial over ~4 levels, deterministic order
				}
			}
		}
	}

	// ---- Phase 2: burn each (uncovered) cell using its stored columns.
	// Each cell is independent here (its shielding columns are already in ncol_mf/aux), so
	// the burn threads over tiles. OpenMP only in the PDR_PARALLEL_NCOL_AUX build -- the
	// #else path writes the shared network_rp::ncol_* globals per cell and must stay serial.
	// The domain is a single box per level, so tile x explicitly (the default tile size does
	// not split x in 1-D); on GPU keep the whole box in one ParallelFor launch.
#ifdef AMREX_USE_GPU
	const amrex::IntVect burn_tile(AMREX_D_DECL(1024000, 1024000, 1024000));
#else
	const amrex::IntVect burn_tile(AMREX_D_DECL(8, 1024000, 1024000));
#endif
	for (int lev = finest; lev >= 0; --lev) {
		amrex::MultiFab &mf = state_new_cc_[lev];
#if defined(PDR_PARALLEL_NCOL_AUX) && (NAUX_NET >= 4) && defined(AMREX_USE_OMP)
#pragma omp parallel
#endif
		for (amrex::MFIter iter(mf, amrex::MFItInfo().EnableTiling(burn_tile)); iter.isValid(); ++iter) {
			const amrex::Box &bx = iter.tilebox();
			auto const &state = mf.array(iter);
			auto const &ncol = ncol_mf[lev].const_array(iter);
			amrex::Array4<int const> m = (lev < finest) ? mask[lev].const_array(iter) : amrex::Array4<int const>{};

#if defined(PDR_PARALLEL_NCOL_AUX) && (NAUX_NET >= 4)
			// Parallel path: columns delivered per-cell via burn_t::aux (no shared global).
			// Requires the GOW RHS to read state.aux[0..3] for ncol_{H2,CO,C,H}.
			amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				if ((lev < finest) && (m(i, j, k) == 0)) {
					return;
				}
				burn_t cs;
				cs.success = true;
				cs.c_hat = c_hat;
				cs.aux[ncol_comp::H2] = ncol(i, j, k, ncol_comp::H2);
				cs.aux[ncol_comp::CO] = ncol(i, j, k, ncol_comp::CO);
				cs.aux[ncol_comp::C] = ncol(i, j, k, ncol_comp::C);
				cs.aux[ncol_comp::H] = ncol(i, j, k, ncol_comp::H);
				const Real rho = state(i, j, k, HydroSystem<PDRTest>::density_index);
				for (int nn = 0; nn < NumSpec; ++nn) {
					cs.xn[nn] = state(i, j, k, HydroSystem<PDRTest>::scalar0_index + nn) / spmasses[nn];
				}
				const Real Eint = quokka::EOS<PDRTest>::ComputeEintFromEgas(
				    rho, state(i, j, k, HydroSystem<PDRTest>::x1Momentum_index), state(i, j, k, HydroSystem<PDRTest>::x2Momentum_index),
				    state(i, j, k, HydroSystem<PDRTest>::x3Momentum_index), state(i, j, k, HydroSystem<PDRTest>::energy_index), 0.0);
				cs.rho = rho;
				cs.e = Eint / rho;
				for (int g = 0; g < nBands; ++g) {
					const Real E_band = state(i, j, k, RadSystem<PDRTest>::radEnergy_index + nRadVars * g);
					cs.rn[0 + MicrophysicsNumRadVarsPerGroup * g] = amrex::max(E_band / quanta[g], 0.0_rt);
					cs.rn[1 + MicrophysicsNumRadVarsPerGroup * g] = 1.0_rt;
				}
				eos(eos_input_re, cs);
				quokka::photochemistry::photochem_burner(cs, dt);
				for (int nn = 0; nn < NumSpec; ++nn) {
					cs.xn[nn] = amrex::max(cs.xn[nn], small_x);
					state(i, j, k, HydroSystem<PDRTest>::scalar0_index + nn) = cs.xn[nn] * spmasses[nn];
				}
				for (int g = 0; g < nBands; ++g) {
					const Real ng = amrex::max(cs.rn[0 + MicrophysicsNumRadVarsPerGroup * g], 0.0_rt);
					state(i, j, k, RadSystem<PDRTest>::radEnergy_index + nRadVars * g) = amrex::max(ng * quanta[g], Erad_floor_val);
					const Real ff = cs.rn[1 + MicrophysicsNumRadVarsPerGroup * g];
					state(i, j, k, RadSystem<PDRTest>::x1RadFlux_index + nRadVars * g) *= ff;
					state(i, j, k, RadSystem<PDRTest>::x2RadFlux_index + nRadVars * g) *= ff;
					state(i, j, k, RadSystem<PDRTest>::x3RadFlux_index + nRadVars * g) *= ff;
				}
				eos(eos_input_rt, cs);
				const Real Eint_new = cs.e * cs.rho;
				state(i, j, k, HydroSystem<PDRTest>::density_index) = g_rho0;
				state(i, j, k, HydroSystem<PDRTest>::x1Momentum_index) = 0.0;
				state(i, j, k, HydroSystem<PDRTest>::x2Momentum_index) = 0.0;
				state(i, j, k, HydroSystem<PDRTest>::x3Momentum_index) = 0.0;
				state(i, j, k, HydroSystem<PDRTest>::internalEnergy_index) = Eint_new;
				state(i, j, k, HydroSystem<PDRTest>::energy_index) = Eint_new;
			});
#else
			// Serial path: columns delivered via the network global (set per cell).
			const auto lo = amrex::lbound(bx);
			const auto hi = amrex::ubound(bx);
			for (int k = lo.z; k <= hi.z; ++k) {
				for (int j = lo.y; j <= hi.y; ++j) {
					for (int i = lo.x; i <= hi.x; ++i) {
						if ((lev < finest) && (m(i, j, k) == 0)) {
							continue;
						}
						network_rp::ncol_H2 = ncol(i, j, k, ncol_comp::H2);
						network_rp::ncol_CO = ncol(i, j, k, ncol_comp::CO);
						network_rp::ncol_C = ncol(i, j, k, ncol_comp::C);
						network_rp::ncol_H = ncol(i, j, k, ncol_comp::H);

						const Real rho = state(i, j, k, HydroSystem<PDRTest>::density_index);
						burn_t cs;
						cs.success = true;
						cs.c_hat = c_hat;
						for (int nn = 0; nn < NumSpec; ++nn) {
							cs.xn[nn] = state(i, j, k, HydroSystem<PDRTest>::scalar0_index + nn) / spmasses[nn];
						}
						const Real Eint =
						    quokka::EOS<PDRTest>::ComputeEintFromEgas(rho, state(i, j, k, HydroSystem<PDRTest>::x1Momentum_index),
											      state(i, j, k, HydroSystem<PDRTest>::x2Momentum_index),
											      state(i, j, k, HydroSystem<PDRTest>::x3Momentum_index),
											      state(i, j, k, HydroSystem<PDRTest>::energy_index), 0.0);
						cs.rho = rho;
						cs.e = Eint / rho;
						for (int g = 0; g < nBands; ++g) {
							const Real E_band = state(i, j, k, RadSystem<PDRTest>::radEnergy_index + nRadVars * g);
							cs.rn[0 + MicrophysicsNumRadVarsPerGroup * g] = amrex::max(E_band / quanta[g], 0.0_rt);
							cs.rn[1 + MicrophysicsNumRadVarsPerGroup * g] = 1.0_rt;
						}
						eos(eos_input_re, cs);
						quokka::photochemistry::photochem_burner(cs, dt);
						if (std::isnan(cs.xn[0]) || std::isnan(cs.rho) || std::isnan(cs.rn[0])) {
							amrex::Abort("Burner returned NAN");
						}
						for (int nn = 0; nn < NumSpec; ++nn) {
							cs.xn[nn] = amrex::max(cs.xn[nn], small_x);
							state(i, j, k, HydroSystem<PDRTest>::scalar0_index + nn) = cs.xn[nn] * spmasses[nn];
						}
						for (int g = 0; g < nBands; ++g) {
							const Real ng = amrex::max(cs.rn[0 + MicrophysicsNumRadVarsPerGroup * g], 0.0_rt);
							state(i, j, k, RadSystem<PDRTest>::radEnergy_index + nRadVars * g) =
							    amrex::max(ng * quanta[g], Erad_floor_val);
							const Real ff = cs.rn[1 + MicrophysicsNumRadVarsPerGroup * g];
							state(i, j, k, RadSystem<PDRTest>::x1RadFlux_index + nRadVars * g) *= ff;
							state(i, j, k, RadSystem<PDRTest>::x2RadFlux_index + nRadVars * g) *= ff;
							state(i, j, k, RadSystem<PDRTest>::x3RadFlux_index + nRadVars * g) *= ff;
						}
						eos(eos_input_rt, cs);
						const Real Eint_new = cs.e * cs.rho;
						state(i, j, k, HydroSystem<PDRTest>::density_index) = g_rho0;
						state(i, j, k, HydroSystem<PDRTest>::x1Momentum_index) = 0.0;
						state(i, j, k, HydroSystem<PDRTest>::x2Momentum_index) = 0.0;
						state(i, j, k, HydroSystem<PDRTest>::x3Momentum_index) = 0.0;
						state(i, j, k, HydroSystem<PDRTest>::internalEnergy_index) = Eint_new;
						state(i, j, k, HydroSystem<PDRTest>::energy_index) = Eint_new;
					}
				}
			}
#endif
		}
	}

	// make the coarse (covered) cells consistent with the burned fine cells
	for (int lev = finest; lev > 0; --lev) {
		amrex::average_down(state_new_cc_[lev], state_new_cc_[lev - 1], 0, state_new_cc_[lev].nComp(), refRatio(lev - 1));
	}
}

auto problem_main() -> int
{
	double max_time = 2.0e13; // s; long enough to relax the PDR structure (override with pdr.stop_time)
	amrex::ParmParse const pp("pdr");
	pp.query("stop_time", max_time);
	const int max_timesteps = 500000;

	// Boundary conditions: illuminated lo-x face is a custom radiation inflow
	// (ext_dir -> setCustomBoundaryConditions); hi-x is outflow (foextrap);
	// transverse directions are periodic (singleton in 1-D).
	constexpr int nvars = RadSystem<PDRTest>::nvar_;
	amrex::Vector<amrex::BCRec> BCs_cc(nvars);
	for (int n = 0; n < nvars; ++n) {
		BCs_cc[n].setLo(0, amrex::BCType::ext_dir);  // custom radiation inflow at x=0
		BCs_cc[n].setHi(0, amrex::BCType::foextrap); // outflow at x=Lx
		for (int i = 1; i < AMREX_SPACEDIM; ++i) {
			BCs_cc[n].setLo(i, amrex::BCType::int_dir); // periodic
			BCs_cc[n].setHi(i, amrex::BCType::int_dir);
		}
	}

	// Domain length is set by the density: the slab must span a fixed optical depth
	// Av_max = Lx * n_H / N_H_per_Av, so Lx = Av_max * N_H_per_Av / n_H. Derive it from
	// pdr.ninit here and override geometry.prob_hi[0] BEFORE the Geometry is built (in the
	// QuokkaSimulation constructor), so changing the density never leaves the box too short.
	{
		constexpr double N_H_per_Av = 1.87e21; // H column per unit Av (cm^-2)
		double ninit_geom = 1000.0;
		pp.query("ninit", ninit_geom);
		double av_max = 10.0;
		pp.query("Av_max", av_max); // target max optical depth spanned by the slab
		const double Lx = av_max * N_H_per_Av / ninit_geom;

		amrex::ParmParse pp_geom("geometry");
		amrex::Vector<amrex::Real> prob_hi;
		pp_geom.getarr("prob_hi", prob_hi); // [Lx, 1, 1] from the input
		prob_hi[0] = Lx;
		pp_geom.addarr("prob_hi", prob_hi); // last value wins when the Geometry queries it
		amrex::Print() << "PDR: n_H = " << ninit_geom << ", Av_max = " << av_max << " -> Lx = " << Lx << " cm\n";
	}

	QuokkaSimulation<PDRTest> sim(BCs_cc);

	// Load the GOW cooling + shielding tables now that amrex::Initialize() is up.
	gow_tables::registerGowTables();

	sim.radiationReconstructionOrder_ = 1; // PPM
	sim.radiationCflNumber_ = 0.3;
	// Cap the step so the per-cell chemistry burn stays integrable: a coarse grid's larger
	// CFL dt can push a cell through a stiff transition in one step and fail VODE. Combined
	// with [integrator] use_burn_retry. Override with pdr.max_dt.
	sim.maxDt_ = 2.0e9;
	pp.query("max_dt", sim.maxDt_);
	sim.maxTimesteps_ = max_timesteps;
	sim.stopTime_ = max_time;
	sim.plotfileInterval_ = -1;
	sim.checkpointInterval_ = -1;

	sim.setInitialConditions();
	sim.evolve();

	// write the equilibrium profile at native (composite) resolution: walk levels
	// finest->coarsest, keep only cells not covered by a finer level, sort by depth x.
	int status = 1;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		constexpr int nCols = 2 + Physics_Traits<PDRTest>::nGroups + NumSpec; // x, Tgas, Erad[g], species
		std::vector<std::array<double, nCols>> rows;
		bool finite = true;

		const int finest = sim.finestLevel();
		for (int lev = finest; lev >= 0; --lev) {
			const amrex::MultiFab &mf = sim.getNewMF_cc()[lev];
			const Real xlo = sim.Geom(lev).ProbLoArray()[0];
			const Real dx = sim.Geom(lev).CellSizeArray()[0];

			amrex::iMultiFab mask;
			if (lev < finest) {
				mask = amrex::makeFineMask(mf, sim.getNewMF_cc()[lev + 1], amrex::IntVect(0), sim.refRatio(lev), sim.Geom(lev).periodicity(),
							   /*uncovered=*/1, /*covered=*/0);
			}

			for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
				const amrex::Box &box = mfi.validbox();
				auto const &state = mf.const_array(mfi);
				amrex::Array4<int const> maskarr = (lev < finest) ? mask.const_array(mfi) : amrex::Array4<int const>{};
				const auto lo = amrex::lbound(box);
				const auto hi = amrex::ubound(box);
				for (int k = lo.z; k <= hi.z; ++k) {
					for (int j = lo.y; j <= hi.y; ++j) {
						for (int i = lo.x; i <= hi.x; ++i) {
							if ((lev < finest) && (maskarr(i, j, k) == 0)) {
								continue; // covered by a finer level
							}
							const Real x = xlo + (static_cast<Real>(i) + 0.5) * dx;
							const Real rho = state(i, j, k, HydroSystem<PDRTest>::density_index);
							const Real Eint = state(i, j, k, HydroSystem<PDRTest>::internalEnergy_index);
							std::array<Real, NumSpec> numdens = {};
							for (int nn = 0; nn < NumSpec; ++nn) {
								numdens[nn] = state(i, j, k, HydroSystem<PDRTest>::scalar0_index + nn) / spmasses[nn];
							}
							const Real Tgas = compute_Tgas(rho, Eint, numdens);

							std::array<double, nCols> row{};
							int c = 0;
							row[c++] = x;
							row[c++] = Tgas;
							for (int g = 0; g < Physics_Traits<PDRTest>::nGroups; ++g) {
								row[c++] = state(i, j, k,
										 RadSystem<PDRTest>::radEnergy_index + Physics_NumVars::numRadVarsPerGroup * g);
							}
							for (int nn = 0; nn < NumSpec; ++nn) {
								row[c++] = numdens[nn];
							}
							rows.push_back(row);

							finite = finite && std::isfinite(Tgas) && (Tgas > 0.0);
							for (int nn = 0; nn < NumSpec; ++nn) {
								finite = finite && std::isfinite(numdens[nn]) && (numdens[nn] >= 0.0);
							}
						}
					}
				}
			}
		}

		std::sort(rows.begin(), rows.end(), [](auto const &a, auto const &b) { return a[0] < b[0]; });

		std::filesystem::create_directories("pdr_State");
		std::ofstream out("pdr_State/profile.txt");
		out << "x Tgas Erad0 Erad1 Erad2 H Hp e H2 H2p He Hep C Cp CO HCOp O Si Sip CH OH H3p Op\n";
		for (auto const &row : rows) {
			for (int c = 0; c < nCols; ++c) {
				out << (c == 0 ? "" : " ") << row[c];
			}
			out << "\n";
		}
		out.close();

		amrex::Print() << "PDR: n_H = " << sim.userData_.ninit << ", finest level = " << finest << ", ncell(composite) = " << rows.size() << "\n";
		amrex::Print() << "Wrote pdr_State/profile.txt\n";
		if (finite) {
			status = 0;
		}
	}

	amrex::Print() << "Finished." << '\n';
	return status;
}
