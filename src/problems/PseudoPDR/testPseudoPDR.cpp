//==============================================================================
// TwoMomentRad - a radiation transport library for patch-based AMR codes
// Copyright 2020 Benjamin Wibking.
// Released under the MIT license. See LICENSE file included in the GitHub repo.
//==============================================================================
/// \file testPseudoPDR.cpp
/// \brief 1-D "pseudo" photodissociation region (PDR) using the Gong, Ostriker
///        & Wolfire (2017) chemistry + thermal network (gow17, 18 species).
///
///        Each cell is an independent one-zone chemistry/thermal problem at a
///        FIXED gas density n_H, laid out along a log-spaced column-density
///        (N_H) grid. The FUV field enters exactly as in the GOW one-zone test:
///        a constant Draine field chi attenuated by the local visual extinction
///        Av (dust continuum) via the network photorates chi*exp(-gamma*Av).
///        This is NOT radiation transport -- there is no photon field evolved.
///        The "isotropic source" is emulated by evaluating Av at 2*N_H
///        (60-degree incidence path length; see readme.md), which is set
///        PER CELL from that cell's column position.
///
///        Since the GOW network reads the single global scalar network_rp::Av,
///        the per-cell attenuation is driven by a custom serial (CPU) burn loop
///        that assigns network_rp::Av before burning each cell -- the built-in
///        parallel chemistry burner (which shares one global Av across all
///        cells) is intentionally disabled.
#include "util/BC.hpp"
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "AMReX.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"

#include "QuokkaSimulation.hpp"
#include "SimulationData.hpp"
#include "chemistry/Chemistry.hpp"
#include "hydro/hydro_system.hpp"
#include "radiation/radiation_system.hpp"
#include "util/fextract.hpp"

#include "actual_eos_data.H"
#include "burn_type.H"
#include "eos.H"
#include "extern_parameters.H"
#include "network.H"

#include <actual_rhs.H>
#include <gow_interp.H>
#include <gow_tables.hpp>
#include <integrator_data.H>

using amrex::Real;
struct PseudoPDRTest {}; // dummy type to allow compile-time polymorphism via template specialization

template <> struct Physics_Traits<PseudoPDRTest> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = NumSpec;		     // 18 chemical species
	static constexpr int numPassiveScalars = numMassScalars + 0; // only mass scalars
};

template <> struct SimulationData<PseudoPDRTest> {
	amrex::Real small_temp;
	amrex::Real small_dens;
	amrex::Real temperature; // initial gas temperature (K)
	amrex::Real ninit;	 // fixed gas number density n_H (cm^-3)
	amrex::Real metallicity; // scales metal initial abundances
	amrex::Real chi;	 // FUV field (Draine units)

	// column-density (Av) sweep grid
	amrex::Real NH_min;	// surface column density (cm^-2)
	amrex::Real NH_total;	// deepest column density (cm^-2)
	amrex::Real iso_factor; // path-length factor: 2 = isotropic 60-deg incidence
	amrex::Real NH_over_Av; // N_H per unit Av at Zd=1 (= 1.87e21 cm^-2)

	std::array<amrex::Real, NumSpec> rel_abundance; // initial abundance relative to n_init
	amrex::Real t_prev_ = 0.0;			// previous sim time (for burn dt)
};

// Column density (cm^-2) at cell index i on a log-spaced N_H grid of ngrid cells.
// Matches the reference slab grid (slab.cpp SolveEq): NH[0] = NH_min,
// NH[i] = NH_min * (NH_total/NH_min)^(i/ngrid) (note ngrid, not ngrid-1, so the
// deepest zone sits one step short of NH_total, exactly as in colH_arr.dat).
AMREX_INLINE auto columnDensity(int i, int ngrid, Real NH_min, Real NH_total) -> Real
{
	const Real frac = (ngrid > 0) ? (static_cast<Real>(i) / static_cast<Real>(ngrid)) : 0.0;
	return NH_min * std::pow(NH_total / NH_min, frac);
}

// Local visual extinction seen by cell i (dust continuum). The isotropic
// approximation uses an effective column of iso_factor*N_H (see readme.md).
AMREX_INLINE auto cellAv(int i, int ngrid, const SimulationData<PseudoPDRTest> &d) -> Real
{
	const Real NH = columnDensity(i, ngrid, d.NH_min, d.NH_total);
	return d.iso_factor * NH / d.NH_over_Av;
}

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

// Initial number densities of the surface zone, set up as in the reference
// marched slab with Andrea's initial abundances (andrea_sweep.cpp): ninit is the
// total H-nucleus density, the C, O, Si and He totals per H nucleus are fixed
// (metals scaled by metallicity; the seeds are not), and the species carrying
// each element total (H, C, O, Si, He) take whatever the seeds leave, exactly as
// gow17's ghost species (gow17::SetGhostSpec_). The electrons start equal to the
// cation density so the initial state is charge neutral; after t = 0 every
// species, e- included, is evolved by the network.
auto initial_numdens(const SimulationData<PseudoPDRTest> &ud) -> std::array<Real, NumSpec>
{
	constexpr Real xC_tot = 1.6e-4;
	constexpr Real xO_tot = 3.2e-4;
	constexpr Real xSi_tot = 1.7e-6;
	constexpr Real xHe_tot = 0.1;

	std::array<Real, NumSpec> x = ud.rel_abundance;
	const Real met = ud.metallicity;

	// index order 0=H,1=H+,2=e-,3=H2,4=H2+,5=He,6=He+,7=C,8=C+,9=CO,10=HCO+,
	// 11=O,12=Si,13=Si+,14=CH,15=OH,16=H3+,17=O+
	x[0] = 1.0 - (x[1] + 2.0 * x[3] + 2.0 * x[4] + x[10] + x[14] + x[15] + 3.0 * x[16]); // H
	x[5] = xHe_tot - x[6];								     // He
	x[7] = met * xC_tot - (x[8] + x[9] + x[10] + x[14]);				     // C
	x[11] = met * xO_tot - (x[9] + x[10] + x[15] + x[17]);				     // O
	x[12] = met * xSi_tot - x[13];							     // Si
	x[2] = x[1] + x[4] + x[6] + x[8] + x[10] + x[13] + x[16] + x[17];		     // e-

	std::array<Real, NumSpec> numdens{};
	for (int n = 0; n < NumSpec; ++n) {
		numdens[n] = x[n] * ud.ninit;
	}
	return numdens;
}

template <> void QuokkaSimulation<PseudoPDRTest>::preCalculateInitialConditions()
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

	// column-density sweep grid (defaults from readme.md; Zd=1)
	userData_.NH_over_Av = 1.87e21;
	pp.query("NH_over_Av", userData_.NH_over_Av);
	userData_.NH_min = 1.0e17; // Av ~ 5.3e-5
	pp.query("NH_min", userData_.NH_min);
	userData_.NH_total = 1.87e22; // Av = 10
	pp.query("NH_total", userData_.NH_total);
	userData_.iso_factor = 2.0; // isotropic 60-deg incidence
	pp.query("iso_factor", userData_.iso_factor);
	gow_co_shield::iso_factor_CO = userData_.iso_factor; // CO self-shielding uses the same path factor

	// metallicity/chi are network runtime parameters; read here only to tag output
	userData_.metallicity = network_rp::metallicity;
	userData_.chi = network_rp::chi;

	// Initial seed abundances per H nucleus (Andrea's setup, andrea_sweep.cpp:
	// partly molecular start with H2 = 0.1, C+ = 1e-4, no H+), in xn[] index
	// order 0=H,1=H+,2=e-,3=H2,4=H2+,5=He,6=He+,7=C,8=C+,9=CO,10=HCO+,11=O,
	// 12=Si,13=Si+,14=CH,15=OH,16=H3+,17=O+. Entries marked "total" are set
	// from the element totals / charge neutrality in initial_numdens. Species
	// that are zero in the reference are seeded at 1e-40.
	userData_.rel_abundance = {
	    0.0,	  // H   (total)
	    1.0e-40,	  // H+
	    0.0,	  // e-  (cation density)
	    0.1,	  // H2
	    1.0e-40,	  // H2+
	    0.0,	  // He  (total)
	    1.450654e-08, // He+
	    0.0,	  // C   (total)
	    1.0e-4,	  // C+
	    1.0e-7,	  // CO
	    1.0e-40,	  // HCO+
	    0.0,	  // O   (total)
	    0.0,	  // Si  (total)
	    1.0e-40,	  // Si+
	    1.0e-40,	  // CH
	    1.0e-40,	  // OH
	    2.681411e-07, // H3+
	    1.0e-40,	  // O+
	};

	eos_init(userData_.small_temp, userData_.small_dens);
	network_init();
}

template <> void QuokkaSimulation<PseudoPDRTest>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	// every cell starts from the SAME (surface) one-zone IC at fixed n_H; the
	// depth dependence enters only through the per-cell Av in the burn loop.
	burn_t state;
	const std::array<Real, NumSpec> numdens = initial_numdens(userData_);

	state.T = userData_.temperature;

	Real rhotot = 0.0_rt;
	for (int n = 0; n < NumSpec; ++n) {
		state.xn[n] = numdens[n];
		rhotot += state.xn[n] * spmasses[n]; // spmasses defined in EOS
	}
	state.rho = rhotot;

	// call the EOS to set initial internal energy e
	eos(eos_input_rt, state);

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		Real const rho = state.rho;
		Real const xmom = 0;
		Real const ymom = 0;
		Real const zmom = 0;
		Real const Eint = rho * state.e;

		static_assert(!Physics_Traits<PseudoPDRTest>::is_mhd_enabled, "MHD is enabled; pass magnetic_energy instead of 0.0");
		Real const Egas = quokka::EOS<PseudoPDRTest>::ComputeEgasFromEint(rho, xmom, ymom, zmom, Eint, 0.0);

		state_cc(i, j, k, HydroSystem<PseudoPDRTest>::energy_index) = Egas;
		state_cc(i, j, k, HydroSystem<PseudoPDRTest>::internalEnergy_index) = Eint;
		state_cc(i, j, k, HydroSystem<PseudoPDRTest>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<PseudoPDRTest>::x1Momentum_index) = xmom;
		state_cc(i, j, k, HydroSystem<PseudoPDRTest>::x2Momentum_index) = ymom;
		state_cc(i, j, k, HydroSystem<PseudoPDRTest>::x3Momentum_index) = zmom;

		for (int nn = 0; nn < NumSpec; ++nn) {
			// scalar indices carry partial densities (not number densities)
			state_cc(i, j, k, HydroSystem<PseudoPDRTest>::scalar0_index + nn) = state.xn[nn] * spmasses[nn];
		}
	});
}

// Custom per-cell chemistry burn. Runs serially on the CPU so that the single
// global network_rp::Av can be assigned per cell before each burn (the GOW RHS
// reads network_rp::Av, so this is the only way to give each column-density
// zone its own dust attenuation without editing the network). Mirrors the
// mass-fraction <-> number-density bookkeeping of quokka::chemistry::computeChemistry.
template <> void QuokkaSimulation<PseudoPDRTest>::computeAfterTimestep()
{
	amrex::MultiFab &mf = state_new_cc_[0];
	const int ngrid = Geom(0).Domain().length(0);

	// advance the chemistry by the same dt the simulation just took
	const Real t_now = tNew_[0];
	const Real dt = t_now - userData_.t_prev_;
	userData_.t_prev_ = t_now;
	if (dt <= 0.0) {
		return;
	}

	// Cumulative shielding columns (cm^-2), accumulated as we march inward
	// low->high N_H. Kept before the isotropic path-length factor; that factor
	// is applied when the per-zone network_rp::ncol_* are set. Species columns:
	// N(H2) [idx 3], N(CO) [idx 9], N(C) [neutral atomic carbon, idx 7].
	Real col_H2 = 0.0;
	Real col_CO = 0.0;
	Real col_C = 0.0;
	const Real inv_nH = 1.0 / userData_.ninit; // x_X = n_X / n_H

	// serial CPU loop: per-cell Av and shielding columns require mutating the
	// global network_rp::* before each burn
	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		const amrex::Box &bx = iter.validbox();
		auto const &state = mf.array(iter);
		const auto lo = amrex::lbound(bx);
		const auto hi = amrex::ubound(bx);

		for (int k = lo.z; k <= hi.z; ++k) {
			for (int j = lo.y; j <= hi.y; ++j) {
				for (int i = lo.x; i <= hi.x; ++i) {
					// Marched slab:  march low->high N_H and start each
					// zone from the previous (less-shielded) zone's converged state.
					// This warm start eases the stiff cold-start->high-Av jump and
					// keeps the elemental budget consistent from zone to zone.
					// (Requires the whole column in one box, i.e. max_grid_size >=
					// n_cell and a single rank -- true for this 1-D test.)
					if (i > lo.x) {
						state(i, j, k, HydroSystem<PseudoPDRTest>::energy_index) =
						    state(i - 1, j, k, HydroSystem<PseudoPDRTest>::energy_index);
						state(i, j, k, HydroSystem<PseudoPDRTest>::internalEnergy_index) =
						    state(i - 1, j, k, HydroSystem<PseudoPDRTest>::internalEnergy_index);
						for (int nn = 0; nn < NumSpec; ++nn) {
							state(i, j, k, HydroSystem<PseudoPDRTest>::scalar0_index + nn) =
							    state(i - 1, j, k, HydroSystem<PseudoPDRTest>::scalar0_index + nn);
						}
					}

					// per-cell dust attenuation (this is the shielding hook)
					network_rp::Av = cellAv(i, ngrid, userData_);

					// Per-zone shielding + cooling columns, matching the reference marched
					// slab (slab.cpp SolveEq): the columns seen by zone i are accumulated
					// from the PREVIOUS zones only (0..i-1) -- a zone is shielded by the
					// material in front of it, not by itself. The current zone's converged
					// contribution is added afterwards with the FORWARD interval
					// dNH = NH[i+1]-NH[i]. Shielding columns carry the isotropic path-length
					// factor (2x); the CO-cooling column and bCO_L use the RAW (un-doubled)
					// columns, as in the reference (SetNCO/SetNH).
					const Real NH_i = columnDensity(i, ngrid, userData_.NH_min, userData_.NH_total);
					const Real NH_next = (i < hi.x) ? columnDensity(i + 1, ngrid, userData_.NH_min, userData_.NH_total) : NH_i;
					const Real dNH = NH_next - NH_i; // forward difference (0 at the last zone)

					// Shielding columns (H2/C self-shielding) carry the isotropic path-length
					// factor (2x, 60-deg incidence). ncol_CO is kept RAW: it feeds BOTH the CO
					// self-shielding lookup AND the CO rotational-cooling escape column. gow17
					// uses the raw column for cooling and only doubles the shielding path, so
					// the isotropic factor for CO *shielding* is re-applied inside the CO shield
					// interpolation (fshield_co_interp2d), not here.
					network_rp::ncol_H2 = userData_.iso_factor * col_H2;
					network_rp::ncol_C = userData_.iso_factor * col_C;
					network_rp::ncol_CO = col_CO; // raw; CO shield re-doubles internally
					// Raw H column of this zone, for the CO-cooling cloud depth L = ncol_H/nH
					// (physical, chemistry-independent -> correct through the C/CO transition).
					network_rp::ncol_H = NH_i;

					// DIAG (env QK_COL_DUMP): per-zone shielding/cooling columns fed to the burn.
					if (std::getenv("QK_COL_DUMP") != nullptr && j == lo.y && k == lo.z && (i % 100 == 0 || i == hi.x)) {
						amrex::Print()
						    << "COL zone=" << i << " NH_i=" << NH_i << " Av=" << network_rp::Av << " ncol_H2=" << network_rp::ncol_H2
						    << " ncol_CO=" << network_rp::ncol_CO << " ncol_C=" << network_rp::ncol_C
						    << " ncol_H=" << network_rp::ncol_H << " dNH=" << dNH << "\n";
					}

					const Real rho = state(i, j, k, HydroSystem<PseudoPDRTest>::density_index);
					const Real xmom = state(i, j, k, HydroSystem<PseudoPDRTest>::x1Momentum_index);
					const Real ymom = state(i, j, k, HydroSystem<PseudoPDRTest>::x2Momentum_index);
					const Real zmom = state(i, j, k, HydroSystem<PseudoPDRTest>::x3Momentum_index);
					const Real Ener = state(i, j, k, HydroSystem<PseudoPDRTest>::energy_index);
					const Real Eint = quokka::EOS<PseudoPDRTest>::ComputeEintFromEgas(rho, xmom, ymom, zmom, Ener, 0.0);

					burn_t chemstate;
					chemstate.success = true;
					std::array<Real, NumSpec> inmfracs = {-1.0};
					for (int nn = 0; nn < NumSpec; ++nn) {
						// scalars are partial densities -> number densities
						const Real massfrac = state(i, j, k, HydroSystem<PseudoPDRTest>::scalar0_index + nn) / rho;
						inmfracs[nn] = massfrac * rho / spmasses[nn];
						chemstate.xn[nn] = inmfracs[nn];
					}
					chemstate.rho = rho;
					chemstate.e = Eint / rho;
					eos(eos_input_re, chemstate);

					// Sub-cycle this zone to equilibrium with a growing sub-step instead of one
					// giant dt. A single huge burn jumps through the fast (~1e3 s) chemical/
					// thermal transient, overshoots to the small_temp floor, then fails (VODE
					// istate=-6). Starting below the transient and doubling dt keeps each
					// sub-burn near the local timescale; once the zone hits its fixed point the
					// later large sub-steps sit on it and are cheap. Drives every zone to
					// equilibrium in a SINGLE hydro step, so the slab is marched once, not
					// re-burned every step.
					{
						constexpr Real t_equil = 7.0e20; // s, >> chemical/thermal relaxation time
						constexpr Real dt_first = 1.0;	 // s, first sub-step (below the transient)
						constexpr Real growth = 2.0;
						// Stop once the zone sits on its fixed point: after t_conv_min, a
						// sub-burn over [t, 2t] that changes T and every species above
						// x_floor by less than conv_tol (relative) means the zone is in
						// equilibrium (cf. the reference SolveEq, which uses 1e-3). The
						// slowest relaxation time is ~1e16 s, so an unconverged zone
						// changes by >~ t/tau ~ 1e-4 per doubling after 1e12 s. Without
						// this the remaining huge sub-steps (dt ~ 1e19-1e20 s) on an
						// already converged zone exhaust ode_max_steps (istate = -4).
						constexpr Real t_conv_min = 1.0e12; // s
						constexpr Real conv_tol = 1.0e-10;
						constexpr Real x_floor = 1.0e-30; // cm^-3
						Real t_acc = 0.0;
						Real dt_sub = dt_first;
						while (t_acc < t_equil) {
							if (t_acc + dt_sub > t_equil) {
								dt_sub = t_equil - t_acc;
							}
							const burn_t prev = chemstate;
							quokka::chemistry::chemburner(chemstate, dt_sub);
							if (std::isnan(chemstate.xn[0]) || std::isnan(chemstate.rho)) {
								amrex::Abort("Burner returned NAN");
							}
							// update T from the integrated e before the next sub-burn: the
							// T left by the integrator comes from its last RHS call, and
							// the next burn recomputes e from (rho, T)
							eos(eos_input_re, chemstate);
							t_acc += dt_sub;
							dt_sub *= growth;

							if (t_acc >= t_conv_min && chemstate.success) {
								Real maxrel = std::abs(chemstate.T - prev.T) / prev.T;
								for (int nn = 0; nn < NumSpec; ++nn) {
									if (prev.xn[nn] > x_floor) {
										maxrel =
										    std::max(maxrel, std::abs(chemstate.xn[nn] - prev.xn[nn]) / prev.xn[nn]);
									}
								}
								if (maxrel < conv_tol) {
									break;
								}
							}
						}
					}

					// positivity only. NO sum-to-one mass-fraction renormalization: this
					// is number-density mode (renormalize_abundances = 0). The burner's
					// evolved xn are the answer; rescaling every species so its mass
					// fraction sums to 1 against a slightly drifted rho uniformly inflates
					// all number densities in the cold molecular zones (Sum C, Sum O ~1.8x
					// too high) and breaks the elemental budget. Write xn back directly.
					for (int nn = 0; nn < NumSpec; ++nn) {
						chemstate.xn[nn] = amrex::max(chemstate.xn[nn], small_x);
					}

					// accumulate this zone's converged contribution to the running columns
					col_H2 += (chemstate.xn[3] * inv_nH) * dNH;
					col_CO += (chemstate.xn[9] * inv_nH) * dNH;
					col_C += (chemstate.xn[7] * inv_nH) * dNH;

					// keep the integrated e; T follows it
					eos(eos_input_re, chemstate);

					const Real dEint = (chemstate.e * chemstate.rho) - Eint;
					state(i, j, k, HydroSystem<PseudoPDRTest>::internalEnergy_index) += dEint;
					state(i, j, k, HydroSystem<PseudoPDRTest>::energy_index) += dEint;
					for (int nn = 0; nn < NumSpec; ++nn) {
						state(i, j, k, HydroSystem<PseudoPDRTest>::scalar0_index + nn) = chemstate.xn[nn] * spmasses[nn];
					}
				}
			}
		}
	}
}

auto problem_main() -> int
{
	const double max_time = 7.0e20; // s; >> chemical/thermal relaxation time
	const int max_timesteps = 500000;

	QuokkaSimulation<PseudoPDRTest> sim;

	// Load the GOW cooling tables now that amrex::Initialize() (and MPI) are up.
	gow_tables::registerGowTables();

	// --- DEBUG (env QK_INTERP_PROBE=1): evaluate every CO/dust interp function and
	// its analytic-Jacobian partials over PseudoPDR's real depth ranges, flagging
	// any NaN/Inf. NCOeff spans 0 (surface, no CO column) to ~1e18 (deep). ---
	if (std::getenv("QK_INTERP_PROBE") != nullptr) {
		auto chk = [](const char *name, double T, double y2, double v) {
			if (!std::isfinite(v)) {
				amrex::Print() << "  !!! NONFINITE " << name << "(T=" << T << ", y=" << y2 << ") = " << v << "\n";
			}
		};
		amrex::Print() << "=== INTERP PROBE ===\n";
		for (const double Tv : {17.0, 33.0, 100.0}) {
			for (const double yv : {0.0, 1.0e5, 1.0e13, 1.0e15, 1.0e18}) {
				amrex::Print() << "IVAL T=" << Tv << " y=" << yv << " l0=" << l0_co_interp1d(Tv) << " llte=" << llte_co_interp2d(Tv, yv)
					       << " nhalf=" << nhalf_co_interp2d(Tv, yv) << " alpha=" << alpha_co_interp2d(Tv, yv) << "\n";
			}
		}
		for (const double T : {10.0, 17.0, 33.0, 100.0, 2000.0}) {
			for (const double y : {0.0, 1.0e-30, 1.0e-5, 1.0e5, 1.0e12, 1.0e18, 1.0e25}) {
				chk("llte", T, y, llte_co_interp2d(T, y));
				chk("llte_p0", T, y, llte_co_interp2d_partial_0(T, y));
				chk("llte_p1", T, y, llte_co_interp2d_partial_1(T, y));
				chk("nhalf", T, y, nhalf_co_interp2d(T, y));
				chk("nhalf_p0", T, y, nhalf_co_interp2d_partial_0(T, y));
				chk("nhalf_p1", T, y, nhalf_co_interp2d_partial_1(T, y));
				chk("alpha", T, y, alpha_co_interp2d(T, y));
				chk("alpha_p0", T, y, alpha_co_interp2d_partial_0(T, y));
				chk("alpha_p1", T, y, alpha_co_interp2d_partial_1(T, y));
			}
			chk("l0", T, 0, l0_co_interp1d(T));
			chk("l0_p0", T, 0, l0_co_interp1d_partial_0(T));
		}
		for (const double nH : {1.0, 100.0, 1000.0, 3000.0, 1.0e6}) {
			for (const double T : {10.0, 33.0, 100.0, 2000.0}) {
				chk("psigd", T, nH, psigd_coll_interp2d(nH, T));
				chk("psigd_p0", T, nH, psigd_coll_interp2d_partial_0(nH, T));
				chk("psigd_p1", T, nH, psigd_coll_interp2d_partial_1(nH, T));
			}
		}
		for (const double nco : {0.0, 1.0e5, 1.0e15, 1.0e19}) {
			for (const double nh2 : {0.0, 1.0e15, 1.0e21, 1.0e23}) {
				chk("fshield_co", nco, nh2, fshield_co_interp2d(nco, nh2));
			}
		}

		// Evaluate actual_rhs + actual_jac at representative surface/deep states and
		// scan every entry for NaN/Inf, to locate what the analytic Jacobian NaNs on.
		auto scan_state = [](const char *label, double Av, double nH2, double nCO, double nC, double nH, double T,
				     const std::array<double, NumSpec> &x) {
			network_rp::Av = Av;
			network_rp::ncol_H2 = nH2;
			network_rp::ncol_CO = nCO;
			network_rp::ncol_C = nC;
			network_rp::ncol_H = nH;
			burn_t st;
			st.T = T;
			double rho = 0.0;
			for (int n = 0; n < NumSpec; ++n) {
				st.xn[n] = x[n];
				rho += x[n] * spmasses[n];
			}
			st.rho = rho;
			eos(eos_input_rt, st);
			amrex::Array1D<Real, 1, neqs> yd{};
			actual_rhs(st, yd);
			int ydnan = 0;
			for (int i = 1; i <= neqs; ++i) {
				if (!std::isfinite(yd(i))) {
					amrex::Print() << "  [" << label << "] ydot(" << i << ") NONFINITE = " << yd(i) << "\n";
					ydnan++;
				}
			}
			JacNetArray2D jac;
			actual_jac(st, jac);
			int jnan = 0;
			for (int i = 1; i <= neqs; ++i) {
				for (int j = 1; j <= neqs; ++j) {
					if (!std::isfinite(jac(i, j))) {
						amrex::Print() << "  [" << label << "] jac(" << i << "," << j << ") NONFINITE = " << jac(i, j) << "\n";
						jnan++;
					}
				}
			}
			amrex::Print() << "  [" << label << "] T=" << st.T << " rho=" << st.rho << " ydot_nan=" << ydnan << " jac_nan=" << jnan << "\n";

			// Per-column FD of actual_rhs matching actual_jac's convention: species
			// columns -> d/dxn[j-1] at CONSTANT T (sympy holds tgas fixed); energy
			// column -> d/de (perturb e, recompute T via eos_input_re).
			const double T0 = st.T;
			const double e0 = st.e;
			std::array<double, NumSpec> x0{};
			for (int n = 0; n < NumSpec; ++n) {
				x0[n] = st.xn[n];
			}
			auto fd_col = [&](int col, amrex::Array1D<Real, 1, neqs> &out) {
				burn_t pp;
				burn_t pm;
				amrex::Array1D<Real, 1, neqs> yp{};
				amrex::Array1D<Real, 1, neqs> ym{};
				for (int n = 0; n < NumSpec; ++n) {
					pp.xn[n] = x0[n];
					pm.xn[n] = x0[n];
				}
				if (col <= NumSpec) {
					const double d = 1.0e-6 * std::abs(x0[col - 1]) + 1.0e-30;
					pp.T = T0;
					pm.T = T0;
					pp.xn[col - 1] = x0[col - 1] + d;
					pm.xn[col - 1] = x0[col - 1] - d;
					double rp = 0.0;
					double rm = 0.0;
					for (int n = 0; n < NumSpec; ++n) {
						rp += pp.xn[n] * spmasses[n];
						rm += pm.xn[n] * spmasses[n];
					}
					pp.rho = rp;
					pm.rho = rm;
					actual_rhs(pp, yp);
					actual_rhs(pm, ym);
					for (int i = 1; i <= neqs; ++i) {
						out(i) = (yp(i) - ym(i)) / (2.0 * d);
					}
				} else {
					const double d = 1.0e-6 * std::abs(e0) + 1.0e-30;
					double r0 = 0.0;
					for (int n = 0; n < NumSpec; ++n) {
						r0 += x0[n] * spmasses[n];
					}
					pp.rho = r0;
					pm.rho = r0;
					pp.e = e0 + d;
					pm.e = e0 - d;
					eos(eos_input_re, pp);
					eos(eos_input_re, pm);
					actual_rhs(pp, yp);
					actual_rhs(pm, ym);
					for (int i = 1; i <= neqs; ++i) {
						out(i) = (yp(i) - ym(i)) / (2.0 * d);
					}
				}
			};
			int nmis = 0;
			for (int j = 1; j <= neqs; ++j) {
				if (j <= NumSpec && std::abs(x0[j - 1]) < 1.0e-25) {
					continue;
				}
				amrex::Array1D<Real, 1, neqs> fdc{};
				fd_col(j, fdc);
				for (int i = 1; i <= neqs; ++i) {
					const double an = jac(i, j);
					const double fd = fdc(i);
					const double sc = std::abs(an) + std::abs(fd);
					if (sc < 1.0e-18) {
						continue;
					}
					const double rel = std::abs(an - fd) / sc;
					if (rel > 0.2) {
						amrex::Print() << "  [" << label << "] DIFF jac(" << i << "," << j << ") an=" << an << " fd=" << fd
							       << " rel=" << rel << "\n";
						nmis++;
					}
				}
			}
			amrex::Print() << "  [" << label << "] total mismatches=" << nmis << "\n";
		};
		// surface (atomic, no CO column) and deep (molecular, large columns)
		std::array<double, NumSpec> xs{};
		xs[0] = 1000.0;
		xs[1] = 0.1;
		xs[2] = 0.260283;
		xs[3] = 1e-7;
		xs[4] = 1e-37;
		xs[5] = 100.0;
		xs[6] = 1.45065e-5;
		xs[7] = 1e-37;
		xs[8] = 0.16;
		xs[9] = 1e-4;
		xs[10] = 1e-37;
		xs[11] = 0.32;
		xs[12] = 1.7e-3;
		xs[13] = 1e-37;
		xs[14] = 1e-37;
		xs[15] = 1e-37;
		xs[16] = 2.681411e-4;
		xs[17] = 1e-37;
		scan_state("surfaceIC", 1.069519e-4, 0.0, 0.0, 0.0, 1.0e17, 100.0, xs);
		std::array<double, NumSpec> xd{};
		xd[0] = 10.0;
		xd[3] = 495.0;
		xd[5] = 100.0;
		xd[7] = 0.05;
		xd[9] = 0.15;
		xd[11] = 0.17;
		xd[2] = 3e-4;
		for (const double Td : {10.0, 11.0, 12.0, 13.0, 15.0, 17.0, 20.0, 30.0}) {
			scan_state("deep", 20.0, 1.66e22, 2.1e18, 1.5e18, 1.8e22, Td, xd);
		}
		amrex::Print() << "=== INTERP PROBE DONE ===\n";
	}

	sim.reconstructionOrder_ = 3; // PLM
	sim.cflNumber_ = 0.25;
	sim.maxTimesteps_ = max_timesteps;
	sim.stopTime_ = max_time;
	sim.plotfileInterval_ = -1;
	sim.checkpointInterval_ = -1;

	sim.setInitialConditions();
	sim.evolve();

	// write the equilibrium profile: column density N_H, Av, T, and the 18
	// species number densities for every cell along the 1-D column.
	int status = 1;
	if (amrex::ParallelDescriptor::IOProcessor()) {
		// fextract's signature takes a non-const MultiFab& but does not mutate it.
		auto &mf = const_cast<amrex::MultiFab &>(sim.getNewMF_cc()[0]); // NOLINT
		auto [position, values] = fextract(mf, sim.Geom(0), 0, 0.5);	// NOLINT
		const int ngrid = static_cast<int>(position.size());

		std::filesystem::create_directories("pdr_State");
		std::ofstream out("pdr_State/profile.txt");
		out << "NH Av Tgas H Hp e H2 H2p He Hep C Cp CO HCOp O Si Sip CH OH H3p Op\n";

		bool finite = true;
		for (int i = 0; i < ngrid; ++i) {
			const Real rho = values.at(HydroSystem<PseudoPDRTest>::density_index)[i];
			const Real Eint = values.at(HydroSystem<PseudoPDRTest>::internalEnergy_index)[i];
			std::array<double, NumSpec> numdens = {};
			for (int nn = 0; nn < NumSpec; ++nn) {
				numdens[nn] = values.at(HydroSystem<PseudoPDRTest>::scalar0_index + nn)[i] / spmasses[nn];
			}
			const Real Tgas = compute_Tgas(rho, Eint, numdens);
			const Real NH = columnDensity(i, ngrid, sim.userData_.NH_min, sim.userData_.NH_total);
			const Real Av = cellAv(i, ngrid, sim.userData_);

			out << NH << " " << Av << " " << Tgas;
			for (int nn = 0; nn < NumSpec; ++nn) {
				out << " " << numdens[nn];
			}
			out << "\n";

			finite = finite && std::isfinite(Tgas) && (Tgas > 0.0);
			for (int nn = 0; nn < NumSpec; ++nn) {
				finite = finite && std::isfinite(numdens[nn]) && (numdens[nn] >= 0.0);
			}
		}
		out.close();

		amrex::Print() << "PseudoPDR: chi = " << sim.userData_.chi << ", n_H = " << sim.userData_.ninit << ", ngrid = " << ngrid << "\n";
		amrex::Print() << "Wrote pdr_State/profile.txt\n";
		if (finite) {
			status = 0;
		}
	}

	amrex::Print() << "Finished." << '\n';
	return status;
}
