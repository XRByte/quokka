/// \file testTurbulence.cpp
/// \brief Driven turbulence in a periodic box with the radiation-free Gong, Ostriker & Wolfire
///        (2017) 18-species chemistry + thermal network (GOW_nr).
///
///        Initial conditions follow the GOW one-zone test (initial_numdens): uniform gas at
///        H-nucleus density n_H and temperature T0, at rest. The FUV field is a constant Draine
///        field chi attenuated per cell by a LOCAL column-density approximation: every column is
///        n_X * L_shield with L_shield the Jeans length evaluated at a temperature capped at
///        T_cap = 40 K (Safranek-Shrader et al. 2017, MNRAS 465, 885, Eq. 27, model L1a). The
///        per-cell columns and Av are passed to the network RHS through burn_t::aux[0..4]
///        (PDR_PARALLEL_NCOL_AUX), so the burn runs per cell in parallel on CPU or GPU.
///
///        The chemistry-free baseline of this setup is src/problems/TurbulenceNoChem.

#include "AMReX_Print.H"
#include "QuokkaSimulation.hpp"
#include "SimulationData.hpp"
#include "chemistry/Chemistry.hpp"
#include "hydro/hydro_system.hpp"
#include "turbulence/TurbulentDriving.hpp"
#include "util/BC.hpp"
#ifdef HAVE_PYTHON
#include "util/matplotlibcpp.h"
#endif

#include "AMReX_FabArray.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_MultiFab.H"
#include "AMReX_ParmParse.H"
#include "AMReX_REAL.H"

#include "actual_eos_data.H"
#include "burn_type.H"
#include "eos.H"
#include "extern_parameters.H"
#include "fundamental_constants.H"
#include "network.H"

#include <gow_co_shield.H>
#include <gow_tables.hpp>

#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

using amrex::Real;

static_assert(NumAux >= 5, "Turbulence needs NAUX_NET >= 5: aux[0..3] = ncol_{H2,CO,C,H}, aux[4] = Av");

struct TurbulentBox {}; // dummy type to allow compile-type polymorphism via template specialization

template <> struct Physics_Traits<TurbulentBox> : DefaultPhysicsTraits {
	static constexpr bool is_hydro_enabled = true;
	static constexpr int numMassScalars = NumSpec;		     // 18 chemical species
	static constexpr int numPassiveScalars = numMassScalars + 0; // only mass scalars
};

template <> struct HydroSystem_Traits<TurbulentBox> {
	static constexpr bool reconstruct_eint = false;
};

template <> struct SimulationData<TurbulentBox> {
	Real small_temp;
	Real small_dens;
	Real nH;	      // initial H-nucleus number density [cm^-3]
	Real T0;	      // initial gas temperature [K]
	Real rho0;	      // initial mass density [g cm^-3]
	Real T_cap;	      // temperature cap in the Jeans shielding length [K]
	Real NH_over_Av;      // N_H per unit Av [cm^-2]
	Real chem_start_time; // chemistry (and heating/cooling) is off before this time [s]
	bool chem_started = false;
	bool chem_initialized = false; // preCalculateInitialConditions() has run (it is also needed on restart)
	Real metallicity;
	std::array<Real, NumSpec> rel_abundance; // initial seed abundances per H nucleus

	std::vector<double> t_vec_;
	std::vector<double> Disp3d_vec_;
	std::ofstream output_file_;
};

// species indices (xn[] order of the GOW_nr network)
namespace gow_idx
{
constexpr int H = 0, Hp = 1, e = 2, H2 = 3, H2p = 4, He = 5, Hep = 6, Cn = 7, Cp = 8, CO = 9, HCOp = 10, O = 11, Si = 12, Sip = 13, CH = 14, OH = 15, H3p = 16,
	      Op = 17;
} // namespace gow_idx

// H-nucleus number density from the species number densities [cm^-3]
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto hNucleiDensity(const Real *xn) -> Real
{
	using namespace gow_idx;
	return xn[H] + xn[Hp] + 2.0 * (xn[H2] + xn[H2p]) + 3.0 * xn[H3p] + xn[HCOp] + xn[CH] + xn[OH];
}

// Initial number densities, as in the GOW one-zone test (Gong+17 singlezone_gen.cpp): nH is the total
// H-nucleus density, the C, O, Si and He totals per H nucleus are fixed (metals scaled by metallicity),
// and the species carrying each element total (H, C+, O, Si, He) take whatever the seeds leave. The
// electrons start equal to the cation density so the initial state is charge neutral.
auto initial_numdens(const SimulationData<TurbulentBox> &ud) -> std::array<Real, NumSpec>
{
	constexpr Real xC_tot = 1.6e-4;
	constexpr Real xO_tot = 3.2e-4;
	constexpr Real xSi_tot = 1.7e-6;
	constexpr Real xHe_tot = 0.1;

	std::array<Real, NumSpec> x = ud.rel_abundance;
	const Real met = ud.metallicity;

	using namespace gow_idx;
	x[H] = 1.0 - (x[Hp] + 2.0 * x[H2] + 2.0 * x[H2p] + x[HCOp] + x[CH] + x[OH] + 3.0 * x[H3p]);
	x[He] = xHe_tot - x[Hep];
	x[Cp] = met * xC_tot - (x[Cn] + x[CO] + x[HCOp] + x[CH]);
	x[O] = met * xO_tot - (x[CO] + x[HCOp] + x[OH] + x[Op]);
	x[Si] = met * xSi_tot - x[Sip];
	x[e] = x[Hp] + x[H2p] + x[Hep] + x[Cp] + x[HCOp] + x[Sip] + x[H3p] + x[Op];

	std::array<Real, NumSpec> numdens{};
	for (int n = 0; n < NumSpec; ++n) {
		numdens[n] = x[n] * ud.nH;
	}
	return numdens;
}

template <> void QuokkaSimulation<TurbulentBox>::preCalculateInitialConditions()
{
	// AMReX calls this once per level built from scratch, but never on restart: problem_main therefore calls it
	// explicitly as well, and this guard makes the repeated calls no-ops.
	if (userData_.chem_initialized) {
		return;
	}
	userData_.chem_initialized = true;

	// initialize microphysics routines
	init_extern_parameters();

	amrex::ParmParse const pp("turbchem");
	userData_.small_temp = 1.0e1;
	pp.query("small_temp", userData_.small_temp);
	userData_.small_dens = 1.0e-60;
	pp.query("small_dens", userData_.small_dens);
	userData_.nH = 100.0;
	pp.query("nH", userData_.nH);
	userData_.T0 = 100.0;
	pp.query("T0", userData_.T0);
	userData_.T_cap = 40.0; // Safranek-Shrader+17 model L1a
	pp.query("T_cap", userData_.T_cap);
	userData_.NH_over_Av = 1.87e21; // Zd = 1
	pp.query("NH_over_Av", userData_.NH_over_Av);
	// Micic+12 protocol: drive pure hydro turbulence to steady state first (e.g. 2 T, T = L / 2 v_rms),
	// then switch the chemistry on. 0 = chemistry from the start.
	userData_.chem_start_time = 0.0;
	pp.query("chem_start_time", userData_.chem_start_time);

	userData_.metallicity = network_rp::metallicity;

	// GOW one-zone seed abundances per H nucleus (xn[] order). Entries marked "total" are set from
	// the element totals / charge neutrality in initial_numdens.
	userData_.rel_abundance = {
	    0.0,	  // H   (total)
	    1.0e-4,	  // H+
	    0.0,	  // e-  (cation density)
	    1.0e-10,	  // H2
	    1.0e-40,	  // H2+
	    0.0,	  // He  (total)
	    1.450654e-08, // He+
	    1.0e-40,	  // C
	    0.0,	  // C+  (total)
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

	// the local columns are already the full shielding columns (N = n L): no slab path-length factor
	gow_co_shield::iso_factor_CO = 1.0;

	eos_init(userData_.small_temp, userData_.small_dens);
	network_init();

	const std::array<Real, NumSpec> numdens = initial_numdens(userData_);
	Real rho0 = 0.0;
	for (int n = 0; n < NumSpec; ++n) {
		rho0 += numdens[n] * spmasses[n];
	}
	userData_.rho0 = rho0;

	if (amrex::ParallelDescriptor::IOProcessor()) {
		// on restart, append to the existing time series instead of truncating it
		const bool restarting = amrex::ParmParse("").contains("restartfile");
		userData_.output_file_.open("turbulence_chem.txt", restarting ? std::ios::app : std::ios::trunc);
		userData_.output_file_ << std::setprecision(10);
		userData_.output_file_ << "# time[s] vdisp3d[cm/s] T_massweighted[K] f_H2(=2nH2/nH) f_CO(=nCO/nC_tot) f_Cp(=nCp/nC_tot)\n";
	}
}

template <> void QuokkaSimulation<TurbulentBox>::setInitialConditionsOnGrid(quokka::grid const &grid_elem)
{
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state_cc = grid_elem.array_;

	burn_t state;
	const std::array<Real, NumSpec> numdens = initial_numdens(userData_);

	state.T = userData_.T0;
	Real rhotot = 0.0;
	for (int n = 0; n < NumSpec; ++n) {
		state.xn[n] = numdens[n];
		rhotot += state.xn[n] * spmasses[n];
	}
	state.rho = rhotot;

	// EOS gives the specific internal energy at (rho, T0, composition)
	eos(eos_input_rt, state);

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
		const Real rho = state.rho;
		const Real Eint = rho * state.e; // gas at rest: E = Eint

		state_cc(i, j, k, HydroSystem<TurbulentBox>::density_index) = rho;
		state_cc(i, j, k, HydroSystem<TurbulentBox>::x1Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<TurbulentBox>::x2Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<TurbulentBox>::x3Momentum_index) = 0.0;
		state_cc(i, j, k, HydroSystem<TurbulentBox>::energy_index) = Eint;
		state_cc(i, j, k, HydroSystem<TurbulentBox>::internalEnergy_index) = Eint;

		for (int nn = 0; nn < NumSpec; ++nn) {
			// scalar indices carry partial densities (not number densities)
			state_cc(i, j, k, HydroSystem<TurbulentBox>::scalar0_index + nn) = state.xn[nn] * spmasses[nn];
		}
	});
}

template <> void QuokkaSimulation<TurbulentBox>::refineGrid(int lev, amrex::TagBoxArray &tags, amrex::Real /*time*/, int /*ngrow*/)
{
	// tag cells for refinement

	const amrex::Real eta_threshold = 0.2;		  // gradient refinement threshold
	const amrex::Real rho_min = 0.1 * userData_.rho0; // minimum density for refinement [g cm^-3]

	const auto state = state_new_cc_[lev].const_arrays();
	const auto tag = tags.arrays();

	amrex::ParallelFor(state_new_cc_[lev], [=] AMREX_GPU_DEVICE(int bx, int i, int j, int k) noexcept {
		const int n = HydroSystem<TurbulentBox>::density_index;
		amrex::Real const rho = state[bx](i, j, k, n);
		amrex::Real const rho_xplus = state[bx](i + 1, j, k, n);
		amrex::Real const rho_xminus = state[bx](i - 1, j, k, n);
		amrex::Real const rho_yplus = state[bx](i, j + 1, k, n);
		amrex::Real const rho_yminus = state[bx](i, j - 1, k, n);
		amrex::Real const rho_zplus = state[bx](i, j, k + 1, n);
		amrex::Real const rho_zminus = state[bx](i, j, k - 1, n);

		amrex::Real const del_x = 0.5 * (rho_xplus - rho_xminus);
		amrex::Real const del_y = 0.5 * (rho_yplus - rho_yminus);
		amrex::Real const del_z = 0.5 * (rho_zplus - rho_zminus);

		amrex::Real const gradient_indicator = std::sqrt(del_x * del_x + del_y * del_y + del_z * del_z) / rho;

		if ((gradient_indicator > eta_threshold) && (rho > rho_min)) {
			tag[bx](i, j, k) = amrex::TagBox::SET;
		}
	});
	amrex::Gpu::streamSynchronize();
}

// Per-cell chemistry + thermal burn with local Jeans-length shielding. Called on every level as a
// Strang-split source (dt is half the level timestep). Mirrors quokka::chemistry::computeChemistry,
// except that (i) each cell's shielding columns and Av go into burn_t::aux, and (ii) the evolved
// number densities are written back directly (no mass-fraction renormalization), as in PseudoPDR.
template <> void QuokkaSimulation<TurbulentBox>::addStrangSplitSources(amrex::MultiFab &state, int /*lev*/, amrex::Real time, amrex::Real dt)
{
	// hydro-only spin-up phase: species and internal energy are left untouched (no chemistry, heating or cooling)
	if (time < userData_.chem_start_time) {
		return;
	}
	if (!userData_.chem_started) {
		userData_.chem_started = true;
		amrex::Print() << "\n>> Chemistry switched on at t = " << time << " s\n\n";
	}

	const BL_PROFILE("Turbulence::chemistryBurn()");

	const Real T_cap = userData_.T_cap;
	const Real inv_NH_over_Av = 1.0 / userData_.NH_over_Av;
	const Real min_density = min_density_allowed;

	amrex::Gpu::Buffer<int> d_num_failed({0});
	int *p_num_failed = d_num_failed.data();

	for (amrex::MFIter iter(state); iter.isValid(); ++iter) {
		const amrex::Box &indexRange = iter.validbox();
		auto const &s = state.array(iter);

		amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			const Real rho = s(i, j, k, HydroSystem<TurbulentBox>::density_index);
			if (rho < min_density) {
				return;
			}
			const Real xmom = s(i, j, k, HydroSystem<TurbulentBox>::x1Momentum_index);
			const Real ymom = s(i, j, k, HydroSystem<TurbulentBox>::x2Momentum_index);
			const Real zmom = s(i, j, k, HydroSystem<TurbulentBox>::x3Momentum_index);
			const Real Ener = s(i, j, k, HydroSystem<TurbulentBox>::energy_index);
			const Real Eint = quokka::EOS<TurbulentBox>::ComputeEintFromEgas(rho, xmom, ymom, zmom, Ener, 0.0);

			burn_t chemstate;
			chemstate.success = true;
			Real ntot = 0.0;
			for (int nn = 0; nn < NumSpec; ++nn) {
				// scalars are partial densities -> number densities
				chemstate.xn[nn] = s(i, j, k, HydroSystem<TurbulentBox>::scalar0_index + nn) / spmasses[nn];
				ntot += chemstate.xn[nn];
			}
			chemstate.rho = rho;
			chemstate.e = Eint / rho;
			eos(eos_input_re, chemstate); // sets chemstate.T

			// local Jeans shielding length (Safranek-Shrader+17 Eq. 27) with T capped at T_cap;
			// isothermal sound speed of the local mixture: c_s^2 = k_B T n_tot / rho
			const Real T_shield = amrex::min(chemstate.T, T_cap);
			const Real cs2 = C::k_B * T_shield * ntot / rho;
			const Real L_shield = std::sqrt(M_PI * cs2 / (C::Gconst * rho));

			// shielding columns N_X = n_X * L_shield and dust extinction Av = N_H / (N_H per Av)
			const Real NH = hNucleiDensity(chemstate.xn) * L_shield;
			chemstate.aux[0] = chemstate.xn[gow_idx::H2] * L_shield; // ncol_H2
			chemstate.aux[1] = chemstate.xn[gow_idx::CO] * L_shield; // ncol_CO
			chemstate.aux[2] = chemstate.xn[gow_idx::Cn] * L_shield; // ncol_C (neutral atomic C)
			chemstate.aux[3] = NH;					 // ncol_H (CO-cooling depth L = ncol_H / nH)
			chemstate.aux[4] = NH * inv_NH_over_Av;			 // Av

			quokka::chemistry::chemburner(chemstate, dt);

			if (std::isnan(chemstate.xn[0]) || std::isnan(chemstate.rho)) {
				amrex::Abort("Burner returned NAN");
			}
			if (!chemstate.success) {
				amrex::Gpu::Atomic::Add(p_num_failed, 1);
			}

			// positivity only (number-density mode, renormalize_abundances = 0)
			for (int nn = 0; nn < NumSpec; ++nn) {
				chemstate.xn[nn] = amrex::max(chemstate.xn[nn], small_x);
			}

			// keep the integrated e; T follows it
			eos(eos_input_re, chemstate);

			const Real dEint = (chemstate.e * chemstate.rho) - Eint;
			s(i, j, k, HydroSystem<TurbulentBox>::internalEnergy_index) += dEint;
			s(i, j, k, HydroSystem<TurbulentBox>::energy_index) += dEint;
			for (int nn = 0; nn < NumSpec; ++nn) {
				s(i, j, k, HydroSystem<TurbulentBox>::scalar0_index + nn) = chemstate.xn[nn] * spmasses[nn];
			}
		});

#if defined(AMREX_USE_HIP)
		amrex::Gpu::streamSynchronize(); // otherwise HIP may fail to allocate the necessary resources
#endif
	}

	int num_failed = *(d_num_failed.copyToHost());
	amrex::ParallelDescriptor::ReduceIntSum(num_failed);
	if (num_failed > 0) {
		amrex::Print() << "\t>> WARNING: chemistry burn failed in " << num_failed << " cells\n";
	}
}

template <> void QuokkaSimulation<TurbulentBox>::computeAfterTimestep()
{
	const auto disp = quokka::turbulence::calculate_dispersion<TurbulentBox>(state_new_cc_[0]).dispersion;
	const amrex::Real disp3d = std::sqrt(disp[0] * disp[0] + disp[1] * disp[1] + disp[2] * disp[2]);

	// domain sums on the coarse level: rho, rho*T, n_H, 2 n_H2, n_CO, n_C+, n_C,tot
	amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
			 amrex::ReduceOpSum>
	    reduce_op;
	amrex::ReduceData<Real, Real, Real, Real, Real, Real, Real> reduce_data(reduce_op);

	for (amrex::MFIter mfi(state_new_cc_[0]); mfi.isValid(); ++mfi) {
		const amrex::Box &bx = mfi.validbox();
		auto const &s = state_new_cc_[0].const_array(mfi);

		reduce_op.eval(bx, reduce_data, [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<Real, Real, Real, Real, Real, Real, Real> {
			using namespace gow_idx;
			burn_t st;
			const Real rho = s(i, j, k, HydroSystem<TurbulentBox>::density_index);
			for (int nn = 0; nn < NumSpec; ++nn) {
				st.xn[nn] = s(i, j, k, HydroSystem<TurbulentBox>::scalar0_index + nn) / spmasses[nn];
			}
			st.rho = rho;
			st.e = s(i, j, k, HydroSystem<TurbulentBox>::internalEnergy_index) / rho;
			eos(eos_input_re, st);

			const Real nC_tot = st.xn[Cn] + st.xn[Cp] + st.xn[CO] + st.xn[HCOp] + st.xn[CH];
			return {rho, rho * st.T, hNucleiDensity(st.xn), 2.0 * st.xn[H2], st.xn[CO], st.xn[Cp], nC_tot};
		});
	}

	auto [s_rho, s_rhoT, s_nH, s_2nH2, s_nCO, s_nCp, s_nC] = reduce_data.value();
	amrex::GpuArray<Real, 7> sums = {s_rho, s_rhoT, s_nH, s_2nH2, s_nCO, s_nCp, s_nC};
	amrex::ParallelDescriptor::ReduceRealSum(sums.data(), 7);

	if (amrex::ParallelDescriptor::IOProcessor()) {
		userData_.t_vec_.push_back(tNew_[0]);
		userData_.Disp3d_vec_.push_back(disp3d);
		userData_.output_file_ << tNew_[0] << " " << disp3d << " " << sums[1] / sums[0] << " " << sums[3] / sums[2] << " " << sums[4] / sums[6] << " "
				       << sums[5] / sums[6] << "\n";
		userData_.output_file_.flush();
	}
}

auto problem_main() -> int
{
	auto BCs_cc = quokka::BC<TurbulentBox>(quokka::BCType::int_dir,	 // x: periodic
					       quokka::BCType::int_dir,	 // y: periodic
					       quokka::BCType::int_dir); // z: periodic

	QuokkaSimulation<TurbulentBox> sim(BCs_cc);

	// Load the GOW cooling tables now that amrex::Initialize() (and MPI) are up; must precede any RHS evaluation.
	gow_tables::registerGowTables();

	// microphysics init + turbchem parameters; must run before a restart too (AMReX only calls it when building levels from scratch)
	sim.preCalculateInitialConditions();

	sim.setInitialConditions();

	// Main time loop
	sim.evolve();

	// Check solution validity
	int status = 0;

	if (amrex::ParallelDescriptor::IOProcessor()) {
		if (!sim.userData_.Disp3d_vec_.empty()) {
			const auto disp_last = sim.userData_.Disp3d_vec_.back();
			const double target_vdisp = std::stod(sim.turbParams_["target_vdisp"]);
			const double rel_error = std::abs(target_vdisp - disp_last) / target_vdisp;
			const double err_tol = 0.075;

			amrex::Print() << "\n" << "Target velocity dispersion: " << target_vdisp << "\n";
			amrex::Print() << "Last calculated velocity dispersion: " << disp_last << "\n";
			amrex::Print() << "Relative error: " << rel_error << "\n\n";

			if ((rel_error > err_tol) || std::isnan(rel_error)) {
				status = 1;
			}

#if HAVE_PYTHON
			// Plot dispersion
			const std::vector<double> &time = sim.userData_.t_vec_;
			const std::vector<double> &disp3d = sim.userData_.Disp3d_vec_;

			matplotlibcpp::clf();
			std::map<std::string, std::string> Vdisp_args;
			Vdisp_args["label"] = "Velocity dispersion vs time";
			Vdisp_args["linestyle"] = "-";
			Vdisp_args["color"] = "C1";
			matplotlibcpp::plot(time, disp3d, Vdisp_args);
			matplotlibcpp::xlabel("t (s)");
			matplotlibcpp::ylabel("vdisp (cm/s)");
			matplotlibcpp::legend();
			matplotlibcpp::tight_layout();
			matplotlibcpp::save("./Turbulence_vdisp.pdf");
#endif

		} else {
			amrex::Print() << "Error: Dispersion vector is empty!\n";
			status = 1;
		}
	}

	return status;
}
