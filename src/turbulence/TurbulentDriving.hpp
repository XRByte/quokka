#ifndef TURBULENTDRIVING_HPP
#define TURBULENTDRIVING_HPP

#include "AMReX.H"
#include "AMReX_AmrParticles.H"
#include "AMReX_Arena.H"
#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_BCRec.H"
#include "AMReX_BLassert.H"
#include "AMReX_Box.H"
#include "AMReX_Extension.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_FabArray.H"
#include "AMReX_FabFactory.H"
#include "AMReX_Geometry.H"
#include "AMReX_GpuContainers.H"
#include "AMReX_GpuControl.H"
#include "AMReX_GpuDevice.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_IntVect.H"
#include "AMReX_MultiFab.H"
#include "AMReX_MultiFabUtil.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_ParmParse.H"
#include "AMReX_PlotFileUtil.H"
#include "AMReX_Print.H"
#include "AMReX_REAL.H"
#include "AMReX_SPACE.H"
#include "AMReX_iMultiFab.H"

#include "TurbGenEx.h"
#include "fundamental_constants.H"
#include "hydro/hydro_system.hpp"
#include "math/FastMath.hpp"
#include "radiation/radiation_system.hpp"
#include <array>
#include <cmath>
#include <format>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace quokka::turbulence
{
/// Mass-weighted mean velocity and velocity dispersion of the computational domain.
struct VelocityMoments {
	amrex::GpuArray<amrex::Real, 3> mean;	    ///< mass-weighted mean velocity, per component
	amrex::GpuArray<amrex::Real, 3> dispersion; ///< mass-weighted velocity dispersion, per component
};

/// Compute the mass-weighted mean velocity and velocity dispersion over the domain.
template <typename problem_t> auto calculate_dispersion(amrex::MultiFab &state) -> VelocityMoments;

/// Domain totals of density and density-weighted forcing, used to correct the forcing's own mean bias.
struct ForcingTotals {
	amrex::Real total_density;					///< sum of rho over the domain
	amrex::GpuArray<amrex::Real, 3> total_density_weighted_forcing; ///< sum of rho*a_turb, per component
};

/// TurbGenEx with restart support for the auto-adjusted driving amplitude.
///
/// TurbGen::init_driving(params) always initialises at time = 0, which resets the auto-adjusted
/// ampl_factor to the user value on restart. The OU phases need no restoring: they are replayed
/// deterministically from the random seed up to the restart time on the first pattern update.
///
/// TurbGen's own read_ampl_factor_from_evol_file() is not used: it can pick the entry *before* the
/// last pattern update, and the first check_for_update() after a restart would then apply an extra
/// amplitude adjustment that the uninterrupted run never made. Here the restored value is the one
/// in effect at the restart time, and the catch-up update skips the adjustment unless a pattern
/// update is genuinely due at the restart time.
class RestartableTurbGen final : public TurbGenEx
{
      private:
	int last_update_step_ = -1; // OU step of the last pattern update before the restart

      public:
	/// Restore ampl_factor from the TurbGen evolution file (TurbGen.dat) for restart time `time`.
	/// Must be called after init_driving() and before the first check_for_update().
	/// Returns false if amplitude auto-adjustment is on but no valid entry can be read.
	auto restore_ampl_factor(const double time) -> bool
	{
		if (ampl_auto_adjust != 1) {
			return true; // fixed amplitude: the user value is already correct
		}
		std::ifstream infile(evolfile);
		if (!infile.is_open()) {
			return false;
		}
		std::vector<std::string> lines;
		for (std::string line; std::getline(infile, line);) {
			lines.push_back(line);
		}
		// entries are appended in run order, so after earlier restarts the file can hold entries of an
		// abandoned future; the last entry written at or before the restart time is the value in effect.
		// An entry at exactly the restart time belongs to the step that ended there: the driving is applied
		// in both Strang half-steps, at t and at t + dt, so an update due at the checkpoint time has already
		// been made by the second half of the last step before the checkpoint.
		const double t_cut = time * (1.0 + 1.0e-12);
		for (auto it = lines.rbegin(); it != lines.rend(); ++it) {
			if (it->empty() || it->find('#') != std::string::npos) {
				continue; // header line
			}
			std::istringstream buffer(*it);
			double t_file = 0.0;
			double t_in_turb = 0.0;
			std::array<double, 3> ampl_in_file = {0.0, 0.0, 0.0};
			if (!(buffer >> t_file >> t_in_turb >> ampl_in_file[0] >> ampl_in_file[1] >> ampl_in_file[2])) {
				continue;
			}
			if (t_file <= t_cut) {
				// the evolution file stores ampl_factor^(1/1.5); internally TurbGen uses ampl_factor^1.5
				for (int d = 0; d < ncmp; d++) {
					ampl_factor[d] = std::pow(ampl_in_file[d], 1.5);
				}
				last_update_step_ = static_cast<int>(std::floor(t_file / dt));
				amrex::Print() << std::format(
				    "TurbGen: restored ampl_factor = {:e} {:e} {:e} from '{}' (entry at t = {:e}) for restart time {:e}\n", ampl_in_file[0],
				    ampl_in_file[1], ampl_in_file[2], evolfile, t_file, time);
				return true;
			}
		}
		return false;
	}

	/// True if the uninterrupted run would have made a pattern update (with amplitude adjustment) at `time`.
	[[nodiscard]] auto update_due_since_restore(const double time) const -> bool { return std::floor(time / dt) > last_update_step_; }
};

template <typename problem_t> class turbulentDriving
{
      private:
	RestartableTurbGen tg;
	bool restore_on_first_update = false; // set when restarting from a checkpoint
	bool updated_forcing_pattern = false;
	amrex::GpuArray<amrex::Real, 3> velocity_dispersion = {-1.0, -1.0, -1.0};
	bool remove_mean_flow = false;
	amrex::GpuArray<amrex::Real, 3> accumulated_forcing_bias = {0.0, 0.0, 0.0};

	// the forcing pattern is exactly zero-mean by construction, but the momentum source applied
	// is density-weighted, so a net mean flow can still build up if the forcing correlates with
	// density over time; this stays small for weakly compressible turbulence, but can grow to a
	// large fraction of the dispersion for strongly compressible turbulence
	static constexpr amrex::Real mean_flow_to_dispersion_threshold = 0.1;

	void update(const amrex::Real &time, amrex::MultiFab &state)
	{
		bool catch_up_without_adjust = false;
		if (restore_on_first_update) {
			restore_on_first_update = false;
			if (!tg.restore_ampl_factor(time)) {
				amrex::Abort(
				    "TurbulentDriving: restarting with turbulence.ampl_auto_adjust = 1, but the driving amplitude could not be restored "
				    "from 'TurbGen.dat'. Copy TurbGen.dat from the original run into the run directory.");
			}
			// replay the OU sequence to the restart time, adjusting the amplitude only if an update is due now
			catch_up_without_adjust = !tg.update_due_since_restore(time);
		}

		updated_forcing_pattern = tg.is_update_available(time);

		if (updated_forcing_pattern) {
			const VelocityMoments velocity_moments = quokka::turbulence::calculate_dispersion<problem_t>(state);
			velocity_dispersion = velocity_moments.dispersion;
			if (catch_up_without_adjust) {
				tg.check_for_update(time);
			} else {
				tg.check_for_update(time, velocity_dispersion.data());
			}

			const amrex::Real forcing_bias_magnitude =
			    std::sqrt(accumulated_forcing_bias[0] * accumulated_forcing_bias[0] + accumulated_forcing_bias[1] * accumulated_forcing_bias[1] +
				      accumulated_forcing_bias[2] * accumulated_forcing_bias[2]);
			const amrex::Real dispersion_magnitude =
			    std::sqrt(velocity_dispersion[0] * velocity_dispersion[0] + velocity_dispersion[1] * velocity_dispersion[1] +
				      velocity_dispersion[2] * velocity_dispersion[2]);

			if (forcing_bias_magnitude > mean_flow_to_dispersion_threshold * dispersion_magnitude) {
				const std::string msg =
				    std::format("TurbulentDriving: accumulated forcing bias ({:.3e}) exceeds {:.0f}% of the velocity dispersion "
						"({:.3e}) at time {:.3e}; the density-weighted forcing has injected a net bulk flow.",
						forcing_bias_magnitude, mean_flow_to_dispersion_threshold * 100.0, dispersion_magnitude, time);
				if (remove_mean_flow) {
					amrex::Abort(("[FATAL] " + msg).c_str());
				} else {
					amrex::Print()
					    << "[WARNING] " << msg << " Enable turbulence.remove_mean_flow for this to be automatically corrected.\n";
				}
			}
		}
	}

      public:
	turbulentDriving() = default;
	explicit turbulentDriving(const std::map<std::string, std::string> &turb_params, bool remove_mean_flow_in) : remove_mean_flow(remove_mean_flow_in)
	{
		tg.init_driving(turb_params);
		// the restart time is not known yet (the checkpoint is read later), so restore on the first update
		restore_on_first_update = amrex::ParmParse("").contains("restartfile");
	}

	auto applyDriving(amrex::MultiFab &state, const amrex::Real time, const amrex::Real dt_in,
			  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &cellSizes) -> bool
	{
		update(time, state);
		const amrex::Real dt = dt_in;

		amrex::MultiFab forcing(state.boxArray(), state.DistributionMap(), AMREX_SPACEDIM, 0);

		// domain totals needed to remove the net mean flow the forcing would otherwise inject
		amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum> reduce_op;
		amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real> reduce_data(reduce_op);

		for (amrex::MFIter mf(state); mf.isValid(); ++mf) {
			const amrex::Box &bx = mf.validbox();
			auto const &data = state.array(mf);

			amrex::FArrayBox &axFab = forcing[mf];
			tg.get_turb_vector_unigrid(axFab, cellSizes);
			auto const &ax = forcing.const_array(mf);

			reduce_op.eval(bx, reduce_data,
				       [=] AMREX_GPU_DEVICE(int i, int j, int k) -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real, amrex::Real> {
					       const amrex::Real rho = data(i, j, k, HydroSystem<problem_t>::density_index);

					       return {rho, rho * ax(i, j, k, 0), (AMREX_SPACEDIM > 1) ? rho * ax(i, j, k, 1) : 0.0,
						       (AMREX_SPACEDIM > 2) ? rho * ax(i, j, k, 2) : 0.0};
				       });
		}

		const auto [sum_rho, sum_rax, sum_ray, sum_raz] = reduce_data.value();

		amrex::GpuArray<amrex::Real, 4> volume_summed_quantities = {sum_rho, sum_rax, sum_ray, sum_raz};
		amrex::ParallelDescriptor::ReduceRealSum(volume_summed_quantities.data(), 4);

		const ForcingTotals forcing_totals{
		    .total_density = volume_summed_quantities[0],
		    .total_density_weighted_forcing = {volume_summed_quantities[1], volume_summed_quantities[2], volume_summed_quantities[3]}};

		const amrex::GpuArray<amrex::Real, 3> forcing_bias_increment = {
		    dt * forcing_totals.total_density_weighted_forcing[0] / forcing_totals.total_density,
		    dt * forcing_totals.total_density_weighted_forcing[1] / forcing_totals.total_density,
		    dt * forcing_totals.total_density_weighted_forcing[2] / forcing_totals.total_density,
		};

		// mean velocity this step's forcing would inject; subtracting this keeps the forcing's own
		// contribution to the domain-mean velocity at zero every step, without touching any
		// pre-existing bulk motion from other sources (initial conditions, gravity, feedback)
		amrex::GpuArray<amrex::Real, 3> mean_correction = {0.0, 0.0, 0.0};
		if (remove_mean_flow) {
			mean_correction = forcing_bias_increment;
		}

		// residual bias left uncorrected this step; zero by construction when remove_mean_flow is
		// enabled, so a nonzero residual there signals the correction itself has broken
		accumulated_forcing_bias[0] += forcing_bias_increment[0] - mean_correction[0];
		accumulated_forcing_bias[1] += forcing_bias_increment[1] - mean_correction[1];
		accumulated_forcing_bias[2] += forcing_bias_increment[2] - mean_correction[2];

		for (amrex::MFIter mf(state); mf.isValid(); ++mf) {
			const amrex::Box &bx = mf.validbox();
			auto const &data = state.array(mf);
			auto const &ax = forcing.const_array(mf);

			amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
				const amrex::Real rho = data(i, j, k, HydroSystem<problem_t>::density_index);

				amrex::Real dE = 0;

				for (int m = 0; m < AMREX_SPACEDIM; m++) {
					const amrex::Real vel = data(i, j, k, HydroSystem<problem_t>::x1Momentum_index + m) / rho;
					const amrex::Real dMom = (ax(i, j, k, m) * dt - mean_correction[m]) * rho;

					data(i, j, k, HydroSystem<problem_t>::x1Momentum_index + m) += dMom;
					dE += vel * dMom + dMom * dMom / (2 * rho);
				}

				data(i, j, k, HydroSystem<problem_t>::energy_index) += dE;
			});
		}

		amrex::Gpu::streamSynchronize();
		return updated_forcing_pattern;
	}
};

// Function to calculate the mass weighted mean velocity and velocity dispersion in the computational domain
template <typename problem_t> auto calculate_dispersion(amrex::MultiFab &state) -> VelocityMoments
{
	amrex::ReduceOps<amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum, amrex::ReduceOpSum,
			 amrex::ReduceOpSum>
	    reduce_op;
	amrex::ReduceData<amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real> reduce_data(reduce_op);

	for (amrex::MFIter mfi(state); mfi.isValid(); ++mfi) {
		const amrex::Box &bx = mfi.validbox();
		auto const &data = state.array(mfi);

		reduce_op.eval(bx, reduce_data,
			       [=] AMREX_GPU_DEVICE(int i, int j, int k)
				   -> amrex::GpuTuple<amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real, amrex::Real> {
				       const amrex::Real rho = data(i, j, k, HydroSystem<problem_t>::density_index);
				       const amrex::Real px = data(i, j, k, HydroSystem<problem_t>::x1Momentum_index);
				       const amrex::Real py = data(i, j, k, HydroSystem<problem_t>::x2Momentum_index);
				       const amrex::Real pz = data(i, j, k, HydroSystem<problem_t>::x3Momentum_index);

				       const amrex::Real vx = px / rho;
				       const amrex::Real vy = py / rho;
				       const amrex::Real vz = pz / rho;

				       return {rho, px, py, pz, px * vx, py * vy, pz * vz};
			       });
	}

	auto [sum_rho, sum_px, sum_py, sum_pz, sum_pvx, sum_pvy, sum_pvz] = reduce_data.value();

	amrex::GpuArray<amrex::Real, 7> reduce_vec = {sum_rho, sum_px, sum_py, sum_pz, sum_pvx, sum_pvy, sum_pvz};
	amrex::ParallelDescriptor::ReduceRealSum(reduce_vec.data(), 7);

	const amrex::Real total_rho = reduce_vec[0];
	const amrex::Real total_px = reduce_vec[1];
	const amrex::Real total_py = reduce_vec[2];
	const amrex::Real total_pz = reduce_vec[3];
	const amrex::Real total_pvx = reduce_vec[4];
	const amrex::Real total_pvy = reduce_vec[5];
	const amrex::Real total_pvz = reduce_vec[6];

	const amrex::Real v_avg_x = total_px / total_rho;
	const amrex::Real v_avg_y = total_py / total_rho;
	const amrex::Real v_avg_z = total_pz / total_rho;

	// Compute dispersion using the identity: Var(X) = E[X^2] - (E[X])^2
	const amrex::Real dispx = std::sqrt(std::max(0.0, (total_pvx / total_rho) - (v_avg_x * v_avg_x)));
	const amrex::Real dispy = std::sqrt(std::max(0.0, (total_pvy / total_rho) - (v_avg_y * v_avg_y)));
	const amrex::Real dispz = std::sqrt(std::max(0.0, (total_pvz / total_rho) - (v_avg_z * v_avg_z)));

	return VelocityMoments{.mean = {v_avg_x, v_avg_y, v_avg_z}, .dispersion = {dispx, dispy, dispz}};
}
} // namespace quokka::turbulence

#endif // TURBULENTDRIVING_HPP
