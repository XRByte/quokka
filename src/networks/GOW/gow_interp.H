#ifndef GOW_INTERP_H
#define GOW_INTERP_H

// Free-function interface to the GOW CO-cooling and dust photo-electric tables,
// as called by the network RHS/Jacobian in actual_rhs.H:
//   l0_co_interp1d(T)                 -> CO L0 cooling coeff
//   llte_co_interp2d(T, NCOeff1e5)    -> CO LTE cooling coeff
//   nhalf_co_interp2d(T, NCOeff1e5)   -> CO n_1/2
//   alpha_co_interp2d(T, NCOeff1e5)   -> CO alpha exponent
//   psigd_coll_interp2d(nHtot, T)     -> dust psi_gd
// plus *_partial_0 / *_partial_1 partial derivatives (analytic Jacobian only;
// the runtime uses the numerical Jacobian).
//
// All lookups delegate to quokka::DataTable views held in the managed registry
// (gow_tables.hpp, loaded at init in actual_network_data.cpp).  The tables use
// log-spaced axes storing physical values, so queries pass physical (T, N, nH)
// directly and DataTable applies the log/linear transforms internally.  Out-of-
// range queries are clamped to the table edges (DataTable clamp policy), which
// also caps CO temperatures at the 2000 K table edge.
//
// The second argument of the CO 2-D functions is NCOeff*1e5 (= n_CO/gradeff
// times 1e5), a physical N_CO,eff; DataTable's log axis handles the log10-style
// transform and the low-N floor via clamping.

#include <array>

#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <gow_tables.hpp>

// CO self-shielding factor fshield_co_interp2d(N_CO, N_H2) (Visser+2009 Table 5)
#include <gow_co_shield.H>

// ---------------------------------------------------------------------------
// Forward interpolation functions
// ---------------------------------------------------------------------------

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto l0_co_interp1d(const amrex::Real Tin) -> amrex::Real
{
	const std::array<amrex::Real, 1> point = {Tin};
	return gow_tables::getGowTables().l0.interpolate_single(point, 0);
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto llte_co_interp2d(const amrex::Real Tin, const amrex::Real NCOeff1e5) -> amrex::Real
{
	const std::array<amrex::Real, 2> point = {Tin, NCOeff1e5};
	return gow_tables::getGowTables().co.interpolate_single(point, gow_tables::CO_LLTE_IDX);
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto nhalf_co_interp2d(const amrex::Real Tin, const amrex::Real NCOeff1e5) -> amrex::Real
{
	const std::array<amrex::Real, 2> point = {Tin, NCOeff1e5};
	return gow_tables::getGowTables().co.interpolate_single(point, gow_tables::CO_NHALF_IDX);
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto alpha_co_interp2d(const amrex::Real Tin, const amrex::Real NCOeff1e5) -> amrex::Real
{
	const std::array<amrex::Real, 2> point = {Tin, NCOeff1e5};
	return gow_tables::getGowTables().co.interpolate_single(point, gow_tables::CO_ALPHA_IDX);
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto psigd_coll_interp2d(const amrex::Real nHtot, const amrex::Real Tin) -> amrex::Real
{
	// dust table axes are (Tg, nH) -> point = {T, nH}
	const std::array<amrex::Real, 2> point = {Tin, nHtot};
	return gow_tables::getGowTables().dust.interpolate_single(point, 0);
}

// Atomic-C photodissociation (C + PHOTON -> C+ + e-) self-shielding factors,
// tabulated vs the shielding column density (cm^-2): shielding by H2 evaluated
// at N(H2), and self-shielding evaluated at N(C). Called by the network RHS.
AMREX_GPU_HOST_DEVICE AMREX_INLINE auto interp_8_shielding_H2(const amrex::Real NH2) -> amrex::Real
{
	const std::array<amrex::Real, 1> point = {NH2};
	return gow_tables::getGowTables().c_shield.interpolate_single(point, gow_tables::SHIELD_H2_IDX);
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto interp_8_shielding_self(const amrex::Real NC) -> amrex::Real
{
	const std::array<amrex::Real, 1> point = {NC};
	return gow_tables::getGowTables().c_shield.interpolate_single(point, gow_tables::SHIELD_SELF_IDX);
}

// CO photodissociation (CO + PHOTON -> C + O) self-shielding factors, tabulated
// vs the shielding column density (cm^-2): shielding by H2 evaluated at N(H2),
// and self-shielding evaluated at N(CO). Called by the network RHS.
AMREX_GPU_HOST_DEVICE AMREX_INLINE auto interp_10_shielding_H2(const amrex::Real NH2) -> amrex::Real
{
	const std::array<amrex::Real, 1> point = {NH2};
	return gow_tables::getGowTables().co_shield.interpolate_single(point, gow_tables::SHIELD_H2_IDX);
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto interp_10_shielding_self(const amrex::Real NCO) -> amrex::Real
{
	const std::array<amrex::Real, 1> point = {NCO};
	return gow_tables::getGowTables().co_shield.interpolate_single(point, gow_tables::SHIELD_SELF_IDX);
}

// ---------------------------------------------------------------------------
// Partial derivatives (analytic Jacobian only; runtime uses numerical Jacobian).
// Delegated to DataTable::partial_derivative (centered finite difference tied to
// the local grid spacing), returning derivatives in physical units.
// ---------------------------------------------------------------------------

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto l0_co_interp1d_partial_0(const amrex::Real T) -> amrex::Real
{
	const std::array<amrex::Real, 1> point = {T};
	return gow_tables::getGowTables().l0.partial_derivative(point, 0, 0);
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto llte_co_interp2d_partial_0(const amrex::Real T, const amrex::Real y) -> amrex::Real
{
	const std::array<amrex::Real, 2> point = {T, y};
	return gow_tables::getGowTables().co.partial_derivative(point, 0, gow_tables::CO_LLTE_IDX);
}
AMREX_GPU_HOST_DEVICE AMREX_INLINE auto llte_co_interp2d_partial_1(const amrex::Real T, const amrex::Real y) -> amrex::Real
{
	const std::array<amrex::Real, 2> point = {T, y};
	return gow_tables::getGowTables().co.partial_derivative(point, 1, gow_tables::CO_LLTE_IDX);
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto nhalf_co_interp2d_partial_0(const amrex::Real T, const amrex::Real y) -> amrex::Real
{
	const std::array<amrex::Real, 2> point = {T, y};
	return gow_tables::getGowTables().co.partial_derivative(point, 0, gow_tables::CO_NHALF_IDX);
}
AMREX_GPU_HOST_DEVICE AMREX_INLINE auto nhalf_co_interp2d_partial_1(const amrex::Real T, const amrex::Real y) -> amrex::Real
{
	const std::array<amrex::Real, 2> point = {T, y};
	return gow_tables::getGowTables().co.partial_derivative(point, 1, gow_tables::CO_NHALF_IDX);
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto alpha_co_interp2d_partial_0(const amrex::Real T, const amrex::Real y) -> amrex::Real
{
	const std::array<amrex::Real, 2> point = {T, y};
	return gow_tables::getGowTables().co.partial_derivative(point, 0, gow_tables::CO_ALPHA_IDX);
}
AMREX_GPU_HOST_DEVICE AMREX_INLINE auto alpha_co_interp2d_partial_1(const amrex::Real T, const amrex::Real y) -> amrex::Real
{
	const std::array<amrex::Real, 2> point = {T, y};
	return gow_tables::getGowTables().co.partial_derivative(point, 1, gow_tables::CO_ALPHA_IDX);
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto psigd_coll_interp2d_partial_0(const amrex::Real nHtot, const amrex::Real T) -> amrex::Real
{
	// derivative w.r.t. nHtot -> dust axis 1
	const std::array<amrex::Real, 2> point = {T, nHtot};
	return gow_tables::getGowTables().dust.partial_derivative(point, 1, 0);
}
AMREX_GPU_HOST_DEVICE AMREX_INLINE auto psigd_coll_interp2d_partial_1(const amrex::Real nHtot, const amrex::Real T) -> amrex::Real
{
	// derivative w.r.t. T -> dust axis 0
	const std::array<amrex::Real, 2> point = {T, nHtot};
	return gow_tables::getGowTables().dust.partial_derivative(point, 0, 0);
}

#endif
