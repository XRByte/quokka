#ifndef GOW_INTERP_H
#define GOW_INTERP_H

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
	// The NCOeff (y) axis is log-spaced, so partial_derivative's step h_p ~ y; a
	// zero column (y = 0 at zones with no CO ahead) would give h_p = 0 and a
	// 0/0 = NaN derivative. Floor y to a tiny positive value (below the table, the
	// cooling coeff is clamped-flat, so the derivative there is ~0).
	const std::array<amrex::Real, 2> point = {T, amrex::max(y, static_cast<amrex::Real>(1.0e-30))};
	return gow_tables::getGowTables().co.partial_derivative(point, 1, gow_tables::CO_LLTE_IDX);
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto nhalf_co_interp2d_partial_0(const amrex::Real T, const amrex::Real y) -> amrex::Real
{
	const std::array<amrex::Real, 2> point = {T, y};
	return gow_tables::getGowTables().co.partial_derivative(point, 0, gow_tables::CO_NHALF_IDX);
}
AMREX_GPU_HOST_DEVICE AMREX_INLINE auto nhalf_co_interp2d_partial_1(const amrex::Real T, const amrex::Real y) -> amrex::Real
{
	// Floor y > 0 so the log-axis step h_p ~ y is nonzero (see llte_co_interp2d_partial_1).
	const std::array<amrex::Real, 2> point = {T, amrex::max(y, static_cast<amrex::Real>(1.0e-30))};
	return gow_tables::getGowTables().co.partial_derivative(point, 1, gow_tables::CO_NHALF_IDX);
}

AMREX_GPU_HOST_DEVICE AMREX_INLINE auto alpha_co_interp2d_partial_0(const amrex::Real T, const amrex::Real y) -> amrex::Real
{
	const std::array<amrex::Real, 2> point = {T, y};
	return gow_tables::getGowTables().co.partial_derivative(point, 0, gow_tables::CO_ALPHA_IDX);
}
AMREX_GPU_HOST_DEVICE AMREX_INLINE auto alpha_co_interp2d_partial_1(const amrex::Real T, const amrex::Real y) -> amrex::Real
{
	// Floor y > 0 so the log-axis step h_p ~ y is nonzero (see llte_co_interp2d_partial_1).
	const std::array<amrex::Real, 2> point = {T, amrex::max(y, static_cast<amrex::Real>(1.0e-30))};
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
