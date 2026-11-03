#ifndef actual_rhs_H
#define actual_rhs_H

#include <AMReX_Array.H>
#include <AMReX_REAL.H>
#include <cmath>

#include <actual_network.H>
#include <burn_type.H>
#include <gow_interp.H>
#include <extern_parameters.H>
#include <jacobian_utilities.H>

using namespace ArrayUtil;
using namespace network_rp;

// Network symbols mapped onto burn_t, shared by the RHS and the Jacobian.
// $$JAFF DEFINE microphysics {
    // REPLACE {
        // "radeden\[\s*(\d+)\s*\]" -> "state.rn[2*\1]"
        // "photden\[\s*(\d+)\s*\]" -> "state.rn[2*\1]"
        // "rflux\[\s*(\d+)\s*\]" -> "state.rn[2*\1+1]"
        // "nden\[\s*(\d+)\s*\]" -> "state.xn[\1]"
        // "tgas" -> "state.T"
        // "c_hat" -> "state.c_hat"
    // }
    // THERMAL { MODE -> dedt  ENERGY -> specific }
    // # RADIATION { ENABLED -> True }
// }

AMREX_GPU_HOST_DEVICE AMREX_INLINE void
actual_rhs(const burn_t &state, amrex::Array1D<amrex::Real, 1, neqs> &ydot) {

  using namespace Rates;
#ifdef PDR_PARALLEL_NCOL_AUX
	const amrex::Real ncol_H2 = state.aux[0];
	const amrex::Real ncol_CO = state.aux[1];
	const amrex::Real ncol_C = state.aux[2];
	const amrex::Real ncol_H = state.aux[3];
#else
	const amrex::Real ncol_H2 = network_rp::ncol_H2;
	const amrex::Real ncol_CO = network_rp::ncol_CO;
	const amrex::Real ncol_C = network_rp::ncol_C;
	const amrex::Real ncol_H = network_rp::ncol_H;
#endif
#if defined(PDR_PARALLEL_NCOL_AUX) && (NAUX_NET >= 5)
	const amrex::Real av = state.aux[4]; // per-cell visual extinction (host-supplied)
#else
	const amrex::Real av = network_rp::Av;
#endif
	const amrex::Real d2g = network_rp::dust2gas_ratio;
	const amrex::Real gradv = 3.0 * 3.0e-14;
	const amrex::Real vdisp = 3.0 * 1.0e5 / pow(2, 0.5);


  // $$JAFF REPEAT idx, rhs, cse IN rhses { USE microphysics }
  const amrex::Real cse0 = 6.3999999999999996e-10*state.xn[0]*state.xn[4];
  const amrex::Real cse1 = -cse0;
  const amrex::Real cse2 = std::sqrt(state.T);
  const amrex::Real cse3 = cse2/state.xn[2];
  const amrex::Real cse4 = 0.00011320754716981131*cse3*state.xn[3];
  const amrex::Real cse5 = std::min(10000000000.0, cse4);
  const amrex::Real cse6 = 1.0/(cse5 + 1);
  const amrex::Real cse7 = cse6*state.xn[17]*state.xn[3];
  const amrex::Real cse8 = 1.6000000000000001e-9*cse7;
  const amrex::Real cse9 = cse5*cse8;
  const amrex::Real cse10 = 1.99e-9*std::pow(state.T, -0.19)*state.xn[11]*state.xn[16];
  const amrex::Real cse11 = cse10*cse6;
  const amrex::Real cse12 = 3.5000000000000002e-11*state.xn[11]*state.xn[15];
  const amrex::Real cse13 = 3.7999999999999998e-10*chi*std::exp(-2.6600000000000001*av)*state.xn[15];
  const amrex::Real cse14 = 1.0/state.T;
  const amrex::Real cse15 = 7.9513696852843427e-10*std::pow(state.T, -0.33900000000000002)*std::exp(0.11*cse14)*state.xn[15]*state.xn[7];
  const amrex::Real cse16 = M_LN10;
  const amrex::Real cse17 = 1.0/cse16;
  const amrex::Real cse18 = 2*state.xn[3];
  const amrex::Real cse19 = cse18 + state.xn[0] + state.xn[10] + state.xn[14] + state.xn[15] + 3*state.xn[16] + state.xn[1] + 2*state.xn[4];
  const amrex::Real cse20 = std::log((1.0/10000.0)*state.T);
  const amrex::Real cse21 = std::pow(cse16, -2);
  const amrex::Real cse22 = std::pow(cse20, 2)*cse21;
  const amrex::Real cse23 = 1.0/cse19;
  const amrex::Real cse24 = cse23*state.xn[3];
  const amrex::Real cse25 = cse23*state.xn[0];
  const amrex::Real cse26 = std::pow(10, 0.41599999999999998*cse17*cse20 + 0.32700000000000001*cse22 - 3)*cse25 + std::pow(10, 1.3*cse17*cse20 - 1.6200000000000001*cse22 - 4.8449999999999998)*cse24;
  const amrex::Real cse27 = cse19*((cse26 < 1.0e-50) ? (
   1.0e-50
)
: (
   cse26
));
  const amrex::Real cse28 = cse17/(cse27 + 1);
  const amrex::Real cse29 = cse27*cse28;
  const amrex::Real cse30 = state.T > 700.0;
  const amrex::Real cse31 = ((cse30) ? (
   std::pow(10, cse28*std::log(5.9959999999999997e-30*std::pow(state.T, 4.1881000000000004)*std::pow(6.7610000000000002e-6*state.T + 1, -5.6881000000000004)*std::exp(-54657.400000000001*cse14)) + cse29*std::log(1.3000000000000001e-9*std::exp(-53300*cse14)))
)
: (
   0.0
))*std::pow(state.xn[3], 2);
  const amrex::Real cse32 = (7.5e-10*std::pow(state.T, -0.45800000000000002) + 4.9899999999999997e-11*std::pow(state.T, 0.40500000000000003))*state.xn[0]*state.xn[17];
  const amrex::Real cse33 = ((cse30) ? (
   std::pow(10, cse28*std::log(6.6699999999999996e-12*cse2*std::exp(-63590*cse14 - 1)) + cse29*std::log(3.5199999999999998e-9*std::exp(-43900*cse14)))
)
: (
   0.0
))*state.xn[0];
  const amrex::Real cse34 = std::pow(state.T, -0.52000000000000002);
  const amrex::Real cse35 = cse34*state.xn[16]*state.xn[2];
  const amrex::Real cse36 = cse19*d2g;
  const amrex::Real cse37 = (4.0000000000000001e-10*std::pow(state.T, 0.0066899999999999998) + 1.1000000000000001e-11*std::pow(state.T, 0.51700000000000002))*std::exp(-227*cse14)*state.xn[11]*state.xn[1];
  const amrex::Real cse38 = std::sqrt(2.0000000000000002e-15*ncol_H2 + 1);
  const amrex::Real cse39 = 0.96499999999999997/std::pow(1.4142135623730951e-10*ncol_H2/vdisp + 1, 2) + 0.035000000000000003*std::exp(-0.00084999999999999995*cse38)/cse38;
  const amrex::Real cse40 = std::exp(-4.1799999999999997*av);
  const amrex::Real cse41 = chi*cse39*cse40;
  const amrex::Real cse42 = std::pow(state.T, -1.3);
  const amrex::Real cse43 = std::exp(-23*cse14);
  const amrex::Real cse44 = cse42*cse43*state.xn[3]*state.xn[8];
  const amrex::Real cse45 = 7.0000000000000004e-11*state.xn[11]*state.xn[14];
  const amrex::Real cse46 = 9.0999999999999996e-10*chi*std::exp(-2.1200000000000001*av)*state.xn[14];
  const amrex::Real cse47 = cse45 + cse46;
  const amrex::Real cse48 = (26.218500000000002*std::sqrt(3)*std::sqrt(cse14) + 0.62)*state.xn[15];
  const amrex::Real cse49 = 1.3500000000000001e-9*cse48*state.xn[6];
  const amrex::Real cse50 = state.xn[3]*state.xn[6];
  const amrex::Real cse51 = 1.2599999999999999e-13*cse50*std::exp(-22.5*cse14);
  const amrex::Real cse52 = cse49 + cse51;
  const amrex::Real cse53 = 2.8142984982406731e-11*std::pow(state.T, 0.26000000000000001)*state.xn[0]*state.xn[14];
  const amrex::Real cse54 = 1.7599999999999999e-9*std::pow(state.T, 0.042000000000000003)*std::exp(-2.145922746781116e-5*state.T)*state.xn[3]*state.xn[4];
  const amrex::Real cse55 = -cse53 + cse54;
  const amrex::Real cse56 = crate*state.xn[9];
  const amrex::Real cse57 = 6.5199999999999996*cse56;
  const amrex::Real cse58 = crate*(2.2999999999999998*cse24 + 1.5*cse25);
  const amrex::Real cse59 = cse58*state.xn[0];
  const amrex::Real cse60 = std::log(8.6173000000000003e-5*state.T);
  const amrex::Real cse61 = state.xn[0]*state.xn[2];
  const amrex::Real cse62 = cse61*((cse30) ? (
   5.9078447946382801e-70*std::pow(state.T, 13.536555999999999)*std::exp(-2.0391498499999999e-6*std::pow(cse60, 8) + 0.000111954395*std::pow(cse60, 7) - 0.00263197617*std::pow(cse60, 6) + 0.034825597700000002*std::pow(cse60, 5) - 0.28770560000000001*std::pow(cse60, 4) + 1.56315498*std::pow(cse60, 3) - 5.7393287500000003*std::pow(cse60, 2))
)
: (
   0.0
));
  const amrex::Real cse63 = 1.05849862170431e-5*std::pow(state.T, -0.64000000000000001)*state.xn[10]*state.xn[2];
  const amrex::Real cse64 = 4.8813573300130879e-6*std::pow(cse14, 1.5)*std::pow(114.81344434488582*std::pow(cse14, 0.40699999999999997) + 1, -2.242)*state.xn[1]*state.xn[2];
  const amrex::Real cse65 = std::exp(-1.8700000000000001*av);
  const amrex::Real cse66 = chi*cse65;
  const amrex::Real cse67 = cse3*cse66;
  const amrex::Real cse68 = 1.7*cse67;
  const amrex::Real cse69 = std::log(state.T);
  const amrex::Real cse70 = 1.2250000000000001e-11*cse36*state.xn[1]/(1.6774414051711492e-5*std::pow(cse67, 1.3779999999999999)*(508.69999999999999*std::pow(cse68, -1.102e-5*cse69 - 0.4723)*std::pow(state.T, 0.015859999999999999) + 1) + 1);
  const amrex::Real cse71 = -cse57 - cse59 - cse62 + cse63 + cse64 + cse70;
  const amrex::Real cse72 = cse32 - cse37;
  const amrex::Real cse73 = cse18*cse58;
  const amrex::Real cse74 = 1.4595000290776864e-10*std::pow(state.T, -0.62)*state.xn[13]*state.xn[2];
  const amrex::Real cse75 = 4.4999999999999998e-9*chi*std::exp(-2.6099999999999999*av)*state.xn[12];
  const amrex::Real cse76 = 2.1659999999999999e-12*cse36*state.xn[13]/(1.534817233071468e-7*std::pow(cse67, 1.8740000000000001)*(43750.0*std::pow(cse68, -7.538e-5*cse69 - 0.89639999999999997)*std::pow(state.T, 1.635e-6) + 1) + 1);
  const amrex::Real cse77 = 16800.0*crate*cse24*state.xn[12];
  const amrex::Real cse78 = (1.0537934368240927e-9*std::pow(cse14, 0.00231) + std::pow(state.T, -1.5)*(0.00015100000000000001*std::exp(-8110.0*cse14) + 1.31e-7*std::exp(-26.600000000000001*cse14) + 3.4e-8*std::exp(-7.6200000000000001*cse14) + 6.9699999999999997e-9*std::exp(-1.3799999999999999*cse14)))*state.xn[16]*state.xn[7];
  const amrex::Real cse79 = cse23*state.xn[2];
  const amrex::Real cse80 = ((cse79 < 1.0e-50) ? (
   10000000000.0
)
: (
   cse4
));
  const amrex::Real cse81 = cse10*cse80/(cse80 + 1);
  const amrex::Real cse82 = cse11 + 1.3007021411393064e-6*cse35 + cse78 + cse81;
  const amrex::Real cse83 = cse17*cse69;
  const amrex::Real cse84 = -1.1000000000000001*crate*state.xn[5] + 5.5720000000000001e-12*cse36*state.xn[6]/(7.104740267838494e-7*std::pow(cse67, 1.512)*(5115.0*std::pow(cse68, -5.4939999999999998e-7*cse69 - 0.49559999999999998)*std::pow(state.T, 3.9029999999999999e-7) + 1) + 1) + 9.9999999999999994e-12*std::pow(state.T, -0.5)*(cse83*(cse83*(0.044330000000000001*cse83 - 0.28520000000000001) - 1.6759999999999999) + 11.19)*state.xn[2]*state.xn[6];
  const amrex::Real cse85 = crate*state.xn[7];
  const amrex::Real cse86 = 3.8500000000000001*cse85;
  const amrex::Real cse87 = 0.15970000000000001*std::exp(-49550.0*cse14);
  const amrex::Real cse88 = ((1.6339999999999999e-6*std::exp(-15230.0*cse14) + 9.7930000000000004e-9*std::exp(-73.799999999999997*cse14) + 6.3460000000000002e-9*std::exp(-12.17*cse14))/std::pow(state.T, 3.0/2.0) + 2.4460185352936309e-10*std::pow(0.00071740367932898903*cse2 + 1, -cse87 - 1.7848999999999999)*std::pow(12.244387999457523*cse2 + 1, cse87 - 0.21509999999999996)/cse2)*state.xn[2]*state.xn[8];
  const amrex::Real cse89 = 4.5579999999999993e-11*cse36*state.xn[8]/(0.011078790689599145*std::pow(cse67, 1.1279999999999999)*(433.10000000000002*std::pow(cse68, -0.00013329999999999999*cse69 - 0.81200000000000006)*std::pow(state.T, 0.04845) + 1) + 1);
  const amrex::Real cse90 = 1040.0*cse24*cse85;
  const amrex::Real cse91 = 2.808e-22*ncol_H2;
  const amrex::Real cse92 = 3.4999999999999998e-10*chi*std::exp(-3.7599999999999998*av)*std::exp(-cse91)*std::exp(-1.6000000000000001e-17*ncol_C)*state.xn[7]/(cse91 + 1);
  const amrex::Real cse93 = -cse86 + cse88 + cse89 - cse90 - cse92;
  const amrex::Real cse94 = 3.2999999999999996e-13*cse44;
  const amrex::Real cse95 = cse8 + cse9;
  const amrex::Real cse96 = cse94 + cse95;
  const amrex::Real cse97 = -cse78;
  const amrex::Real cse98 = -cse81;
  const amrex::Real cse99 = 1.6999999999999999e-9*state.xn[16]*state.xn[9];
  const amrex::Real cse100 = cse33*state.xn[3];
  const amrex::Real cse101 = 5.6999999999999997e-11*cse41;
  const amrex::Real cse102 = 7.2000000000000002e-15*cse50;
  const amrex::Real cse103 = cse1 + cse102 + cse73;
  const amrex::Real cse104 = -cse54;
  const amrex::Real cse105 = 1.6000000000000001e-9*state.xn[6]*state.xn[9];
  const amrex::Real cse106 = cse102 + cse105 + cse52 + cse84;
  const amrex::Real cse107 = cse53 + cse97;
  const amrex::Real cse108 = 184.0*cse24*cse56;
  const amrex::Real cse109 = 2.4e-10*chi*fshield_co_interp2d(ncol_CO, ncol_H2)*std::exp(-3.8799999999999999*av)*state.xn[9];
  const amrex::Real cse110 = cse108 + cse109 - cse15;
  const amrex::Real cse111 = 9.1500000000000005e-10*cse48*state.xn[8];
  const amrex::Real cse112 = cse57 - cse63 + cse99;
  const amrex::Real cse113 = cse105 - cse45;
  const amrex::Real cse114 = cse12 + cse13 + cse98;
  const amrex::Real cse115 = -cse74 + cse75 - cse76 + cse77;
  const amrex::Real cse116 = std::pow(state.T, 0.14999999999999999)/(0.00088240812266906664*std::pow(state.T, 0.65000000000000002) + 1);
  const amrex::Real cse117 = cse116*state.xn[2];
  const amrex::Real cse118 = std::exp(-118400.0*cse14)/(4.4460389837719569e-9*cse117 + 626500000.0);
  const amrex::Real cse119 = cse68 + 50;
  const amrex::Real cse120 = std::log(1.7*cse2*cse66/(state.xn[2] + 1.0e-50) + 50);
  const amrex::Real cse121 = 0.001*state.T;
  const amrex::Real cse122 = cse121 + 1;
  const amrex::Real cse123 = std::pow(cse122, -2);
  const amrex::Real cse124 = 1.0/cse122;
  const amrex::Real cse125 = 1.0/(cse23*(cse101 + 1.9999999999999999e-7)/(std::pow(10, -2.3900000000000001*cse123 + 0.055500000000000001*cse124 - 11.06)*cse25 + std::pow(10, -2.0230000000000001*cse123 - 3.6709999999999998*cse124 - 11.08)*cse24) + 1);
  const amrex::Real cse126 = std::sqrt(cse14)*state.xn[2];
  const amrex::Real cse127 = (1.0/100.0)*state.T;
  const amrex::Real cse128 = std::log(cse127);
  const amrex::Real cse129 = state.T < 500;
  const amrex::Real cse130 = std::pow(state.T, 0.070000000000000007);
  const amrex::Real cse131 = 4.5299999999999998e-6*cse126 + 7.5799999999999997e-10*std::pow(cse127, 0.0086999999999999994*cse128 + 0.12809999999999999)*state.xn[0] + (0.75*((cse129) ? (
   1.1000000000000001e-13*state.T + 5.3300000000000002e-10
)
: (
   3.74757785025e-10*cse130
)) + 0.25*((cse129) ? (
   3.3000000000000001e-13*state.T + 4.4299999999999997e-10
)
: (
   3.8899728635600003e-10*cse130
)))*state.xn[3];
  const amrex::Real cse132 = cse131*std::exp(-91.209999999999994*cse14)/(cse131 + 2.3e-6);
  const amrex::Real cse133 = cse17*std::log(cse19);
  const amrex::Real cse134 = std::pow(state.T, 1.0/4.0);
  const amrex::Real cse135 = 1.75*cse134*state.xn[0] + state.xn[3] + 680.10000000000002*state.xn[2]/cse134;
  const amrex::Real cse136 = l0_co_interp1d(state.T);
  const amrex::Real cse137 = 1.0/cse136;
  const amrex::Real cse138 = ((ncol_H <= 0) ? (
   100000.0*state.xn[9]/std::max(gradv, 5.7445531974402559e-18*M_SQRT2*cse2)
)
: (
   1756701454.4310031*ncol_CO/std::sqrt(std::max(1.0000000000000001e-30, cse23*ncol_H))
));
  const amrex::Real cse139 = 1.0/llte_co_interp2d(state.T, cse138);
  const amrex::Real cse140 = nhalf_co_interp2d(state.T, cse138);
  const amrex::Real cse141 = 0.042999999999999997*cse128;
  const amrex::Real cse142 = 4.34e-10*std::pow(cse127, 0.755 - 0.16*cse128)*state.xn[0] + 1.08e-14*std::pow(state.T, 0.92600000000000005)*state.xn[2] + (2.2500000000000003e-12*std::pow(cse127, 0.52500000000000002*cse128 + 1.198) + 5.2499999999999998e-13*std::pow(cse127, cse141 + 0.88900000000000001))*state.xn[3];
  const amrex::Real cse143 = 3.5700000000000001e-10*std::pow(cse127, 0.41899999999999998 - 0.0030000000000000001*cse128)*state.xn[0] + 5.1199999999999999e-10*std::pow(state.T, -0.074999999999999997)*state.xn[2] + (3.7250000000000001e-11*std::pow(cse127, 0.025000000000000001*cse128 + 0.26400000000000001) + 1.0274999999999999e-10*std::pow(cse127, cse141 + 0.29599999999999999))*state.xn[3];
  const amrex::Real cse144 = (1.0/5.0)*std::exp(-326.60000000000002*cse14);
  const amrex::Real cse145 = 3.1899999999999998e-10*std::pow(cse127, 0.36899999999999999 - 0.0060000000000000001*cse128)*state.xn[0] + 4.8599999999999998e-10*std::pow(state.T, -0.025999999999999999)*state.xn[2] + (4.7499999999999998e-11*std::pow(cse127, 0.041000000000000002*cse128 + 0.20300000000000001) + 1.6725000000000002e-10*std::pow(cse127, 0.058000000000000003*cse128 + 0.23699999999999999))*state.xn[3];
  const amrex::Real cse146 = cse143 + 8.9099999999999997e-5;
  const amrex::Real cse147 = cse144*cse145;
  const amrex::Real cse148 = cse142*cse143*cse144 + (1.0/15.0)*cse142*cse145*std::exp(-425.40000000000003*cse14) + cse146*cse147;
  const amrex::Real cse149 = cse142 + 1.7499999999999998e-5;
  const amrex::Real cse150 = cse145 + 1.34e-10;
  const amrex::Real cse151 = (3.0/5.0)*cse143*std::exp(-227.80000000000001*cse14);
  const amrex::Real cse152 = cse147*cse149 + cse149*cse151 + cse150*cse151;
  const amrex::Real cse153 = 1.0/((1.0/3.0)*cse142*cse150*std::exp(-98.800000000000011*cse14) + cse146*cse149 + cse146*cse150 + cse148 + cse152);
  const amrex::Real cse154 = state.T < 100.0;
  const amrex::Real cse155 = state.T < 6000.0;
  const amrex::Real cse156 = state.T < 10;
  const amrex::Real cse157 = std::log(cse121);
  const amrex::Real cse158 = cse157*cse17;
  const amrex::Real cse159 = std::pow(cse16, -8);
  const amrex::Real cse160 = std::pow(cse157, 8)*cse159;
  const amrex::Real cse161 = std::pow(cse16, -7);
  const amrex::Real cse162 = std::pow(cse157, 7)*cse161;
  const amrex::Real cse163 = std::pow(cse16, -6);
  const amrex::Real cse164 = std::pow(cse157, 6)*cse163;
  const amrex::Real cse165 = std::pow(cse16, -5);
  const amrex::Real cse166 = std::pow(cse157, 5)*cse165;
  const amrex::Real cse167 = std::pow(cse16, -4);
  const amrex::Real cse168 = std::pow(cse157, 4)*cse167;
  const amrex::Real cse169 = std::pow(cse16, -3);
  const amrex::Real cse170 = std::pow(cse157, 3);
  const amrex::Real cse171 = cse169*cse170;
  const amrex::Real cse172 = std::pow(cse157, 2)*cse21;
  const amrex::Real cse173 = std::pow(10.0, 16.815729999999999*cse158 + 70.609154000000004*cse160 + 364.14445999999998*cse162 + 801.81246999999996*cse164 + 983.67575999999997*cse166 + 734.71650999999997*cse168 + 343.1918*cse171 + 96.743155000000002*cse172 - 21.928795999999998)*state.xn[2];
  const amrex::Real cse174 = state.xn[5] + state.xn[6];
  const amrex::Real cse175 = std::pow(10.0, 1.5714710999999999*cse158 + 0.32168730000000001*cse166 - 0.51002221000000003*cse168 - 0.23619984999999999*cse171 + 0.015391166*cse172 - 22.089523)*state.xn[1] + std::pow(10.0, 2.1892372*cse158 + 0.19191374999999999*cse166 - 0.16596184*cse168 + 0.29036281000000003*cse171 - 0.81520437999999995*cse172 - 23.689236999999999)*cse174 + std::pow(10.0, 2.0943374000000001*cse157*cse17 - 0.033638326000000003*cse166 - 0.14913216000000001*cse168 + 0.43693353000000001*cse169*cse170 - 0.77151435999999995*cse172 - 23.962112000000001)*state.xn[3];
  const amrex::Real cse176 = std::pow(10, 37.383713*cse158 + 3.8479610000000002*cse166 + 20.159831000000001*cse168 + 48.656103000000002*cse171 + 58.145166000000003*cse172 - 16.818342000000001)*state.xn[0] + cse175;
  const amrex::Real cse177 = std::pow(10.0, 1.6802758*cse158 - 6.3701156000000001*cse160 + 6.4380698000000001*cse162 + 8.9167182999999994*cse164 - 8.8077017000000009*cse166 - 4.7274035999999997*cse168 + 4.0406627000000004*cse171 + 0.93310621999999999*cse172 - 22.921188999999998)*state.xn[2];
  const amrex::Real cse178 = state.T < 1000.0;
  const amrex::Real cse179 = std::pow(10.0, 3.5692468000000002*cse157*cse17 - 4.2519023000000002*cse166 - 21.328264000000001*cse168 - 27.850082*cse171 - 11.33286*cse172 - 24.311209000000002)*state.xn[0] + cse175;
  const amrex::Real cse180 = std::pow(10.0, 4.6450521*cse158 + 1.5538288*cse166 - 5.5108049000000001*cse168 + 5.9369081000000001*cse171 - 3.7209846*cse172 - 24.311209000000002)*state.xn[0] + cse175;
  const amrex::Real cse181 = std::pow(10.0, -0.62120210234156448*cse165 - 1.5370575959566981*cse167 + 2.5133588394664677*cse169 + 3.7525488682084647*cse17 - 2.4768712409536775*cse21 - 23.962112000000001)*state.xn[3] + std::pow(10.0, 3.5440891133599637*cse165 - 1.7105157385968939*cse167 + 1.6702447513374001*cse169 + 3.9225864834863131*cse17 - 2.6171337683481011*cse21 - 23.689236999999999)*cse174 + std::pow(10.0, 5.9406293599919788*cse165 - 5.2566362076906961*cse167 - 1.3586848802337363*cse169 + 2.8156982240432273*cse17 + 0.049411830040524525*cse21 - 22.089523)*state.xn[1] + std::pow(10.0, 28.69470131298657*cse165 - 56.79810800956944*cse167 + 34.150687525022562*cse169 + 8.3228160852326614*cse17 - 11.945856385319289*cse21 - 24.311209000000002)*state.xn[0];
  const amrex::Real cse182 = ((cse155) ? (
   cse14
)
: (
   0.00016666666666666666
));
  const amrex::Real cse183 = 1.6e-18*std::exp(-11700.0*cse182) + 6.7e-19*std::exp(-5860.0*cse182) + 3.0e-24*std::exp(-510.0*cse182) + 4.985670872372847e-33*((cse155) ? (
   std::pow(state.T, 3.7599999999999998)
)
: (
   160638152785302.53
))*std::exp(-2197000.0*((cse155) ? (
   std::pow(state.T, -3)
)
: (
   4.6296296296296294e-12
)))/(6.0142468035272636e-8*((cse155) ? (
   std::pow(state.T, 2.1000000000000001)
)
: (
   85924621.118090734
)) + 1.0);
  const amrex::Real cse184 = std::log(0.001*((cse155) ? (
   state.T
)
: (
   6000.0
)));
  const amrex::Real cse185 = cse165*std::pow(cse184, 5);
  const amrex::Real cse186 = cse167*std::pow(cse184, 4);
  const amrex::Real cse187 = std::pow(cse184, 3);
  const amrex::Real cse188 = std::pow(cse184, 2)*cse21;
  const amrex::Real cse189 = cse17*cse184;
  const amrex::Real cse190 = cse169*cse187;
  const amrex::Real cse191 = cse159*std::pow(cse184, 8);
  const amrex::Real cse192 = cse161*std::pow(cse184, 7);
  const amrex::Real cse193 = cse163*std::pow(cse184, 6);
  const amrex::Real cse194 = 0.045999999999999999*cse128;
  const amrex::Real cse195 = 1.7258000000000001e-6*cse126;
  const amrex::Real cse196 = 8.9000000000000003e-11*std::pow(cse127, cse194 + 0.22800000000000001)*state.xn[0] + cse195*std::exp(((cse178) ? (
   cse69*(cse69*(cse69*(0.0070527699999999999*cse69 - 0.11133800000000001) + 0.69763799999999998) - 1.3074300000000001) - 7.6973500000000001
)
: (
   cse69*(cse69*(cse69*(0.093813800000000003*cse69 - 3.0328300000000001) + 36.180300000000003) - 187.47399999999999) + 350.60899999999998
))) + (5.1749999999999996e-11*std::pow(cse127, 0.037999999999999999*cse128 + 0.16900000000000001) + 2.15e-11*std::pow(cse127, 0.048000000000000001*cse128 - 0.01))*state.xn[3];
  const amrex::Real cse197 = 2.6400000000000002e-10*std::pow(cse127, cse194 + 0.23100000000000001)*state.xn[0] + cse195*std::exp(((cse178) ? (
   cse69*(cse69*(cse69*(0.00235272*cse69 - 0.041816600000000002) + 0.35826400000000003) - 0.57443) - 7.4386999999999999
)
: (
   cse69*(cse69*(cse69*(0.097857299999999994*cse69 - 3.1926800000000002) + 38.504899999999999) - 202.19300000000001) + 386.18599999999998
))) + (1.11e-10*std::pow(cse127, 0.031*cse128 + 0.26300000000000001) + 4.3749999999999998e-11*std::pow(cse127, 0.064000000000000001*cse128 + 0.071999999999999995))*state.xn[3];
  const amrex::Real cse198 = 2.8763333333333333e-6*cse126*std::exp(((cse178) ? (
   cse69*(cse69*(cse69*(-0.00065632500000000003*cse69 - 0.015089200000000001) + 0.36118400000000001) - 0.77378199999999997) - 9.2514099999999999
)
: (
   cse69*(cse69*(cse69*(0.10508000000000001*cse69 - 3.4762) + 42.594999999999999) - 227.91300000000001) + 444.60000000000002
))) + 1.26e-10*std::pow(cse127, 0.057000000000000002*cse128 + 0.115)*state.xn[0] + (5.3249999999999997e-11*std::pow(cse127, 0.049000000000000002*cse128 - 0.0040000000000000001) + 1.675e-11*std::pow(cse127, 0.10199999999999999*cse128 - 0.085000000000000006))*state.xn[3];
  const amrex::Real cse199 = 5*std::exp(-62.460000000000001*cse14);
  const amrex::Real cse200 = cse198 + 7.8800000000000004e-8;
  const amrex::Real cse201 = cse196*cse199;
  const amrex::Real cse202 = (25.0/3.0)*cse196*cse197*std::exp(-101.30000000000001*cse14) + cse197*cse198*cse199 + cse200*cse201;
  const amrex::Real cse203 = cse197 + 2.65e-7;
  const amrex::Real cse204 = cse196 + 1.81e-14;
  const amrex::Real cse205 = 3*cse198*std::exp(-23.619999999999997*cse14);
  const amrex::Real cse206 = cse201*cse203 + cse203*cse205 + cse204*cse205;
  const amrex::Real cse207 = 1.0/((5.0/3.0)*cse197*cse204*std::exp(-38.840000000000003*cse14) + cse200*cse203 + cse200*cse204 + cse202 + cse206);
  ydot(1) = cse1 + cse11 + cse12 + cse13 + cse15 + cse18*cse33 + 2*cse31 - cse32 + 2.9935562710997168e-6*cse35 - 6.0000000000000005e-15*cse36*state.xn[0] + cse37 + 1.1399999999999999e-10*cse41*state.xn[3] + 4.2899999999999994e-13*cse44 + cse47 + cse52 + cse55 + 3.2000000000000001e-9*cse7 + cse71 + cse9;
  ydot(2) = cse0 + cse51 + cse59 + cse62 - cse64 - cse70 + cse72;
  ydot(3) = -cse71 + cse73 - cse74 + cse75 - cse76 + cse77 - cse82 - cse84 - cse93 - cse96;
  ydot(4) = -cse100 - cse101*state.xn[3] - cse103 + cse11 + 3.0000000000000002e-15*cse19*d2g*state.xn[0] - cse31 + 4.5427507615910105e-7*cse34*state.xn[16]*state.xn[2] - cse51 - cse55 - cse96 - cse97 - cse98 + cse99;
  ydot(5) = cse103 + cse104;
  ydot(6) = cse106;
  ydot(7) = -cse106;
  ydot(8) = cse107 + cse110 + 9.8999999999999995e-14*cse44 + cse46 + cse93;
  ydot(9) = cse105 - cse111 + cse86 - cse88 - cse89 + cse90 + cse92 - cse94;
  ydot(10) = -cse110 - cse112 - cse113;
  ydot(11) = cse111 + cse112;
  ydot(12) = cse108 + cse109 + cse113 + cse114 + cse72 + cse8;
  ydot(13) = -cse115;
  ydot(14) = cse115;
  ydot(15) = -cse107 + 2.3099999999999997e-13*cse42*cse43*state.xn[3]*state.xn[8] - cse47;
  ydot(16) = -cse111 - cse114 - cse15 - cse49 + cse9;
  ydot(17) = -cse104 - cse82 - cse99;
  ydot(18) = cse49 - cse72 - cse95;
  ydot(19) = (1.7e-24*chi*cse19*cse65*d2g*(2.25*std::pow(state.T, 0.14699999999999999) + 5.2199999999999998)/(0.049959999999999997*std::pow(cse119, 0.43099999999999999)*(0.0043*std::pow(cse119, 0.69199999999999995) + 1) + 1) + 5.6999999999999997e-11*chi*cse39*cse40*(2.5634832e-11*cse125 + 6.4087080000000003e-13)*state.xn[3] + 1.602177e-12*crate*(26.399999999999999*std::sqrt(cse79/(cse79 + 0.070000000000000007)) + 6.5)*state.xn[0] + 3.2043540000000001e-12*crate*((cse133 < 2) ? (
   10.0
)
: ((cse133 < 4) ? (
   (3.0/2.0)*cse133 + 7
)
: ((cse133 < 7) ? (
   (4.0/3.0)*cse133 + 23.0/3.0
)
: ((cse133 < 10) ? (
   (1.0/3.0)*cse133 + 44.0/3.0
)
: (
   18.0
)))))*state.xn[3] - 7.1777529600000012e-12*cse100 - 1.3659996292638161e-10*cse116*cse118*cse61/(1.3338116951315871e-8*cse117*cse118 + 1) - 5.7927337833400002e-20*cse132*state.xn[8]/(2*cse132 + 1) - cse135*std::pow(1 - std::exp(-state.T), 1000)*state.xn[9]/(cse135*cse139 + cse137*std::pow(cse135/cse140, alpha_co_interp2d(state.T, cse138))*(-cse136*cse139*cse140 + 1) + cse137) + 3.0000000000000002e-15*cse19*d2g*(6.7291434000000005e-12*cse125 + 3.2043540000000002e-13)*state.xn[0] - 7.1777529600000012e-12*cse31 - 1.8998126555754159e-29*cse36*std::pow(state.T, 0.45350000000000001 + 2.234/cse120)*std::exp(cse120*(1.4419999999999999 - 0.050889999999999998*cse120))*state.xn[2] - 2.1789607199999999e-11*cse62 - (2.3872025442750958e-19*cse148*cse153 + 2.8023005140019997e-18*cse152*cse153)*state.xn[11] - ((((cse129 && cse154 && cse155) ? (
   cse156 || cse173 + cse176 < 1.0e-100
)
: (
   ((cse154 && cse155) ? (
      cse156 || cse176 + cse177 < 1.0e-100
   )
   : (
      ((cse129 && cse155 && cse178) ? (
         cse156 || cse173 + cse179 < 1.0e-100
      )
      : (
         ((cse155 && cse178) ? (
            cse156 || cse177 + cse179 < 1.0e-100
         )
         : (
            ((cse129 && cse155) ? (
               cse156 || cse173 + cse180 < 1.0e-100
            )
            : (
               ((cse155) ? (
                  cse156 || cse177 + cse180 < 1.0e-100
               )
               : (
                  ((cse129) ? (
                     cse156 || std::pow(10.0, 7500.646099484944*cse159 + 21588.957206746247*cse161 + 26530.843406310876*cse163 + 18165.631967965237*cse165 + 7572.4886742758581*cse167 + 1974.131269262874*cse169 + 30.129743459482278*cse17 + 310.58441786958315*cse21 - 21.928795999999998)*state.xn[2] + cse181 < 1.0e-100
                  )
                  : (
                     cse156 || std::pow(10.0, -676.68255490510751*cse159 + 381.69251128040054*cse161 + 295.04162851880631*cse163 - 162.65264843551881*cse165 - 48.723840736518795*cse167 + 23.242975457496804*cse169 + 3.0106500755647456*cse17 + 2.9956460707652877*cse21 - 22.921188999999998)*state.xn[2] + cse181 < 1.0e-100
                  ))
               ))
            ))
         ))
      ))
   ))
))) ? (
   0.0
)
: (
   cse183/(cse183/(std::pow(10.0, 0.19191374999999999*cse185 - 0.16596184*cse186 - 0.81520437999999995*cse188 + 2.1892372*cse189 + 0.29036281000000003*cse190 - 23.689236999999999)*cse174 + std::pow(10.0, 0.32168730000000001*cse185 - 0.51002221000000003*cse186 + 0.015391166*cse188 + 1.5714710999999999*cse189 - 0.23619984999999999*cse190 - 22.089523)*state.xn[1] + std::pow(10.0, 0.43693353000000001*cse169*cse187 + 2.0943374000000001*cse17*cse184 - 0.033638326000000003*cse185 - 0.14913216000000001*cse186 - 0.77151435999999995*cse188 - 23.962112000000001)*state.xn[3] + ((cse129) ? (
      std::pow(10.0, 983.67575999999997*cse185 + 734.71650999999997*cse186 + 96.743155000000002*cse188 + 16.815729999999999*cse189 + 343.1918*cse190 + 70.609154000000004*cse191 + 364.14445999999998*cse192 + 801.81246999999996*cse193 - 21.928795999999998)
   )
   : (
      std::pow(10.0, -8.8077017000000009*cse185 - 4.7274035999999997*cse186 + 0.93310621999999999*cse188 + 1.6802758*cse189 + 4.0406627000000004*cse190 - 6.3701156000000001*cse191 + 6.4380698000000001*cse192 + 8.9167182999999994*cse193 - 22.921188999999998)
   ))*state.xn[2] + ((((cse155) ? (
      cse154
   )
   : (
      false
   ))) ? (
      std::pow(10, 3.8479610000000002*cse185 + 20.159831000000001*cse186 + 58.145166000000003*cse188 + 37.383713*cse189 + 48.656103000000002*cse190 - 16.818342000000001)
   )
   : ((((cse155) ? (
      cse178
   )
   : (
      false
   ))) ? (
      std::pow(10.0, 3.5692468000000002*cse17*cse184 - 4.2519023000000002*cse185 - 21.328264000000001*cse186 - 11.33286*cse188 - 27.850082*cse190 - 24.311209000000002)
   )
   : (
      std::pow(10.0, 1.5538288*cse185 - 5.5108049000000001*cse186 - 3.7209846*cse188 + 4.6450521*cse189 + 5.9369081000000001*cse190 - 24.311209000000002)
   )))*state.xn[0]) + 1)
))*state.xn[3] - ((state.T < 1000000.0) ? (
   1.4210469458259591e-21*cse202*cse207 + 2.5697412351439998e-22*cse206*cse207
)
: (
   0.0
))*state.xn[7] - std::min(100.0*cse36*psigd_coll_interp2d(cse19, state.T), 3.2000000000000002e-32*std::pow(cse19, 2)*cse2*d2g*(state.T - 2.1544346900318838*std::pow(85973.158732959506*cse66 + 4.1397668473788896, 1.0/6.0))))/(1.6737729999999998e-24*state.xn[0] + 4.8185222061629005e-23*state.xn[10] + 2.6567630000000005e-23*state.xn[11] + 4.6637070000000001e-23*state.xn[12] + 4.6636159061629004e-23*state.xn[13] + 2.1618503000000003e-23*state.xn[14] + 2.8241403000000005e-23*state.xn[15] + 5.0204080616289998e-24*state.xn[16] + 2.6566719061629008e-23*state.xn[17] + 1.6728620616289998e-24*state.xn[1] + 9.1093837099999994e-28*state.xn[2] + 3.3475459999999996e-24*state.xn[3] + 3.3466350616289996e-24*state.xn[4] + 6.6464729999999996e-24*state.xn[5] + 6.6455620616289996e-24*state.xn[6] + 1.9944730000000003e-23*state.xn[7] + 1.9943819061629003e-23*state.xn[8] + 4.6512360000000008e-23*state.xn[9]);
  // $$JAFF END
}

template <class MatrixType>
AMREX_GPU_HOST_DEVICE AMREX_INLINE void actual_jac(const burn_t &state,
                                                   MatrixType &jac) {
  jac.zero();
#ifdef PDR_PARALLEL_NCOL_AUX
	const amrex::Real ncol_H2 = state.aux[0];
	const amrex::Real ncol_CO = state.aux[1];
	const amrex::Real ncol_C = state.aux[2];
	const amrex::Real ncol_H = state.aux[3];
#else
	const amrex::Real ncol_H2 = network_rp::ncol_H2;
	const amrex::Real ncol_CO = network_rp::ncol_CO;
	const amrex::Real ncol_C = network_rp::ncol_C;
	const amrex::Real ncol_H = network_rp::ncol_H;
#endif
#if defined(PDR_PARALLEL_NCOL_AUX) && (NAUX_NET >= 5)
	const amrex::Real av = state.aux[4]; // per-cell visual extinction (host-supplied)
#else
	const amrex::Real av = network_rp::Av;
#endif
	const amrex::Real d2g = network_rp::dust2gas_ratio;
	const amrex::Real gradv = 3.0 * 3.0e-14;
	const amrex::Real vdisp = 3.0 * 1.0e5 / pow(2, 0.5);


  // $$JAFF REPEAT idx, expr, cse IN jacobian { USE microphysics }
  const amrex::Real cse0 = 2.8142984982406731e-11*std::pow(state.T, 0.26000000000000001);
  const amrex::Real cse1 = cse0*state.xn[14];
  const amrex::Real cse2 = 7.5e-10*std::pow(state.T, -0.45800000000000002) + 4.9899999999999997e-11*std::pow(state.T, 0.40500000000000003);
  const amrex::Real cse3 = -cse2;
  const amrex::Real cse4 = std::pow(state.xn[3], 2);
  const amrex::Real cse5 = 1.0/state.T;
  const amrex::Real cse6 = std::log(1.3000000000000001e-9*std::exp(-53300*cse5));
  const amrex::Real cse7 = M_LN10;
  const amrex::Real cse8 = 1.0/cse7;
  const amrex::Real cse9 = std::log((1.0/10000.0)*state.T);
  const amrex::Real cse10 = std::pow(cse7, -2);
  const amrex::Real cse11 = cse10*std::pow(cse9, 2);
  const amrex::Real cse12 = std::pow(10, -1.6200000000000001*cse11 + 1.3*cse8*cse9 - 4.8449999999999998);
  const amrex::Real cse13 = 2*state.xn[3];
  const amrex::Real cse14 = cse13 + state.xn[0] + state.xn[10] + state.xn[14] + state.xn[15] + 3*state.xn[16] + state.xn[1] + 2*state.xn[4];
  const amrex::Real cse15 = 1.0/cse14;
  const amrex::Real cse16 = cse12*cse15*state.xn[3];
  const amrex::Real cse17 = std::pow(10, 0.32700000000000001*cse11 + 0.41599999999999998*cse8*cse9 - 3);
  const amrex::Real cse18 = cse15*cse17*state.xn[0];
  const amrex::Real cse19 = cse16 + cse18;
  const amrex::Real cse20 = cse19 < 1.0e-50;
  const amrex::Real cse21 = ((cse20) ? (
   1.0e-50
)
: (
   cse19
));
  const amrex::Real cse22 = cse14*cse21;
  const amrex::Real cse23 = cse22 + 1;
  const amrex::Real cse24 = cse8/cse23;
  const amrex::Real cse25 = cse21*cse24;
  const amrex::Real cse26 = cse25*cse6;
  const amrex::Real cse27 = std::pow(cse14, 2);
  const amrex::Real cse28 = 1.0/cse27;
  const amrex::Real cse29 = cse28*state.xn[3];
  const amrex::Real cse30 = cse12*cse29;
  const amrex::Real cse31 = cse28*state.xn[0];
  const amrex::Real cse32 = cse17*cse31;
  const amrex::Real cse33 = cse30 + cse32;
  const amrex::Real cse34 = cse14*((cse20) ? (
   0
)
: (
   cse15*cse17 - cse33
));
  const amrex::Real cse35 = cse24*cse6;
  const amrex::Real cse36 = std::pow(state.T, 4.1881000000000004);
  const amrex::Real cse37 = 6.7610000000000002e-6*state.T + 1;
  const amrex::Real cse38 = std::pow(cse37, 5.6881000000000004);
  const amrex::Real cse39 = 54657.400000000001*cse5;
  const amrex::Real cse40 = std::exp(-cse39);
  const amrex::Real cse41 = cse40/cse38;
  const amrex::Real cse42 = std::log(5.9959999999999997e-30*cse36*cse41);
  const amrex::Real cse43 = cse8/std::pow(cse23, 2);
  const amrex::Real cse44 = cse43*(-cse21 - cse34);
  const amrex::Real cse45 = cse22*cse44;
  const amrex::Real cse46 = std::pow(10, cse22*cse35 + cse24*cse42);
  const amrex::Real cse47 = cse46*cse7;
  const amrex::Real cse48 = state.T > 700.0;
  const amrex::Real cse49 = ((cse48) ? (
   cse47*(cse26 + cse34*cse35 + cse42*cse44 + cse45*cse6)
)
: (
   0
));
  const amrex::Real cse50 = std::sqrt(state.T);
  const amrex::Real cse51 = 63590*cse5 + 1;
  const amrex::Real cse52 = std::exp(-cse51);
  const amrex::Real cse53 = std::log(6.6699999999999996e-12*cse50*cse52);
  const amrex::Real cse54 = std::log(3.5199999999999998e-9*std::exp(-43900*cse5));
  const amrex::Real cse55 = cse24*cse54;
  const amrex::Real cse56 = std::pow(10, cse22*cse55 + cse24*cse53);
  const amrex::Real cse57 = ((cse48) ? (
   cse56
)
: (
   0.0
));
  const amrex::Real cse58 = 6.0000000000000005e-15*d2g;
  const amrex::Real cse59 = cse58*state.xn[0];
  const amrex::Real cse60 = cse25*cse54;
  const amrex::Real cse61 = cse56*cse7;
  const amrex::Real cse62 = ((cse48) ? (
   cse61*(cse34*cse55 + cse44*cse53 + cse45*cse54 + cse60)
)
: (
   0
));
  const amrex::Real cse63 = 1.6737729999999998e-24*state.xn[0] + 4.8185222061629005e-23*state.xn[10] + 2.6567630000000005e-23*state.xn[11] + 4.6637070000000001e-23*state.xn[12] + 4.6636159061629004e-23*state.xn[13] + 2.1618503000000003e-23*state.xn[14] + 2.8241403000000005e-23*state.xn[15] + 5.0204080616289998e-24*state.xn[16] + 2.6566719061629008e-23*state.xn[17] + 1.6728620616289998e-24*state.xn[1] + 9.1093837099999994e-28*state.xn[2] + 3.3475459999999996e-24*state.xn[3] + 3.3466350616289996e-24*state.xn[4] + 6.6464729999999996e-24*state.xn[5] + 6.6455620616289996e-24*state.xn[6] + 1.9944730000000003e-23*state.xn[7] + 1.9943819061629003e-23*state.xn[8] + 4.6512360000000008e-23*state.xn[9];
  const amrex::Real cse64 = 1.380649e-16*state.xn[0] + 1.380649e-16*state.xn[10] + 1.380649e-16*state.xn[11] + 1.380649e-16*state.xn[12] + 1.380649e-16*state.xn[13] + 1.380649e-16*state.xn[14] + 1.380649e-16*state.xn[15] + 1.380649e-16*state.xn[16] + 1.380649e-16*state.xn[17] + 1.380649e-16*state.xn[1] + 1.380649e-16*state.xn[2] + 1.380649e-16*state.xn[3] + 1.380649e-16*state.xn[4] + 1.380649e-16*state.xn[5] + 1.380649e-16*state.xn[6] + 1.380649e-16*state.xn[7] + 1.380649e-16*state.xn[8] + 1.380649e-16*state.xn[9];
  const amrex::Real cse65 = 1.0/cse64;
  const amrex::Real cse66 = 1.0/cse63;
  const amrex::Real cse67 = 2.0709734999998963e-16*cse66*state.T;
  const amrex::Real cse68 = std::pow(cse63, -2);
  const amrex::Real cse69 = 2.5106594999998743e-24*cse64*cse68*state.T - cse67;
  const amrex::Real cse70 = std::pow(state.T, -2);
  const amrex::Real cse71 = std::exp(-22.5*cse5);
  const amrex::Real cse72 = 2.8349999999999999e-12*cse70*cse71*state.xn[3]*state.xn[6];
  const amrex::Real cse73 = -cse72;
  const amrex::Real cse74 = 7.3171760954257497e-12*std::pow(state.T, -0.73999999999999999)*state.xn[0]*state.xn[14];
  const amrex::Real cse75 = std::sqrt(3);
  const amrex::Real cse76 = std::sqrt(cse5);
  const amrex::Real cse77 = cse5*cse75*cse76*state.xn[15];
  const amrex::Real cse78 = 1.7697487500000004e-8*cse77*state.xn[6];
  const amrex::Real cse79 = cse22*cse24*cse70;
  const amrex::Real cse80 = cse10*cse5;
  const amrex::Real cse81 = cse80*cse9;
  const amrex::Real cse82 = cse5*cse8;
  const amrex::Real cse83 = ((cse20) ? (
   0
)
: (
   cse16*cse7*(1.3*cse5*cse8 - 3.2400000000000002*cse81) + cse18*cse7*(0.65400000000000003*cse81 + 0.41599999999999998*cse82)
));
  const amrex::Real cse84 = cse14*cse83;
  const amrex::Real cse85 = cse43*cse84;
  const amrex::Real cse86 = cse21*cse27*cse43*cse83;
  const amrex::Real cse87 = ((cse48) ? (
   cse47*(1.6677785190126752e+29*cse24*cse38*(-2.3058963562359999e-34*cse36*std::pow(cse37, -6.6881000000000004)*cse40 + 3.2772577039999999e-25*cse41*std::pow(state.T, 2.1881000000000004) + 2.51118476e-29*cse41*std::pow(state.T, 3.1881000000000004))*std::exp(cse39)/cse36 + cse35*cse84 - cse42*cse85 - cse6*cse86 + 53300.0*cse79)
)
: (
   0
));
  const amrex::Real cse88 = (-3.435e-10*std::pow(state.T, -1.458) + 2.0209499999999999e-11*std::pow(state.T, -0.59499999999999997))*state.xn[0]*state.xn[17];
  const amrex::Real cse89 = 1.0/cse50;
  const amrex::Real cse90 = std::pow(state.T, -3.0/2.0);
  const amrex::Real cse91 = ((cse48) ? (
   cse61*(149925037481.25937*cse24*cse89*(3.3349999999999998e-12*cse52*cse89 + 4.2414529999999999e-7*cse52*cse90)*std::exp(cse51) - cse53*cse85 - cse54*cse86 + cse55*cse84 + 43900.0*cse79)
)
: (
   0
));
  const amrex::Real cse92 = std::pow(state.T, -1.52)*state.xn[16]*state.xn[2];
  const amrex::Real cse93 = std::exp(-227*cse5);
  const amrex::Real cse94 = cse93*(2.6759999999999999e-12*std::pow(state.T, -0.99331000000000003) + 5.6870000000000003e-12*std::pow(state.T, -0.48299999999999998))*state.xn[11]*state.xn[1];
  const amrex::Real cse95 = std::pow(state.T, -2.2999999999999998);
  const amrex::Real cse96 = std::exp(-23*cse5);
  const amrex::Real cse97 = cse96*state.xn[3]*state.xn[8];
  const amrex::Real cse98 = cse95*cse97;
  const amrex::Real cse99 = std::pow(state.T, 0.042000000000000003);
  const amrex::Real cse100 = std::exp(-2.145922746781116e-5*state.T);
  const amrex::Real cse101 = cse100*state.xn[4];
  const amrex::Real cse102 = cse101*state.xn[3];
  const amrex::Real cse103 = 3.7768240343347639e-14*cse102*cse99;
  const amrex::Real cse104 = 7.3920000000000002e-11*cse102*std::pow(state.T, -0.95799999999999996);
  const amrex::Real cse105 = std::pow(state.T, -3.2999999999999998);
  const amrex::Real cse106 = 4.0000000000000001e-10*std::pow(state.T, 0.0066899999999999998) + 1.1000000000000001e-11*std::pow(state.T, 0.51700000000000002);
  const amrex::Real cse107 = cse106*cse93;
  const amrex::Real cse108 = cse107*state.xn[11];
  const amrex::Real cse109 = 227*cse108*cse70*state.xn[1];
  const amrex::Real cse110 = 1.0/state.xn[2];
  const amrex::Real cse111 = cse110*cse89;
  const amrex::Real cse112 = cse110*cse50;
  const amrex::Real cse113 = 0.00011320754716981131*cse112;
  const amrex::Real cse114 = cse113*state.xn[3];
  const amrex::Real cse115 = (((cse114 - 10000000000.0 > 0) ? (
   0
)
: ((cse114 - 10000000000.0 == 0) ? (
   1.0/2.0
)
: (
   1
))));
  const amrex::Real cse116 = cse115*state.xn[17];
  const amrex::Real cse117 = cse116*cse4;
  const amrex::Real cse118 = cse111*cse117;
  const amrex::Real cse119 = std::min(10000000000.0, cse114);
  const amrex::Real cse120 = cse119 + 1;
  const amrex::Real cse121 = std::pow(cse120, -2);
  const amrex::Real cse122 = 1.811320754716981e-13*cse121;
  const amrex::Real cse123 = 6.7743911789075847e-6*std::pow(state.T, -1.6400000000000001)*state.xn[10]*state.xn[2];
  const amrex::Real cse124 = std::exp(0.11*cse5);
  const amrex::Real cse125 = cse124*state.xn[15];
  const amrex::Real cse126 = cse125*state.xn[7];
  const amrex::Real cse127 = 8.7465066538127767e-11*cse126*std::pow(state.T, -2.339) + 2.6955143233113923e-10*cse126*std::pow(state.T, -1.339);
  const amrex::Real cse128 = cse123 + cse127;
  const amrex::Real cse129 = std::log(8.6173000000000003e-5*state.T);
  const amrex::Real cse130 = std::pow(cse129, 2);
  const amrex::Real cse131 = std::pow(cse129, 3);
  const amrex::Real cse132 = std::pow(cse129, 4);
  const amrex::Real cse133 = std::pow(cse129, 5);
  const amrex::Real cse134 = std::pow(cse129, 6);
  const amrex::Real cse135 = std::pow(cse129, 7);
  const amrex::Real cse136 = std::exp(-2.0391498499999999e-6*std::pow(cse129, 8) - 5.7393287500000003*cse130 + 1.56315498*cse131 - 0.28770560000000001*cse132 + 0.034825597700000002*cse133 - 0.00263197617*cse134 + 0.000111954395*cse135);
  const amrex::Real cse137 = 5.9078447946382801e-70*cse136*std::pow(state.T, 13.536555999999999);
  const amrex::Real cse138 = state.xn[0]*state.xn[2];
  const amrex::Real cse139 = cse138*((cse48) ? (
   7.9971871901929577e-69*cse136*std::pow(state.T, 12.536555999999999) + cse137*(-11.478657500000001*cse129*cse5 + 4.6894649399999997*cse130*cse5 - 1.1508224*cse131*cse5 + 0.1741279885*cse132*cse5 - 0.015791857020000001*cse133*cse5 + 0.00078368076500000001*cse134*cse5 - 1.6313198799999999e-5*cse135*cse5)
)
: (
   0
));
  const amrex::Real cse140 = std::pow(cse5, 1.5);
  const amrex::Real cse141 = 114.81344434488582*std::pow(cse5, 0.40699999999999997) + 1;
  const amrex::Real cse142 = std::pow(cse141, -2.242);
  const amrex::Real cse143 = cse5*state.xn[1]*state.xn[2];
  const amrex::Real cse144 = 7.3220359950196314e-6*cse140*cse142*cse143;
  const amrex::Real cse145 = 0.0005114031087522854*std::pow(cse141, -3.242)*cse143*std::pow(cse5, 1.907);
  const amrex::Real cse146 = std::exp(-1.8700000000000001*av);
  const amrex::Real cse147 = chi*cse146;
  const amrex::Real cse148 = cse112*cse147;
  const amrex::Real cse149 = std::pow(cse148, 1.3779999999999999);
  const amrex::Real cse150 = 1.7*cse148;
  const amrex::Real cse151 = std::log(state.T);
  const amrex::Real cse152 = 1.102e-5*cse151 + 0.4723;
  const amrex::Real cse153 = std::pow(cse150, -cse152);
  const amrex::Real cse154 = cse153*std::pow(state.T, 0.015859999999999999);
  const amrex::Real cse155 = 508.69999999999999*cse154;
  const amrex::Real cse156 = cse155 + 1;
  const amrex::Real cse157 = cse149*cse156;
  const amrex::Real cse158 = -cse152;
  const amrex::Real cse159 = 0.5*cse5;
  const amrex::Real cse160 = cse5*std::log(cse150);
  const amrex::Real cse161 = 1.6774414051711492e-5*cse149;
  const amrex::Real cse162 = cse156*cse161 + 1;
  const amrex::Real cse163 = cse14*d2g;
  const amrex::Real cse164 = 1.2250000000000001e-11*cse163*state.xn[1]/std::pow(cse162, 2);
  const amrex::Real cse165 = cse164*(-1.1557571281629217e-5*cse157*cse5 - cse161*(8.0679819999999989*cse153*std::pow(state.T, -0.98414000000000001) + cse155*(cse158*cse159 - 1.102e-5*cse160)));
  const amrex::Real cse166 = 1.0/cse120;
  const amrex::Real cse167 = 9.0566037735849048e-14*cse118;
  const amrex::Real cse168 = cse166*cse167;
  const amrex::Real cse169 = cse121*cse167;
  const amrex::Real cse170 = cse119*cse169;
  const amrex::Real cse171 = -cse168 + cse170;
  const amrex::Real cse172 = cse166*state.xn[16];
  const amrex::Real cse173 = 3.7810000000000002e-10*std::pow(state.T, -1.1899999999999999)*state.xn[11];
  const amrex::Real cse174 = cse110*cse115*cse121*state.xn[11]*state.xn[16];
  const amrex::Real cse175 = cse172*cse173 + 1.1264150943396225e-13*cse174*std::pow(state.T, -0.68999999999999995)*state.xn[3];
  const amrex::Real cse176 = cse139 + cse144 - cse145 - cse165 + cse171 + cse175;
  const amrex::Real cse177 = -cse103 + cse104 + 9.8669999999999991e-12*cse105*cse96*state.xn[3]*state.xn[8] + cse109 - cse118*cse122 - cse128 - cse176 + 2*cse4*cse87 - cse73 - cse74 - cse78 - cse88 + 2*cse91*state.xn[0]*state.xn[3] - 1.5566492609718528e-6*cse92 + cse94 - 5.5769999999999994e-13*cse98;
  const amrex::Real cse178 = 1.0/cse162;
  const amrex::Real cse179 = 1.2250000000000001e-11*cse178;
  const amrex::Real cse180 = d2g*state.xn[1];
  const amrex::Real cse181 = cse179*cse180;
  const amrex::Real cse182 = -cse181;
  const amrex::Real cse183 = 6.3999999999999996e-10*state.xn[4];
  const amrex::Real cse184 = crate*(1.5*cse15*state.xn[0] + 2.2999999999999998*cse15*state.xn[3]);
  const amrex::Real cse185 = ((cse48) ? (
   cse137
)
: (
   0.0
));
  const amrex::Real cse186 = cse185*state.xn[2];
  const amrex::Real cse187 = 2.2999999999999998*cse29 + 1.5*cse31;
  const amrex::Real cse188 = 1.5*cse15 - cse187;
  const amrex::Real cse189 = crate*state.xn[0];
  const amrex::Real cse190 = cse188*cse189;
  const amrex::Real cse191 = cse182 + cse183 + cse184 + cse186 + cse190;
  const amrex::Real cse192 = 2.5092930924433742e-24*cse64*cse68*state.T - cse67;
  const amrex::Real cse193 = 0.66666666666670005*cse65;
  const amrex::Real cse194 = cse193*cse63;
  const amrex::Real cse195 = cse177*cse194;
  const amrex::Real cse196 = -cse187;
  const amrex::Real cse197 = cse189*cse196;
  const amrex::Real cse198 = cse181 - cse197;
  const amrex::Real cse199 = cse14*((cse20) ? (
   0
)
: (
   -cse33
));
  const amrex::Real cse200 = cse43*(-cse199 - cse21);
  const amrex::Real cse201 = cse200*cse22;
  const amrex::Real cse202 = cse4*((cse48) ? (
   cse47*(cse199*cse35 + cse200*cse42 + cse201*cse6 + cse26)
)
: (
   0
));
  const amrex::Real cse203 = -cse59;
  const amrex::Real cse204 = ((cse48) ? (
   cse61*(cse199*cse55 + cse200*cse53 + cse201*cse54 + cse60)
)
: (
   0
));
  const amrex::Real cse205 = cse13*state.xn[0];
  const amrex::Real cse206 = 2*cse202 + cse203 + cse204*cse205;
  const amrex::Real cse207 = cse198 + cse206;
  const amrex::Real cse208 = 4.8813573300130879e-6*cse140*cse142;
  const amrex::Real cse209 = cse208*state.xn[2];
  const amrex::Real cse210 = cse163*cse179;
  const amrex::Real cse211 = cse209 + cse210;
  const amrex::Real cse212 = std::pow(state.T, -0.52000000000000002);
  const amrex::Real cse213 = 2.9935562710997168e-6*cse212;
  const amrex::Real cse214 = 1.3664075564999315e-27*cse64*cse68*state.T - cse67;
  const amrex::Real cse215 = std::pow(state.xn[2], 2);
  const amrex::Real cse216 = 1.0/cse215;
  const amrex::Real cse217 = cse216*cse50;
  const amrex::Real cse218 = cse117*cse217;
  const amrex::Real cse219 = 3.6226415094339619e-13*cse121;
  const amrex::Real cse220 = 1.811320754716981e-13*cse166;
  const amrex::Real cse221 = cse218*cse220;
  const amrex::Real cse222 = cse122*cse218;
  const amrex::Real cse223 = cse119*cse222;
  const amrex::Real cse224 = -cse221 + cse223;
  const amrex::Real cse225 = 1.05849862170431e-5*std::pow(state.T, -0.64000000000000001);
  const amrex::Real cse226 = cse225*state.xn[10];
  const amrex::Real cse227 = 2.2528301886792451e-13*std::pow(state.T, 0.31);
  const amrex::Real cse228 = cse115*cse121*cse216*cse227*state.xn[11]*state.xn[16]*state.xn[3];
  const amrex::Real cse229 = cse185*state.xn[0];
  const amrex::Real cse230 = cse164*(0.0085331444281056355*cse110*cse149*cse154*cse158 + 2.3115142563258434e-5*cse110*cse157) + cse208*state.xn[1] - cse229;
  const amrex::Real cse231 = cse226 + cse228 + cse230;
  const amrex::Real cse232 = 1.2599999999999999e-13*cse71;
  const amrex::Real cse233 = cse232*state.xn[6];
  const amrex::Real cse234 = 1.7599999999999999e-9*cse99;
  const amrex::Real cse235 = cse101*cse234;
  const amrex::Real cse236 = d2g*state.xn[0];
  const amrex::Real cse237 = -1.2000000000000001e-14*cse236;
  const amrex::Real cse238 = 2*cse21;
  const amrex::Real cse239 = cse238*cse35;
  const amrex::Real cse240 = cse13*cse28;
  const amrex::Real cse241 = 2*state.xn[0];
  const amrex::Real cse242 = cse241*cse28;
  const amrex::Real cse243 = cse12*cse240 + cse17*cse242;
  const amrex::Real cse244 = cse14*((cse20) ? (
   0
)
: (
   cse12*cse15 - cse243
));
  const amrex::Real cse245 = cse43*(-cse238 - cse244);
  const amrex::Real cse246 = cse22*cse245;
  const amrex::Real cse247 = cse4*((cse48) ? (
   cse47*(cse239 + cse244*cse35 + cse245*cse42 + cse246*cse6)
)
: (
   0
));
  const amrex::Real cse248 = cse57*state.xn[0];
  const amrex::Real cse249 = ((cse48) ? (
   cse46
)
: (
   0.0
));
  const amrex::Real cse250 = 4*state.xn[3];
  const amrex::Real cse251 = 3.2000000000000001e-9*cse166;
  const amrex::Real cse252 = cse238*cse55;
  const amrex::Real cse253 = ((cse48) ? (
   cse61*(cse244*cse55 + cse245*cse53 + cse246*cse54 + cse252)
)
: (
   0
));
  const amrex::Real cse254 = std::sqrt(2.0000000000000002e-15*ncol_H2 + 1);
  const amrex::Real cse255 = 0.96499999999999997/std::pow(1.4142135623730951e-10*ncol_H2/vdisp + 1, 2) + 0.035000000000000003*std::exp(-0.00084999999999999995*cse254)/cse254;
  const amrex::Real cse256 = std::exp(-4.1799999999999997*av);
  const amrex::Real cse257 = chi*cse255*cse256;
  const amrex::Real cse258 = cse96*std::pow(state.T, -1.3);
  const amrex::Real cse259 = 4.2899999999999994e-13*cse258;
  const amrex::Real cse260 = 5.0213189999997486e-24*cse64*cse68*state.T - cse67;
  const amrex::Real cse261 = cse112*cse116*state.xn[3];
  const amrex::Real cse262 = 1.6000000000000001e-9*cse166;
  const amrex::Real cse263 = cse262*state.xn[17];
  const amrex::Real cse264 = cse122*cse261;
  const amrex::Real cse265 = cse119*cse263 - cse119*cse264 + cse220*cse261;
  const amrex::Real cse266 = cse178*cse180;
  const amrex::Real cse267 = 2.4500000000000003e-11*cse266;
  const amrex::Real cse268 = 4.5999999999999996*cse29 + 3.0*cse31;
  const amrex::Real cse269 = 2.2999999999999998*cse15 - cse268;
  const amrex::Real cse270 = cse189*cse269;
  const amrex::Real cse271 = cse174*cse227;
  const amrex::Real cse272 = cse267 - cse270 - cse271;
  const amrex::Real cse273 = cse100*cse234*state.xn[3];
  const amrex::Real cse274 = 6.3999999999999996e-10*state.xn[0];
  const amrex::Real cse275 = -cse274;
  const amrex::Real cse276 = cse14*((cse20) ? (
   0
)
: (
   -cse243
));
  const amrex::Real cse277 = cse43*(-cse238 - cse276);
  const amrex::Real cse278 = cse22*cse277;
  const amrex::Real cse279 = cse4*((cse48) ? (
   cse47*(cse239 + cse276*cse35 + cse277*cse42 + cse278*cse6)
)
: (
   0
));
  const amrex::Real cse280 = ((cse48) ? (
   cse61*(cse252 + cse276*cse55 + cse277*cse53 + cse278*cse54)
)
: (
   0
));
  const amrex::Real cse281 = 5.0199525924432485e-24*cse64*cse68*state.T - cse67;
  const amrex::Real cse282 = -cse268;
  const amrex::Real cse283 = cse189*cse282;
  const amrex::Real cse284 = cse267 - cse283;
  const amrex::Real cse285 = 9.9697094999995006e-24*cse64*cse68*state.T - cse67;
  const amrex::Real cse286 = 9.9683430924430005e-24*cse64*cse68*state.T - cse67;
  const amrex::Real cse287 = cse232*state.xn[3];
  const amrex::Real cse288 = 3.5394975000000002e-8*cse75*cse76 + 8.37e-10;
  const amrex::Real cse289 = cse288*state.xn[15];
  const amrex::Real cse290 = cse287 + cse289;
  const amrex::Real cse291 = 7.9513696852843427e-10*std::pow(state.T, -0.33900000000000002);
  const amrex::Real cse292 = cse125*cse291;
  const amrex::Real cse293 = 2.9917094999998509e-23*cse64*cse68*state.T - cse67;
  const amrex::Real cse294 = 2.991572859244201e-23*cse64*cse68*state.T - cse67;
  const amrex::Real cse295 = 6.5199999999999996*crate;
  const amrex::Real cse296 = 6.976853999999652e-23*cse64*cse68*state.T - cse67;
  const amrex::Real cse297 = 7.2277833092439893e-23*cse64*cse68*state.T - cse67;
  const amrex::Real cse298 = cse225*state.xn[2];
  const amrex::Real cse299 = cse198 + cse298;
  const amrex::Real cse300 = 7.0000000000000004e-11*state.xn[14];
  const amrex::Real cse301 = 3.5000000000000002e-11*state.xn[15];
  const amrex::Real cse302 = cse107*state.xn[1];
  const amrex::Real cse303 = 1.99e-9*std::pow(state.T, -0.19);
  const amrex::Real cse304 = cse172*cse303;
  const amrex::Real cse305 = 3.9851444999998012e-23*cse64*cse68*state.T - cse67;
  const amrex::Real cse306 = 6.9955604999996505e-23*cse64*cse68*state.T - cse67;
  const amrex::Real cse307 = 6.9954238592440006e-23*cse64*cse68*state.T - cse67;
  const amrex::Real cse308 = 7.0000000000000004e-11*state.xn[11];
  const amrex::Real cse309 = 9.0999999999999996e-10*chi*std::exp(-2.1200000000000001*av);
  const amrex::Real cse310 = cse0*state.xn[0];
  const amrex::Real cse311 = -cse310;
  const amrex::Real cse312 = 3.242775449999838e-23*cse64*cse68*state.T - cse67;
  const amrex::Real cse313 = cse288*state.xn[6];
  const amrex::Real cse314 = 4.2362104499997889e-23*cse64*cse68*state.T - cse67;
  const amrex::Real cse315 = cse124*cse291*state.xn[7];
  const amrex::Real cse316 = 3.7999999999999998e-10*chi*std::exp(-2.6600000000000001*av) + 3.5000000000000002e-11*state.xn[11];
  const amrex::Real cse317 = cse315 + cse316;
  const amrex::Real cse318 = 3*cse21;
  const amrex::Real cse319 = cse14*((cse20) ? (
   0
)
: (
   -3*cse30 - 3*cse32
));
  const amrex::Real cse320 = cse43*(-cse318 - cse319);
  const amrex::Real cse321 = cse22*cse320;
  const amrex::Real cse322 = cse4*((cse48) ? (
   cse47*(cse318*cse35 + cse319*cse35 + cse320*cse42 + cse321*cse6)
)
: (
   0
));
  const amrex::Real cse323 = ((cse48) ? (
   cse61*(cse318*cse55 + cse319*cse55 + cse320*cse53 + cse321*cse54)
)
: (
   0
));
  const amrex::Real cse324 = 7.5306120924431232e-24*cse64*cse68*state.T - cse67;
  const amrex::Real cse325 = cse303*state.xn[11];
  const amrex::Real cse326 = cse166*cse325;
  const amrex::Real cse327 = -6.8999999999999995*cse29 - 4.5*cse31;
  const amrex::Real cse328 = cse189*cse327;
  const amrex::Real cse329 = 3.6750000000000002e-11*cse266;
  const amrex::Real cse330 = cse326 - cse328 + cse329;
  const amrex::Real cse331 = cse3*state.xn[0];
  const amrex::Real cse332 = cse262*state.xn[3];
  const amrex::Real cse333 = cse119*cse332;
  const amrex::Real cse334 = 3.9850078592441519e-23*cse64*cse68*state.T - cse67;
  const amrex::Real cse335 = cse2*state.xn[17];
  const amrex::Real cse336 = cse109 - cse88 + cse94;
  const amrex::Real cse337 = cse139 + cse144 - cse145 - cse165 - cse336 - cse73;
  const amrex::Real cse338 = cse194*cse337;
  const amrex::Real cse339 = -cse106*cse93;
  const amrex::Real cse340 = cse339*state.xn[11];
  const amrex::Real cse341 = cse182 + cse197;
  const amrex::Real cse342 = -cse267;
  const amrex::Real cse343 = cse339*state.xn[1];
  const amrex::Real cse344 = cse2*state.xn[0];
  const amrex::Real cse345 = crate*cse13;
  const amrex::Real cse346 = cse188*cse345;
  const amrex::Real cse347 = -cse346;
  const amrex::Real cse348 = std::pow(cse5, 0.00231);
  const amrex::Real cse349 = std::exp(-8110.0*cse5);
  const amrex::Real cse350 = std::exp(-26.600000000000001*cse5);
  const amrex::Real cse351 = std::exp(-7.6200000000000001*cse5);
  const amrex::Real cse352 = std::exp(-1.3799999999999999*cse5);
  const amrex::Real cse353 = 0.00015100000000000001*cse349 + 1.31e-7*cse350 + 3.4e-8*cse351 + 6.9699999999999997e-9*cse352;
  const amrex::Real cse354 = std::pow(state.T, -1.5);
  const amrex::Real cse355 = (-2.4342628390636543e-12*cse348*cse5 - 1.5*cse353*std::pow(state.T, -2.5) + cse354*(1.2246100000000002*cse349*cse70 + 3.4846000000000001e-6*cse350*cse70 + 2.5908000000000001e-7*cse351*cse70 + 9.6185999999999982e-9*cse352*cse70))*state.xn[16]*state.xn[7];
  const amrex::Real cse356 = -cse355;
  const amrex::Real cse357 = 6.7636511339243933e-7*cse92;
  const amrex::Real cse358 = cse15*state.xn[2];
  const amrex::Real cse359 = cse358 < 1.0e-50;
  const amrex::Real cse360 = ((cse359) ? (
   10000000000.0
)
: (
   cse114
));
  const amrex::Real cse361 = cse360 + 1;
  const amrex::Real cse362 = 1.0/cse361;
  const amrex::Real cse363 = cse360*cse362;
  const amrex::Real cse364 = cse363*state.xn[16];
  const amrex::Real cse365 = cse173*cse364;
  const amrex::Real cse366 = ((cse359) ? (
   0
)
: (
   5.6603773584905654e-5*cse111*state.xn[3]
));
  const amrex::Real cse367 = cse325*state.xn[16];
  const amrex::Real cse368 = cse362*cse367;
  const amrex::Real cse369 = cse366*cse368;
  const amrex::Real cse370 = -cse369;
  const amrex::Real cse371 = cse360*cse367/std::pow(cse361, 2);
  const amrex::Real cse372 = cse366*cse371;
  const amrex::Real cse373 = std::exp(-15230.0*cse5);
  const amrex::Real cse374 = std::exp(-73.799999999999997*cse5);
  const amrex::Real cse375 = std::exp(-12.17*cse5);
  const amrex::Real cse376 = 1.6339999999999999e-6*cse373 + 9.7930000000000004e-9*cse374 + 6.3460000000000002e-9*cse375;
  const amrex::Real cse377 = 0.00071740367932898903*cse50 + 1;
  const amrex::Real cse378 = std::exp(-49550.0*cse5);
  const amrex::Real cse379 = 0.15970000000000001*cse378;
  const amrex::Real cse380 = cse379 + 1.7848999999999999;
  const amrex::Real cse381 = 12.244387999457523*cse50 + 1;
  const amrex::Real cse382 = 0.21509999999999996 - cse379;
  const amrex::Real cse383 = std::pow(cse377, -cse380)*std::pow(cse381, -cse382);
  const amrex::Real cse384 = 7913.1350000000002*cse378*cse70;
  const amrex::Real cse385 = 2.4460185352936309e-10*cse383*cse89;
  const amrex::Real cse386 = (-3.0/2.0*cse376/std::pow(state.T, 5.0/2.0) - 1.2230092676468155e-10*cse383*cse90 + cse385*(-cse384*std::log(cse377) - 0.00035870183966449452*cse380*cse89/cse377) + cse385*(cse384*std::log(cse381) - 6.1221939997287613*cse382*cse89/cse381) + cse90*(0.024885819999999999*cse373*cse70 + 7.2272340000000005e-7*cse374*cse70 + 7.7230820000000007e-8*cse375*cse70))*state.xn[2]*state.xn[8];
  const amrex::Real cse387 = 9.0489001802816564e-11*std::pow(state.T, -1.6200000000000001)*state.xn[13]*state.xn[2];
  const amrex::Real cse388 = cse151*cse8;
  const amrex::Real cse389 = 0.044330000000000001*cse388 - 0.28520000000000001;
  const amrex::Real cse390 = cse388*cse389 - 1.6759999999999999;
  const amrex::Real cse391 = cse388*cse390 + 11.19;
  const amrex::Real cse392 = cse391*state.xn[6];
  const amrex::Real cse393 = 4.9999999999999997e-12*cse354*cse392*state.xn[2];
  const amrex::Real cse394 = cse105*cse97;
  const amrex::Real cse395 = 7.5899999999999998e-12*cse394;
  const amrex::Real cse396 = 9.9999999999999994e-12*std::pow(state.T, -0.5);
  const amrex::Real cse397 = cse396*state.xn[2];
  const amrex::Real cse398 = cse397*(cse388*(0.044330000000000001*cse151*cse80 + cse389*cse82) + cse390*cse82)*state.xn[6];
  const amrex::Real cse399 = std::pow(cse148, 1.8740000000000001);
  const amrex::Real cse400 = 7.538e-5*cse151 + 0.89639999999999997;
  const amrex::Real cse401 = std::pow(cse150, -cse400);
  const amrex::Real cse402 = cse401*std::pow(state.T, 1.635e-6);
  const amrex::Real cse403 = 43750.0*cse402;
  const amrex::Real cse404 = cse403 + 1;
  const amrex::Real cse405 = cse399*cse404;
  const amrex::Real cse406 = -cse400;
  const amrex::Real cse407 = 1.534817233071468e-7*cse399;
  const amrex::Real cse408 = cse404*cse407 + 1;
  const amrex::Real cse409 = 2.1659999999999999e-12*cse163*state.xn[13]/std::pow(cse408, 2);
  const amrex::Real cse410 = cse409*(-1.4381237473879656e-7*cse405*cse5 - cse407*(0.071531250000000005*cse401*std::pow(state.T, -0.99999836499999994) + cse403*(cse159*cse406 - 7.538e-5*cse160)));
  const amrex::Real cse411 = std::pow(cse148, 1.512);
  const amrex::Real cse412 = 5.4939999999999998e-7*cse151 + 0.49559999999999998;
  const amrex::Real cse413 = std::pow(cse150, -cse412);
  const amrex::Real cse414 = cse413*std::pow(state.T, 3.9029999999999999e-7);
  const amrex::Real cse415 = 5115.0*cse414;
  const amrex::Real cse416 = cse415 + 1;
  const amrex::Real cse417 = cse411*cse416;
  const amrex::Real cse418 = -cse412;
  const amrex::Real cse419 = 7.104740267838494e-7*cse411;
  const amrex::Real cse420 = cse416*cse419 + 1;
  const amrex::Real cse421 = 5.5720000000000001e-12*cse163*state.xn[6]/std::pow(cse420, 2);
  const amrex::Real cse422 = cse421*(-5.3711836424859012e-7*cse417*cse5 - cse419*(0.0019963845000000001*cse413*std::pow(state.T, -0.99999960970000001) + cse415*(cse159*cse418 - 5.4939999999999998e-7*cse160)));
  const amrex::Real cse423 = std::pow(cse148, 1.1279999999999999);
  const amrex::Real cse424 = 0.00013329999999999999*cse151 + 0.81200000000000006;
  const amrex::Real cse425 = std::pow(cse150, -cse424);
  const amrex::Real cse426 = cse425*std::pow(state.T, 0.04845);
  const amrex::Real cse427 = 433.10000000000002*cse426;
  const amrex::Real cse428 = cse427 + 1;
  const amrex::Real cse429 = cse423*cse428;
  const amrex::Real cse430 = -cse424;
  const amrex::Real cse431 = 0.011078790689599145*cse423;
  const amrex::Real cse432 = cse428*cse431 + 1;
  const amrex::Real cse433 = 4.5579999999999993e-11*cse163*state.xn[8]/std::pow(cse432, 2);
  const amrex::Real cse434 = cse433*(-0.0062484379489339175*cse429*cse5 - cse431*(20.983695000000001*cse425*std::pow(state.T, -0.95155000000000001) + cse427*(cse159*cse430 - 0.00013329999999999999*cse160)));
  const amrex::Real cse435 = cse123 + cse169 + cse176 + cse356 + cse357 + cse365 + cse370 + cse372 - cse386 + cse387 + cse393 - cse395 - cse398 - cse410 - cse422 - cse434 + 4.2899999999999994e-13*cse98;
  const amrex::Real cse436 = 1.0/cse420;
  const amrex::Real cse437 = 5.5720000000000001e-12*cse436;
  const amrex::Real cse438 = d2g*state.xn[6];
  const amrex::Real cse439 = cse437*cse438;
  const amrex::Real cse440 = 1.0/cse432;
  const amrex::Real cse441 = 4.5579999999999993e-11*cse440;
  const amrex::Real cse442 = d2g*state.xn[8];
  const amrex::Real cse443 = crate*cse29;
  const amrex::Real cse444 = cse443*state.xn[7];
  const amrex::Real cse445 = cse441*cse442 + 1040.0*cse444;
  const amrex::Real cse446 = 1.0/cse408;
  const amrex::Real cse447 = 2.1659999999999999e-12*cse446;
  const amrex::Real cse448 = d2g*state.xn[13];
  const amrex::Real cse449 = cse443*state.xn[12];
  const amrex::Real cse450 = cse447*cse448 + 16800.0*cse449;
  const amrex::Real cse451 = cse439 + cse445 + cse450;
  const amrex::Real cse452 = cse196*cse345;
  const amrex::Real cse453 = cse451 - cse452;
  const amrex::Real cse454 = cse198 + cse453;
  const amrex::Real cse455 = cse433*(4.7982242476653898*cse110*cse423*cse426*cse430 + 0.012496875897867835*cse110*cse429);
  const amrex::Real cse456 = 1.3007021411393064e-6*cse212;
  const amrex::Real cse457 = cse456*state.xn[16];
  const amrex::Real cse458 = cse376*cse90 + cse385;
  const amrex::Real cse459 = -cse458;
  const amrex::Real cse460 = cse392*cse396 + cse421*(0.0036340746469993896*cse110*cse411*cse414*cse418 + 1.0742367284971802e-6*cse110*cse417);
  const amrex::Real cse461 = 1.4595000290776864e-10*std::pow(state.T, -0.62);
  const amrex::Real cse462 = cse409*(0.0067148253946876721*cse110*cse399*cse402*cse406 + 2.8762474947759312e-7*cse110*cse405) + cse461*state.xn[13];
  const amrex::Real cse463 = ((cse359) ? (
   0
)
: (
   -0.00011320754716981131*cse217*state.xn[3]
));
  const amrex::Real cse464 = cse368*cse463;
  const amrex::Real cse465 = cse371*cse463;
  const amrex::Real cse466 = cse464 - cse465;
  const amrex::Real cse467 = cse224 + cse466;
  const amrex::Real cse468 = cse436*cse438;
  const amrex::Real cse469 = 1.1144e-11*cse468;
  const amrex::Real cse470 = 2*cse184;
  const amrex::Real cse471 = cse269*cse345;
  const amrex::Real cse472 = crate*cse15;
  const amrex::Real cse473 = 16800.0*cse472;
  const amrex::Real cse474 = cse473*state.xn[12];
  const amrex::Real cse475 = cse446*cse448;
  const amrex::Real cse476 = 4.3319999999999997e-12*cse475;
  const amrex::Real cse477 = 33600.0*cse449;
  const amrex::Real cse478 = cse476 + cse477;
  const amrex::Real cse479 = -cse474 + cse478;
  const amrex::Real cse480 = 3.2999999999999996e-13*cse258;
  const amrex::Real cse481 = cse480*state.xn[8];
  const amrex::Real cse482 = 1040.0*cse472;
  const amrex::Real cse483 = -cse482*state.xn[7];
  const amrex::Real cse484 = cse440*cse442;
  const amrex::Real cse485 = 2080.0*cse444 + 9.1159999999999986e-11*cse484;
  const amrex::Real cse486 = cse481 + cse483 + cse485;
  const amrex::Real cse487 = ((cse359) ? (
   0
)
: (
   cse113
));
  const amrex::Real cse488 = cse368*cse487;
  const amrex::Real cse489 = cse371*cse487;
  const amrex::Real cse490 = cse265 + cse488 - cse489;
  const amrex::Real cse491 = cse263 - cse264;
  const amrex::Real cse492 = cse282*cse345;
  const amrex::Real cse493 = 1.1000000000000001*crate;
  const amrex::Real cse494 = cse194*cse435;
  const amrex::Real cse495 = cse163*cse437 + cse391*cse397;
  const amrex::Real cse496 = 1.0537934368240927e-9*cse348 + cse353*cse354;
  const amrex::Real cse497 = -cse496;
  const amrex::Real cse498 = cse497*state.xn[16];
  const amrex::Real cse499 = 2.808e-22*ncol_H2;
  const amrex::Real cse500 = 3.4999999999999998e-10*chi*std::exp(-3.7599999999999998*av)*std::exp(-cse499)*std::exp(-1.6000000000000001e-17*ncol_C)/(cse499 + 1) + 3.8500000000000001*crate + cse482*state.xn[3];
  const amrex::Real cse501 = cse480*state.xn[3];
  const amrex::Real cse502 = cse163*cse441;
  const amrex::Real cse503 = cse303*cse364;
  const amrex::Real cse504 = cse304 + cse503;
  const amrex::Real cse505 = 4.4999999999999998e-9*chi*std::exp(-2.6099999999999999*av) + cse473*state.xn[3];
  const amrex::Real cse506 = cse163*cse447 + cse461*state.xn[2];
  const amrex::Real cse507 = 1.6715999999999999e-11*cse468;
  const amrex::Real cse508 = cse325*cse363;
  const amrex::Real cse509 = cse327*cse345;
  const amrex::Real cse510 = -cse509;
  const amrex::Real cse511 = 3120.0*cse444 + 1.3673999999999998e-10*cse484;
  const amrex::Real cse512 = 50400.0*cse449 + 6.4979999999999992e-12*cse475;
  const amrex::Real cse513 = cse456*state.xn[2] - cse497*state.xn[7];
  const amrex::Real cse514 = cse332 + cse333;
  const amrex::Real cse515 = 3.0000000000000002e-15*cse236;
  const amrex::Real cse516 = cse57*state.xn[3];
  const amrex::Real cse517 = state.xn[0]*state.xn[3];
  const amrex::Real cse518 = cse517*cse62;
  const amrex::Real cse519 = cse4*cse49;
  const amrex::Real cse520 = -cse74;
  const amrex::Real cse521 = cse4*cse87;
  const amrex::Real cse522 = cse517*cse91;
  const amrex::Real cse523 = cse395 - 4.2899999999999994e-13*cse95*cse96*state.xn[3]*state.xn[8];
  const amrex::Real cse524 = cse168 - cse170;
  const amrex::Real cse525 = -cse103 + cse104;
  const amrex::Real cse526 = cse175 + cse356 + cse365 + cse370 + cse372 + cse525;
  const amrex::Real cse527 = cse169 - cse520 - cse521 - cse522 - cse523 - cse524 - cse526 - cse72 - 2.3622303960273254e-7*cse92;
  const amrex::Real cse528 = cse194*cse527;
  const amrex::Real cse529 = cse204*cse517;
  const amrex::Real cse530 = cse202 + cse452 + cse529 - 3.0000000000000002e-15*d2g*state.xn[0];
  const amrex::Real cse531 = 4.5427507615910105e-7*cse212;
  const amrex::Real cse532 = cse221 - cse222 - cse223;
  const amrex::Real cse533 = cse228 + cse466;
  const amrex::Real cse534 = 5.6999999999999997e-11*cse257;
  const amrex::Real cse535 = cse253*cse517;
  const amrex::Real cse536 = 7.2000000000000002e-15*state.xn[6];
  const amrex::Real cse537 = cse233 + cse536;
  const amrex::Real cse538 = cse470 + cse471;
  const amrex::Real cse539 = -cse488 + cse489;
  const amrex::Real cse540 = cse235 + cse271 + cse539;
  const amrex::Real cse541 = cse265 + cse491;
  const amrex::Real cse542 = cse280*cse517;
  const amrex::Real cse543 = cse275 + cse492;
  const amrex::Real cse544 = 7.2000000000000002e-15*state.xn[3];
  const amrex::Real cse545 = cse287 + cse544;
  const amrex::Real cse546 = cse496*state.xn[16];
  const amrex::Real cse547 = 1.6999999999999999e-9*state.xn[16];
  const amrex::Real cse548 = cse496*state.xn[7];
  const amrex::Real cse549 = 9.0000000000000011e-15*cse236;
  const amrex::Real cse550 = cse323*cse517;
  const amrex::Real cse551 = 1.6999999999999999e-9*state.xn[9];
  const amrex::Real cse552 = cse326 + cse508 + cse551;
  const amrex::Real cse553 = -cse194*cse525;
  const amrex::Real cse554 = -cse78;
  const amrex::Real cse555 = -cse393 + cse398 + cse422 + cse554 + cse72;
  const amrex::Real cse556 = cse194*cse555;
  const amrex::Real cse557 = cse469 + cse537;
  const amrex::Real cse558 = 1.6000000000000001e-9*state.xn[9];
  const amrex::Real cse559 = cse495 + cse558;
  const amrex::Real cse560 = 1.6000000000000001e-9*state.xn[6];
  const amrex::Real cse561 = -cse555;
  const amrex::Real cse562 = cse194*cse561;
  const amrex::Real cse563 = -cse288;
  const amrex::Real cse564 = cse386 + cse434;
  const amrex::Real cse565 = cse127 + cse356 + 2.2769999999999997e-12*cse394 + cse564 + cse74 - 1.287e-13*cse98;
  const amrex::Real cse566 = cse194*cse565;
  const amrex::Real cse567 = cse443*state.xn[9];
  const amrex::Real cse568 = 184.0*cse567;
  const amrex::Real cse569 = -cse568;
  const amrex::Real cse570 = cse445 + cse569;
  const amrex::Real cse571 = 184.0*cse472;
  const amrex::Real cse572 = cse571*state.xn[9];
  const amrex::Real cse573 = 9.8999999999999995e-14*cse258;
  const amrex::Real cse574 = 368.0*cse567;
  const amrex::Real cse575 = -cse574;
  const amrex::Real cse576 = cse485 + cse575;
  const amrex::Real cse577 = 2.4e-10*chi*fshield_co_interp2d(ncol_CO, ncol_H2)*std::exp(-3.8799999999999999*av) + cse571*state.xn[3];
  const amrex::Real cse578 = cse309 + cse310;
  const amrex::Real cse579 = 552.0*cse567;
  const amrex::Real cse580 = 1.1994963750000002e-8*cse77*state.xn[8];
  const amrex::Real cse581 = -cse580;
  const amrex::Real cse582 = -cse523 - cse564 - cse581;
  const amrex::Real cse583 = cse194*cse582;
  const amrex::Real cse584 = 2.3989927500000003e-8*cse75*cse76 + 5.673e-10;
  const amrex::Real cse585 = -cse584;
  const amrex::Real cse586 = cse585*state.xn[15];
  const amrex::Real cse587 = -cse585*state.xn[8];
  const amrex::Real cse588 = -cse128;
  const amrex::Real cse589 = cse194*cse588;
  const amrex::Real cse590 = cse295 + cse547;
  const amrex::Real cse591 = cse560 + cse577;
  const amrex::Real cse592 = cse308 + cse568;
  const amrex::Real cse593 = cse123 + cse581;
  const amrex::Real cse594 = cse194*cse593;
  const amrex::Real cse595 = -cse365 + cse369 - cse372;
  const amrex::Real cse596 = cse169 + cse336;
  const amrex::Real cse597 = -cse595 - cse596;
  const amrex::Real cse598 = cse194*cse597;
  const amrex::Real cse599 = -cse300;
  const amrex::Real cse600 = -cse387 + cse410;
  const amrex::Real cse601 = cse194*cse600;
  const amrex::Real cse602 = -cse600;
  const amrex::Real cse603 = cse194*cse602;
  const amrex::Real cse604 = cse355 + 5.3129999999999997e-12*cse394 + cse520 - 3.0029999999999999e-13*cse98;
  const amrex::Real cse605 = cse194*cse604;
  const amrex::Real cse606 = 2.3099999999999997e-13*cse258;
  const amrex::Real cse607 = cse127 + cse524 + cse580 + cse595 + cse78;
  const amrex::Real cse608 = cse194*cse607;
  const amrex::Real cse609 = cse357 + cse526;
  const amrex::Real cse610 = cse194*cse609;
  const amrex::Real cse611 = cse171 + cse554 + cse596;
  const amrex::Real cse612 = cse194*cse611;
  const amrex::Real cse613 = 2.1789607199999999e-11*cse186;
  const amrex::Real cse614 = std::pow(state.T, 0.14999999999999999);
  const amrex::Real cse615 = 0.00088240812266906664*std::pow(state.T, 0.65000000000000002) + 1;
  const amrex::Real cse616 = 1.0/cse615;
  const amrex::Real cse617 = cse614*cse616;
  const amrex::Real cse618 = cse616*state.xn[2];
  const amrex::Real cse619 = cse614*cse618;
  const amrex::Real cse620 = 4.4460389837719569e-9*cse619 + 626500000.0;
  const amrex::Real cse621 = 1.0/cse620;
  const amrex::Real cse622 = std::exp(-118400.0*cse5);
  const amrex::Real cse623 = cse621*cse622;
  const amrex::Real cse624 = 1.3338116951315871e-8*cse623;
  const amrex::Real cse625 = cse619*cse624 + 1;
  const amrex::Real cse626 = 1.0/cse625;
  const amrex::Real cse627 = cse623*cse626;
  const amrex::Real cse628 = 1.3659996292638161e-10*cse627;
  const amrex::Real cse629 = cse617*cse628;
  const amrex::Real cse630 = 2.25*std::pow(state.T, 0.14699999999999999) + 5.2199999999999998;
  const amrex::Real cse631 = cse150 + 50;
  const amrex::Real cse632 = 0.0043*std::pow(cse631, 0.69199999999999995) + 1;
  const amrex::Real cse633 = 0.049959999999999997*std::pow(cse631, 0.43099999999999999)*cse632 + 1;
  const amrex::Real cse634 = 1.0/cse633;
  const amrex::Real cse635 = state.xn[2] + 1.0e-50;
  const amrex::Real cse636 = cse147/cse635;
  const amrex::Real cse637 = 1.7*cse50;
  const amrex::Real cse638 = cse636*cse637 + 50;
  const amrex::Real cse639 = std::log(cse638);
  const amrex::Real cse640 = 0.45350000000000001 + 2.234/cse639;
  const amrex::Real cse641 = std::pow(state.T, cse640);
  const amrex::Real cse642 = 1.4419999999999999 - 0.050889999999999998*cse639;
  const amrex::Real cse643 = std::exp(cse639*cse642);
  const amrex::Real cse644 = 1.8998126555754159e-29*cse641*cse643;
  const amrex::Real cse645 = cse163*cse644;
  const amrex::Real cse646 = cse645*state.xn[2];
  const amrex::Real cse647 = cse358 + 0.070000000000000007;
  const amrex::Real cse648 = 1.0/cse647;
  const amrex::Real cse649 = std::sqrt(cse358*cse648);
  const amrex::Real cse650 = 26.399999999999999*cse649 + 6.5;
  const amrex::Real cse651 = state.T - 2.1544346900318838*std::pow(85973.158732959506*cse147 + 4.1397668473788896, 1.0/6.0);
  const amrex::Real cse652 = cse27*d2g;
  const amrex::Real cse653 = 3.2000000000000002e-32*cse50;
  const amrex::Real cse654 = cse652*cse653;
  const amrex::Real cse655 = cse651*cse654;
  const amrex::Real cse656 = psigd_coll_interp2d(cse14, state.T);
  const amrex::Real cse657 = 100.0*cse163;
  const amrex::Real cse658 = cse656*cse657;
  const amrex::Real cse659 = 0.001*state.T;
  const amrex::Real cse660 = cse659 + 1;
  const amrex::Real cse661 = std::pow(cse660, -2);
  const amrex::Real cse662 = 1.0/cse660;
  const amrex::Real cse663 = std::pow(10, -2.0230000000000001*cse661 - 3.6709999999999998*cse662 - 11.08);
  const amrex::Real cse664 = cse15*cse663;
  const amrex::Real cse665 = cse664*state.xn[3];
  const amrex::Real cse666 = std::pow(10, -2.3900000000000001*cse661 + 0.055500000000000001*cse662 - 11.06);
  const amrex::Real cse667 = cse15*cse666;
  const amrex::Real cse668 = cse667*state.xn[0];
  const amrex::Real cse669 = cse665 + cse668;
  const amrex::Real cse670 = 1.0/cse669;
  const amrex::Real cse671 = cse534 + 1.9999999999999999e-7;
  const amrex::Real cse672 = cse670*cse671;
  const amrex::Real cse673 = cse15*cse672 + 1;
  const amrex::Real cse674 = 1.0/cse673;
  const amrex::Real cse675 = 2.5634832e-11*cse674 + 6.4087080000000003e-13;
  const amrex::Real cse676 = cse76*state.xn[2];
  const amrex::Real cse677 = (1.0/100.0)*state.T;
  const amrex::Real cse678 = std::log(cse677);
  const amrex::Real cse679 = 0.0086999999999999994*cse678;
  const amrex::Real cse680 = std::pow(cse677, cse679 + 0.12809999999999999);
  const amrex::Real cse681 = 7.5799999999999997e-10*cse680*state.xn[0];
  const amrex::Real cse682 = state.T < 500;
  const amrex::Real cse683 = std::pow(state.T, 0.070000000000000007);
  const amrex::Real cse684 = 0.75*((cse682) ? (
   1.1000000000000001e-13*state.T + 5.3300000000000002e-10
)
: (
   3.74757785025e-10*cse683
)) + 0.25*((cse682) ? (
   3.3000000000000001e-13*state.T + 4.4299999999999997e-10
)
: (
   3.8899728635600003e-10*cse683
));
  const amrex::Real cse685 = 4.5299999999999998e-6*cse676 + cse681 + cse684*state.xn[3];
  const amrex::Real cse686 = cse685 + 2.3e-6;
  const amrex::Real cse687 = std::exp(-91.209999999999994*cse5);
  const amrex::Real cse688 = cse687/cse686;
  const amrex::Real cse689 = 2*cse688;
  const amrex::Real cse690 = cse685*cse689 + 1;
  const amrex::Real cse691 = 1.0/cse690;
  const amrex::Real cse692 = cse691*state.xn[8];
  const amrex::Real cse693 = cse685*cse688;
  const amrex::Real cse694 = 5.7927337833400002e-20*cse693;
  const amrex::Real cse695 = 6.7291434000000005e-12*cse674 + 3.2043540000000002e-13;
  const amrex::Real cse696 = cse8*std::log(cse14);
  const amrex::Real cse697 = cse696 < 2;
  const amrex::Real cse698 = cse696 < 4;
  const amrex::Real cse699 = cse696 < 7;
  const amrex::Real cse700 = cse696 < 10;
  const amrex::Real cse701 = ((cse697) ? (
   10.0
)
: ((cse698) ? (
   (3.0/2.0)*cse696 + 7
)
: ((cse699) ? (
   (4.0/3.0)*cse696 + 23.0/3.0
)
: ((cse700) ? (
   (1.0/3.0)*cse696 + 44.0/3.0
)
: (
   18.0
)))));
  const amrex::Real cse702 = std::pow(state.T, 1.0/4.0);
  const amrex::Real cse703 = 680.10000000000002/cse702;
  const amrex::Real cse704 = 1.75*cse702;
  const amrex::Real cse705 = cse703*state.xn[2] + cse704*state.xn[0] + state.xn[3];
  const amrex::Real cse706 = std::exp(-state.T);
  const amrex::Real cse707 = 1 - cse706;
  const amrex::Real cse708 = std::pow(cse707, 1000);
  const amrex::Real cse709 = l0_co_interp1d(state.T);
  const amrex::Real cse710 = 1.0/cse709;
  const amrex::Real cse711 = M_SQRT2;
  const amrex::Real cse712 = std::max(gradv, 5.7445531974402559e-18*cse50*cse711);
  const amrex::Real cse713 = 100000.0/cse712;
  const amrex::Real cse714 = ncol_H <= 0;
  const amrex::Real cse715 = cse15*ncol_H;
  const amrex::Real cse716 = std::max(1.0000000000000001e-30, cse715);
  const amrex::Real cse717 = 1756701454.4310031*ncol_CO;
  const amrex::Real cse718 = ((cse714) ? (
   cse713*state.xn[9]
)
: (
   cse717/std::sqrt(cse716)
));
  const amrex::Real cse719 = llte_co_interp2d(state.T, cse718);
  const amrex::Real cse720 = 1.0/cse719;
  const amrex::Real cse721 = nhalf_co_interp2d(state.T, cse718);
  const amrex::Real cse722 = cse709*cse720;
  const amrex::Real cse723 = -cse721*cse722 + 1;
  const amrex::Real cse724 = 1.0/cse721;
  const amrex::Real cse725 = cse705*cse724;
  const amrex::Real cse726 = alpha_co_interp2d(state.T, cse718);
  const amrex::Real cse727 = std::pow(cse725, cse726);
  const amrex::Real cse728 = cse710*cse727;
  const amrex::Real cse729 = cse723*cse728;
  const amrex::Real cse730 = cse705*cse720 + cse710 + cse729;
  const amrex::Real cse731 = 1.0/cse730;
  const amrex::Real cse732 = cse708*cse731;
  const amrex::Real cse733 = cse732*state.xn[9];
  const amrex::Real cse734 = 7.1777529600000012e-12*cse516;
  const amrex::Real cse735 = std::exp(-326.60000000000002*cse5);
  const amrex::Real cse736 = std::pow(state.T, 0.92600000000000005);
  const amrex::Real cse737 = cse736*state.xn[2];
  const amrex::Real cse738 = 0.16*cse678;
  const amrex::Real cse739 = std::pow(cse677, 0.755 - cse738);
  const amrex::Real cse740 = 4.34e-10*cse739;
  const amrex::Real cse741 = cse740*state.xn[0];
  const amrex::Real cse742 = 0.042999999999999997*cse678;
  const amrex::Real cse743 = std::pow(cse677, cse742 + 0.88900000000000001);
  const amrex::Real cse744 = 5.2499999999999998e-13*cse743;
  const amrex::Real cse745 = 0.52500000000000002*cse678;
  const amrex::Real cse746 = std::pow(cse677, cse745 + 1.198);
  const amrex::Real cse747 = 2.2500000000000003e-12*cse746;
  const amrex::Real cse748 = cse744 + cse747;
  const amrex::Real cse749 = cse748*state.xn[3];
  const amrex::Real cse750 = 1.08e-14*cse737 + cse741 + cse749;
  const amrex::Real cse751 = cse735*cse750;
  const amrex::Real cse752 = std::pow(state.T, -0.074999999999999997);
  const amrex::Real cse753 = cse752*state.xn[2];
  const amrex::Real cse754 = 0.0030000000000000001*cse678;
  const amrex::Real cse755 = std::pow(cse677, 0.41899999999999998 - cse754);
  const amrex::Real cse756 = cse755*state.xn[0];
  const amrex::Real cse757 = 3.5700000000000001e-10*cse756;
  const amrex::Real cse758 = 0.025000000000000001*cse678;
  const amrex::Real cse759 = std::pow(cse677, cse758 + 0.26400000000000001);
  const amrex::Real cse760 = 3.7250000000000001e-11*cse759;
  const amrex::Real cse761 = std::pow(cse677, cse742 + 0.29599999999999999);
  const amrex::Real cse762 = 1.0274999999999999e-10*cse761;
  const amrex::Real cse763 = cse760 + cse762;
  const amrex::Real cse764 = cse763*state.xn[3];
  const amrex::Real cse765 = 5.1199999999999999e-10*cse753 + cse757 + cse764;
  const amrex::Real cse766 = (1.0/5.0)*cse765;
  const amrex::Real cse767 = std::exp(-425.40000000000003*cse5);
  const amrex::Real cse768 = cse750*cse767;
  const amrex::Real cse769 = std::pow(state.T, -0.025999999999999999);
  const amrex::Real cse770 = cse769*state.xn[2];
  const amrex::Real cse771 = 0.0060000000000000001*cse678;
  const amrex::Real cse772 = std::pow(cse677, 0.36899999999999999 - cse771);
  const amrex::Real cse773 = cse772*state.xn[0];
  const amrex::Real cse774 = 3.1899999999999998e-10*cse773;
  const amrex::Real cse775 = 0.041000000000000002*cse678;
  const amrex::Real cse776 = std::pow(cse677, cse775 + 0.20300000000000001);
  const amrex::Real cse777 = 4.7499999999999998e-11*cse776;
  const amrex::Real cse778 = 0.058000000000000003*cse678;
  const amrex::Real cse779 = std::pow(cse677, cse778 + 0.23699999999999999);
  const amrex::Real cse780 = 1.6725000000000002e-10*cse779;
  const amrex::Real cse781 = cse777 + cse780;
  const amrex::Real cse782 = cse781*state.xn[3];
  const amrex::Real cse783 = 4.8599999999999998e-10*cse770 + cse774 + cse782;
  const amrex::Real cse784 = (1.0/15.0)*cse783;
  const amrex::Real cse785 = cse765 + 8.9099999999999997e-5;
  const amrex::Real cse786 = cse735*cse785;
  const amrex::Real cse787 = (1.0/5.0)*cse783;
  const amrex::Real cse788 = cse751*cse766 + cse768*cse784 + cse786*cse787;
  const amrex::Real cse789 = cse750 + 1.7499999999999998e-5;
  const amrex::Real cse790 = cse783 + 1.34e-10;
  const amrex::Real cse791 = std::exp(-98.800000000000011*cse5);
  const amrex::Real cse792 = cse790*cse791;
  const amrex::Real cse793 = (1.0/3.0)*cse750;
  const amrex::Real cse794 = (3.0/5.0)*cse765;
  const amrex::Real cse795 = std::exp(-227.80000000000001*cse5);
  const amrex::Real cse796 = cse790*cse795;
  const amrex::Real cse797 = cse789*cse795;
  const amrex::Real cse798 = cse735*cse789;
  const amrex::Real cse799 = cse787*cse798 + cse794*cse796 + cse794*cse797;
  const amrex::Real cse800 = cse785*cse789 + cse785*cse790 + cse788 + cse792*cse793 + cse799;
  const amrex::Real cse801 = 1.0/cse800;
  const amrex::Real cse802 = 2.3872025442750958e-19*cse801;
  const amrex::Real cse803 = 2.8023005140019997e-18*cse801;
  const amrex::Real cse804 = cse788*cse802 + cse799*cse803;
  const amrex::Real cse805 = state.T < 100.0;
  const amrex::Real cse806 = state.T < 6000.0;
  const amrex::Real cse807 = state.T < 10;
  const amrex::Real cse808 = std::log(cse659);
  const amrex::Real cse809 = cse8*cse808;
  const amrex::Real cse810 = std::pow(cse7, -8);
  const amrex::Real cse811 = std::pow(cse808, 8)*cse810;
  const amrex::Real cse812 = std::pow(cse7, -7);
  const amrex::Real cse813 = std::pow(cse808, 7)*cse812;
  const amrex::Real cse814 = std::pow(cse7, -6);
  const amrex::Real cse815 = std::pow(cse808, 6)*cse814;
  const amrex::Real cse816 = std::pow(cse7, -5);
  const amrex::Real cse817 = std::pow(cse808, 5)*cse816;
  const amrex::Real cse818 = std::pow(cse7, -4);
  const amrex::Real cse819 = std::pow(cse808, 4)*cse818;
  const amrex::Real cse820 = std::pow(cse7, -3);
  const amrex::Real cse821 = std::pow(cse808, 3);
  const amrex::Real cse822 = cse820*cse821;
  const amrex::Real cse823 = cse10*std::pow(cse808, 2);
  const amrex::Real cse824 = std::pow(10.0, 16.815729999999999*cse809 + 70.609154000000004*cse811 + 364.14445999999998*cse813 + 801.81246999999996*cse815 + 983.67575999999997*cse817 + 734.71650999999997*cse819 + 343.1918*cse822 + 96.743155000000002*cse823 - 21.928795999999998)*state.xn[2];
  const amrex::Real cse825 = state.xn[5] + state.xn[6];
  const amrex::Real cse826 = std::pow(10.0, 1.5714710999999999*cse809 + 0.32168730000000001*cse817 - 0.51002221000000003*cse819 - 0.23619984999999999*cse822 + 0.015391166*cse823 - 22.089523)*state.xn[1] + std::pow(10.0, 2.1892372*cse809 + 0.19191374999999999*cse817 - 0.16596184*cse819 + 0.29036281000000003*cse822 - 0.81520437999999995*cse823 - 23.689236999999999)*cse825 + std::pow(10.0, 2.0943374000000001*cse8*cse808 - 0.033638326000000003*cse817 - 0.14913216000000001*cse819 + 0.43693353000000001*cse820*cse821 - 0.77151435999999995*cse823 - 23.962112000000001)*state.xn[3];
  const amrex::Real cse827 = std::pow(10, 37.383713*cse809 + 3.8479610000000002*cse817 + 20.159831000000001*cse819 + 48.656103000000002*cse822 + 58.145166000000003*cse823 - 16.818342000000001)*state.xn[0] + cse826;
  const amrex::Real cse828 = std::pow(10.0, 1.6802758*cse809 - 6.3701156000000001*cse811 + 6.4380698000000001*cse813 + 8.9167182999999994*cse815 - 8.8077017000000009*cse817 - 4.7274035999999997*cse819 + 4.0406627000000004*cse822 + 0.93310621999999999*cse823 - 22.921188999999998)*state.xn[2];
  const amrex::Real cse829 = state.T < 1000.0;
  const amrex::Real cse830 = std::pow(10.0, 3.5692468000000002*cse8*cse808 - 4.2519023000000002*cse817 - 21.328264000000001*cse819 - 27.850082*cse822 - 11.33286*cse823 - 24.311209000000002)*state.xn[0] + cse826;
  const amrex::Real cse831 = std::pow(10.0, 4.6450521*cse809 + 1.5538288*cse817 - 5.5108049000000001*cse819 + 5.9369081000000001*cse822 - 3.7209846*cse823 - 24.311209000000002)*state.xn[0] + cse826;
  const amrex::Real cse832 = std::pow(10.0, -11.945856385319289*cse10 + 8.3228160852326614*cse8 + 28.69470131298657*cse816 - 56.79810800956944*cse818 + 34.150687525022562*cse820 - 24.311209000000002)*state.xn[0] + std::pow(10.0, -2.6171337683481011*cse10 + 3.9225864834863131*cse8 + 3.5440891133599637*cse816 - 1.7105157385968939*cse818 + 1.6702447513374001*cse820 - 23.689236999999999)*cse825 + std::pow(10.0, -2.4768712409536775*cse10 + 3.7525488682084647*cse8 - 0.62120210234156448*cse816 - 1.5370575959566981*cse818 + 2.5133588394664677*cse820 - 23.962112000000001)*state.xn[3] + std::pow(10.0, 0.049411830040524525*cse10 + 2.8156982240432273*cse8 + 5.9406293599919788*cse816 - 5.2566362076906961*cse818 - 1.3586848802337363*cse820 - 22.089523)*state.xn[1];
  const amrex::Real cse833 = ((cse682 && cse805 && cse806) ? (
   cse807 || cse824 + cse827 < 1.0e-100
)
: (
   ((cse805 && cse806) ? (
      cse807 || cse827 + cse828 < 1.0e-100
   )
   : (
      ((cse682 && cse806 && cse829) ? (
         cse807 || cse824 + cse830 < 1.0e-100
      )
      : (
         ((cse806 && cse829) ? (
            cse807 || cse828 + cse830 < 1.0e-100
         )
         : (
            ((cse682 && cse806) ? (
               cse807 || cse824 + cse831 < 1.0e-100
            )
            : (
               ((cse806) ? (
                  cse807 || cse828 + cse831 < 1.0e-100
               )
               : (
                  ((cse682) ? (
                     cse807 || std::pow(10.0, 310.58441786958315*cse10 + 30.129743459482278*cse8 + 7500.646099484944*cse810 + 21588.957206746247*cse812 + 26530.843406310876*cse814 + 18165.631967965237*cse816 + 7572.4886742758581*cse818 + 1974.131269262874*cse820 - 21.928795999999998)*state.xn[2] + cse832 < 1.0e-100
                  )
                  : (
                     cse807 || std::pow(10.0, 2.9956460707652877*cse10 + 3.0106500755647456*cse8 - 676.68255490510751*cse810 + 381.69251128040054*cse812 + 295.04162851880631*cse814 - 162.65264843551881*cse816 - 48.723840736518795*cse818 + 23.242975457496804*cse820 - 22.921188999999998)*state.xn[2] + cse832 < 1.0e-100
                  ))
               ))
            ))
         ))
      ))
   ))
));
  const amrex::Real cse834 = ((cse806) ? (
   cse5
)
: (
   0.00016666666666666666
));
  const amrex::Real cse835 = std::exp(-11700.0*cse834);
  const amrex::Real cse836 = std::exp(-5860.0*cse834);
  const amrex::Real cse837 = std::exp(-510.0*cse834);
  const amrex::Real cse838 = std::exp(-2197000.0*((cse806) ? (
   std::pow(state.T, -3)
)
: (
   4.6296296296296294e-12
)));
  const amrex::Real cse839 = cse838*((cse806) ? (
   std::pow(state.T, 3.7599999999999998)
)
: (
   160638152785302.53
));
  const amrex::Real cse840 = 6.0142468035272636e-8*((cse806) ? (
   std::pow(state.T, 2.1000000000000001)
)
: (
   85924621.118090734
)) + 1.0;
  const amrex::Real cse841 = 1.0/cse840;
  const amrex::Real cse842 = 1.6e-18*cse835 + 6.7e-19*cse836 + 3.0e-24*cse837 + 4.985670872372847e-33*cse839*cse841;
  const amrex::Real cse843 = std::log(0.001*((cse806) ? (
   state.T
)
: (
   6000.0
)));
  const amrex::Real cse844 = std::pow(cse843, 5);
  const amrex::Real cse845 = cse816*cse844;
  const amrex::Real cse846 = std::pow(cse843, 4);
  const amrex::Real cse847 = cse818*cse846;
  const amrex::Real cse848 = std::pow(cse843, 3);
  const amrex::Real cse849 = std::pow(cse843, 2);
  const amrex::Real cse850 = cse10*cse849;
  const amrex::Real cse851 = std::pow(10.0, 2.0943374000000001*cse8*cse843 + 0.43693353000000001*cse820*cse848 - 0.033638326000000003*cse845 - 0.14913216000000001*cse847 - 0.77151435999999995*cse850 - 23.962112000000001);
  const amrex::Real cse852 = cse851*state.xn[3];
  const amrex::Real cse853 = cse8*cse843;
  const amrex::Real cse854 = cse820*cse848;
  const amrex::Real cse855 = std::pow(10.0, 0.32168730000000001*cse845 - 0.51002221000000003*cse847 + 0.015391166*cse850 + 1.5714710999999999*cse853 - 0.23619984999999999*cse854 - 22.089523);
  const amrex::Real cse856 = cse855*state.xn[1];
  const amrex::Real cse857 = std::pow(10.0, 0.19191374999999999*cse845 - 0.16596184*cse847 - 0.81520437999999995*cse850 + 2.1892372*cse853 + 0.29036281000000003*cse854 - 23.689236999999999);
  const amrex::Real cse858 = cse825*cse857;
  const amrex::Real cse859 = std::pow(10, 3.8479610000000002*cse845 + 20.159831000000001*cse847 + 58.145166000000003*cse850 + 37.383713*cse853 + 48.656103000000002*cse854 - 16.818342000000001);
  const amrex::Real cse860 = ((cse806) ? (
   cse805
)
: (
   false
));
  const amrex::Real cse861 = std::pow(10.0, 3.5692468000000002*cse8*cse843 - 4.2519023000000002*cse845 - 21.328264000000001*cse847 - 11.33286*cse850 - 27.850082*cse854 - 24.311209000000002);
  const amrex::Real cse862 = ((cse806) ? (
   cse829
)
: (
   false
));
  const amrex::Real cse863 = std::pow(10.0, 1.5538288*cse845 - 5.5108049000000001*cse847 - 3.7209846*cse850 + 4.6450521*cse853 + 5.9369081000000001*cse854 - 24.311209000000002);
  const amrex::Real cse864 = ((cse860) ? (
   cse859
)
: ((cse862) ? (
   cse861
)
: (
   cse863
)));
  const amrex::Real cse865 = cse810*std::pow(cse843, 8);
  const amrex::Real cse866 = std::pow(cse843, 7);
  const amrex::Real cse867 = cse812*cse866;
  const amrex::Real cse868 = std::pow(cse843, 6);
  const amrex::Real cse869 = cse814*cse868;
  const amrex::Real cse870 = std::pow(10.0, 983.67575999999997*cse845 + 734.71650999999997*cse847 + 96.743155000000002*cse850 + 16.815729999999999*cse853 + 343.1918*cse854 + 70.609154000000004*cse865 + 364.14445999999998*cse867 + 801.81246999999996*cse869 - 21.928795999999998);
  const amrex::Real cse871 = std::pow(10.0, -8.8077017000000009*cse845 - 4.7274035999999997*cse847 + 0.93310621999999999*cse850 + 1.6802758*cse853 + 4.0406627000000004*cse854 - 6.3701156000000001*cse865 + 6.4380698000000001*cse867 + 8.9167182999999994*cse869 - 22.921188999999998);
  const amrex::Real cse872 = ((cse682) ? (
   cse870
)
: (
   cse871
));
  const amrex::Real cse873 = cse852 + cse856 + cse858 + cse864*state.xn[0] + cse872*state.xn[2];
  const amrex::Real cse874 = 1.0/cse873;
  const amrex::Real cse875 = cse842*cse874 + 1;
  const amrex::Real cse876 = 1.0/cse875;
  const amrex::Real cse877 = ((cse833) ? (
   0.0
)
: (
   cse842*cse876
));
  const amrex::Real cse878 = std::exp(-101.30000000000001*cse5);
  const amrex::Real cse879 = 0.045999999999999999*cse678;
  const amrex::Real cse880 = std::pow(cse677, cse879 + 0.22800000000000001);
  const amrex::Real cse881 = cse880*state.xn[0];
  const amrex::Real cse882 = 8.9000000000000003e-11*cse881;
  const amrex::Real cse883 = 0.037999999999999999*cse678;
  const amrex::Real cse884 = std::pow(cse677, cse883 + 0.16900000000000001);
  const amrex::Real cse885 = 5.1749999999999996e-11*cse884;
  const amrex::Real cse886 = 0.048000000000000001*cse678;
  const amrex::Real cse887 = std::pow(cse677, cse886 - 0.01);
  const amrex::Real cse888 = 2.15e-11*cse887;
  const amrex::Real cse889 = cse885 + cse888;
  const amrex::Real cse890 = cse889*state.xn[3];
  const amrex::Real cse891 = 0.0070527699999999999*cse151;
  const amrex::Real cse892 = cse891 - 0.11133800000000001;
  const amrex::Real cse893 = cse151*cse892 + 0.69763799999999998;
  const amrex::Real cse894 = cse151*cse893 - 1.3074300000000001;
  const amrex::Real cse895 = 0.093813800000000003*cse151;
  const amrex::Real cse896 = cse895 - 3.0328300000000001;
  const amrex::Real cse897 = cse151*cse896 + 36.180300000000003;
  const amrex::Real cse898 = cse151*cse897 - 187.47399999999999;
  const amrex::Real cse899 = std::exp(((cse829) ? (
   cse151*cse894 - 7.6973500000000001
)
: (
   cse151*cse898 + 350.60899999999998
)));
  const amrex::Real cse900 = 1.7258000000000001e-6*cse676;
  const amrex::Real cse901 = cse899*cse900;
  const amrex::Real cse902 = cse882 + cse890 + cse901;
  const amrex::Real cse903 = cse878*cse902;
  const amrex::Real cse904 = std::pow(cse677, cse879 + 0.23100000000000001);
  const amrex::Real cse905 = cse904*state.xn[0];
  const amrex::Real cse906 = 2.6400000000000002e-10*cse905;
  const amrex::Real cse907 = 0.031*cse678;
  const amrex::Real cse908 = std::pow(cse677, cse907 + 0.26300000000000001);
  const amrex::Real cse909 = 1.11e-10*cse908;
  const amrex::Real cse910 = 0.064000000000000001*cse678;
  const amrex::Real cse911 = std::pow(cse677, cse910 + 0.071999999999999995);
  const amrex::Real cse912 = 4.3749999999999998e-11*cse911;
  const amrex::Real cse913 = cse909 + cse912;
  const amrex::Real cse914 = cse913*state.xn[3];
  const amrex::Real cse915 = 0.00235272*cse151;
  const amrex::Real cse916 = cse915 - 0.041816600000000002;
  const amrex::Real cse917 = cse151*cse916 + 0.35826400000000003;
  const amrex::Real cse918 = cse151*cse917 - 0.57443;
  const amrex::Real cse919 = 0.097857299999999994*cse151;
  const amrex::Real cse920 = cse919 - 3.1926800000000002;
  const amrex::Real cse921 = cse151*cse920 + 38.504899999999999;
  const amrex::Real cse922 = cse151*cse921 - 202.19300000000001;
  const amrex::Real cse923 = std::exp(((cse829) ? (
   cse151*cse918 - 7.4386999999999999
)
: (
   cse151*cse922 + 386.18599999999998
)));
  const amrex::Real cse924 = cse900*cse923;
  const amrex::Real cse925 = cse906 + cse914 + cse924;
  const amrex::Real cse926 = (25.0/3.0)*cse925;
  const amrex::Real cse927 = std::exp(-62.460000000000001*cse5);
  const amrex::Real cse928 = cse925*cse927;
  const amrex::Real cse929 = 0.057000000000000002*cse678;
  const amrex::Real cse930 = std::pow(cse677, cse929 + 0.115);
  const amrex::Real cse931 = cse930*state.xn[0];
  const amrex::Real cse932 = 1.26e-10*cse931;
  const amrex::Real cse933 = 0.049000000000000002*cse678;
  const amrex::Real cse934 = std::pow(cse677, cse933 - 0.0040000000000000001);
  const amrex::Real cse935 = 5.3249999999999997e-11*cse934;
  const amrex::Real cse936 = 0.10199999999999999*cse678;
  const amrex::Real cse937 = std::pow(cse677, cse936 - 0.085000000000000006);
  const amrex::Real cse938 = 1.675e-11*cse937;
  const amrex::Real cse939 = cse935 + cse938;
  const amrex::Real cse940 = cse939*state.xn[3];
  const amrex::Real cse941 = 0.00065632500000000003*cse151;
  const amrex::Real cse942 = -cse941 - 0.015089200000000001;
  const amrex::Real cse943 = cse151*cse942 + 0.36118400000000001;
  const amrex::Real cse944 = cse151*cse943 - 0.77378199999999997;
  const amrex::Real cse945 = 0.10508000000000001*cse151;
  const amrex::Real cse946 = cse945 - 3.4762;
  const amrex::Real cse947 = cse151*cse946 + 42.594999999999999;
  const amrex::Real cse948 = cse151*cse947 - 227.91300000000001;
  const amrex::Real cse949 = std::exp(((cse829) ? (
   cse151*cse944 - 9.2514099999999999
)
: (
   cse151*cse948 + 444.60000000000002
)));
  const amrex::Real cse950 = cse676*cse949;
  const amrex::Real cse951 = 2.8763333333333333e-6*cse950;
  const amrex::Real cse952 = cse932 + cse940 + cse951;
  const amrex::Real cse953 = 5*cse952;
  const amrex::Real cse954 = 5*cse902;
  const amrex::Real cse955 = cse952 + 7.8800000000000004e-8;
  const amrex::Real cse956 = cse927*cse955;
  const amrex::Real cse957 = cse903*cse926 + cse928*cse953 + cse954*cse956;
  const amrex::Real cse958 = cse925 + 2.65e-7;
  const amrex::Real cse959 = cse902 + 1.81e-14;
  const amrex::Real cse960 = std::exp(-38.840000000000003*cse5);
  const amrex::Real cse961 = cse959*cse960;
  const amrex::Real cse962 = (5.0/3.0)*cse925;
  const amrex::Real cse963 = cse927*cse958;
  const amrex::Real cse964 = 3*cse952;
  const amrex::Real cse965 = std::exp(-23.619999999999997*cse5);
  const amrex::Real cse966 = cse959*cse965;
  const amrex::Real cse967 = cse958*cse965;
  const amrex::Real cse968 = cse954*cse963 + cse964*cse966 + cse964*cse967;
  const amrex::Real cse969 = cse955*cse958 + cse955*cse959 + cse957 + cse961*cse962 + cse968;
  const amrex::Real cse970 = 1.0/cse969;
  const amrex::Real cse971 = 1.4210469458259591e-21*cse970;
  const amrex::Real cse972 = 2.5697412351439998e-22*cse970;
  const amrex::Real cse973 = state.T < 1000000.0;
  const amrex::Real cse974 = ((cse973) ? (
   cse957*cse971 + cse968*cse972
)
: (
   0.0
));
  const amrex::Real cse975 = cse68*(1.7e-24*chi*cse14*cse146*cse630*cse634*d2g + 5.6999999999999997e-11*chi*cse255*cse256*cse675*state.xn[3] + 1.602177e-12*crate*cse650*state.xn[0] + 3.2043540000000001e-12*crate*cse701*state.xn[3] - cse138*cse629 + 3.0000000000000002e-15*cse14*cse695*d2g*state.xn[0] - 7.1777529600000012e-12*cse249*cse4 - cse613*state.xn[0] - cse646 - cse692*cse694 - cse705*cse733 - cse734*state.xn[0] - cse804*state.xn[11] - cse877*state.xn[3] - cse974*state.xn[7] - std::min(cse655, cse658));
  const amrex::Real cse976 = std::pow(cse873, -2);
  const amrex::Real cse977 = std::pow(cse875, -2);
  const amrex::Real cse978 = std::pow(cse842, 2)*cse976*cse977;
  const amrex::Real cse979 = 4.4500000000000001e-10*cse880;
  const amrex::Real cse980 = 8.9000000000000003e-11*cse880;
  const amrex::Real cse981 = cse676*cse923;
  const amrex::Real cse982 = cse878*(2.2000000000000003e-9*cse905 + (25.0/3.0)*cse914 + 1.4381666666666668e-5*cse981);
  const amrex::Real cse983 = 2.6400000000000002e-10*cse904;
  const amrex::Real cse984 = cse927*(6.3e-10*cse931 + 5*cse940 + 1.4381666666666666e-5*cse950);
  const amrex::Real cse985 = 1.26e-10*cse930;
  const amrex::Real cse986 = cse676*cse899;
  const amrex::Real cse987 = cse927*(4.4500000000000001e-10*cse881 + 5*cse890 + 8.6289999999999999e-6*cse986);
  const amrex::Real cse988 = 2.2000000000000003e-9*cse903*cse904 + 6.3e-10*cse928*cse930 + cse956*cse979 + cse980*cse982 + cse983*cse984 + cse985*cse987;
  const amrex::Real cse989 = 3.7800000000000004e-10*cse930;
  const amrex::Real cse990 = cse965*(3.7800000000000004e-10*cse931 + 3*cse940 + 8.6289999999999999e-6*cse950);
  const amrex::Real cse991 = cse963*cse979 + cse966*cse989 + cse967*cse989 + cse980*cse990 + cse983*cse987 + cse983*cse990;
  const amrex::Real cse992 = std::pow(cse969, -2);
  const amrex::Real cse993 = cse960*(4.4000000000000003e-10*cse905 + (5.0/3.0)*cse914 + 2.8763333333333337e-6*cse981);
  const amrex::Real cse994 = cse992*(-4.4000000000000003e-10*cse904*cse961 - cse955*cse980 - cse955*cse983 - cse958*cse985 - cse959*cse985 - cse980*cse993 - cse988 - cse991);
  const amrex::Real cse995 = 1.4210469458259591e-21*cse957;
  const amrex::Real cse996 = 2.5697412351439998e-22*cse968;
  const amrex::Real cse997 = 6.3800000000000002e-11*cse772;
  const amrex::Real cse998 = 3.5700000000000001e-10*cse755;
  const amrex::Real cse999 = cse735*(9.7199999999999998e-11*cse770 + 6.3800000000000002e-11*cse773 + (1.0/5.0)*cse782);
  const amrex::Real cse1000 = cse735*(1.024e-10*cse753 + 7.1400000000000007e-11*cse756 + (1.0/5.0)*cse764);
  const amrex::Real cse1001 = cse767*(3.2399999999999999e-11*cse770 + 2.1266666666666666e-11*cse773 + (1.0/15.0)*cse782);
  const amrex::Real cse1002 = cse1000*cse740 + cse1001*cse740 + 7.1400000000000007e-11*cse751*cse755 + 2.1266666666666666e-11*cse768*cse772 + cse786*cse997 + cse998*cse999;
  const amrex::Real cse1003 = 2.1419999999999999e-10*cse755;
  const amrex::Real cse1004 = 3.1899999999999998e-10*cse772;
  const amrex::Real cse1005 = cse795*(3.0719999999999998e-10*cse753 + 2.1419999999999999e-10*cse756 + (3.0/5.0)*cse764);
  const amrex::Real cse1006 = cse1003*cse796 + cse1003*cse797 + cse1004*cse1005 + cse1005*cse740 + cse740*cse999 + cse798*cse997;
  const amrex::Real cse1007 = std::pow(cse800, -2);
  const amrex::Real cse1008 = 1.4466666666666666e-10*cse739;
  const amrex::Real cse1009 = cse791*(cse1008*state.xn[0] + 3.6000000000000001e-15*cse737 + (1.0/3.0)*cse749);
  const amrex::Real cse1010 = cse1007*(-cse1002 - cse1004*cse1009 - cse1004*cse785 - cse1006 - cse1008*cse792 - cse740*cse785 - cse789*cse998 - cse790*cse998);
  const amrex::Real cse1011 = 2.3872025442750958e-19*cse788;
  const amrex::Real cse1012 = 2.8023005140019997e-18*cse799;
  const amrex::Real cse1013 = cse28*ncol_H*(((cse715 - 1.0000000000000001e-30 < 0) ? (
   0
)
: ((cse715 - 1.0000000000000001e-30 == 0) ? (
   1.0/2.0
)
: (
   1
))))/std::pow(cse716, 3.0/2.0);
  const amrex::Real cse1014 = cse1013*ncol_CO;
  const amrex::Real cse1015 = ((cse714) ? (
   0
)
: (
   878350727.21550155*cse1014
));
  const amrex::Real cse1016 = alpha_co_interp2d_partial_1(state.T, cse718);
  const amrex::Real cse1017 = std::log(cse725);
  const amrex::Real cse1018 = cse1016*cse1017;
  const amrex::Real cse1019 = cse705/std::pow(cse721, 2);
  const amrex::Real cse1020 = nhalf_co_interp2d_partial_1(state.T, cse718);
  const amrex::Real cse1021 = cse1015*cse1020;
  const amrex::Real cse1022 = cse726/cse705;
  const amrex::Real cse1023 = cse1022*cse721;
  const amrex::Real cse1024 = std::pow(cse719, -2);
  const amrex::Real cse1025 = llte_co_interp2d_partial_1(state.T, cse718);
  const amrex::Real cse1026 = -cse1015*cse1024*cse1025*cse705 + cse728*(cse1015*cse1024*cse1025*cse709*cse721 - cse1021*cse722);
  const amrex::Real cse1027 = cse705*state.xn[9];
  const amrex::Real cse1028 = cse1027*cse708/std::pow(cse730, 2);
  const amrex::Real cse1029 = cse688*cse692;
  const amrex::Real cse1030 = std::pow(cse673, -2);
  const amrex::Real cse1031 = cse28*cse672;
  const amrex::Real cse1032 = cse29*cse663;
  const amrex::Real cse1033 = cse31*cse666;
  const amrex::Real cse1034 = cse1032 + cse1033;
  const amrex::Real cse1035 = cse671/std::pow(cse669, 2);
  const amrex::Real cse1036 = cse1035*cse15;
  const amrex::Real cse1037 = cse1031 - cse1036*(cse1034 - cse667);
  const amrex::Real cse1038 = 1.5159999999999999e-9*cse680;
  const amrex::Real cse1039 = std::pow(cse686, -2);
  const amrex::Real cse1040 = cse1039*cse685*cse687;
  const amrex::Real cse1041 = cse694*state.xn[8]/std::pow(cse690, 2);
  const amrex::Real cse1042 = cse656*d2g;
  const amrex::Real cse1043 = psigd_coll_interp2d_partial_0(cse14, state.T);
  const amrex::Real cse1044 = cse655 - cse658;
  const amrex::Real cse1045 = (((cse1044 < 0) ? (
   0
)
: ((cse1044 == 0) ? (
   1.0/2.0
)
: (
   1
))));
  const amrex::Real cse1046 = cse1045*(100.0*cse1042 + cse1043*cse657);
  const amrex::Real cse1047 = cse15*cse8;
  const amrex::Real cse1048 = ((cse697) ? (
   0
)
: ((cse698) ? (
   (3.0/2.0)*cse1047
)
: ((cse699) ? (
   (4.0/3.0)*cse1047
)
: ((cse700) ? (
   (1.0/3.0)*cse1047
)
: (
   0
)))));
  const amrex::Real cse1049 = d2g*state.xn[2];
  const amrex::Real cse1050 = cse1049*cse644;
  const amrex::Real cse1051 = (((cse1044 > 0) ? (
   0
)
: ((cse1044 == 0) ? (
   1.0/2.0
)
: (
   1
))));
  const amrex::Real cse1052 = cse1051*cse651*cse653*d2g;
  const amrex::Real cse1053 = cse1052*(cse241 + cse250 + 2*state.xn[10] + 2*state.xn[14] + 2*state.xn[15] + 6*state.xn[16] + 2*state.xn[1] + 4*state.xn[4]);
  const amrex::Real cse1054 = 0.5*cse648;
  const amrex::Real cse1055 = cse28*state.xn[2];
  const amrex::Real cse1056 = std::pow(cse14, -3);
  const amrex::Real cse1057 = std::pow(cse647, -2);
  const amrex::Real cse1058 = -cse1054*cse1055 + 0.5*cse1056*cse1057*cse215;
  const amrex::Real cse1059 = -1.7e-24*chi*cse146*cse630*cse634*d2g - 3.2043540000000001e-12*crate*cse1048*state.xn[3] - 4.22974728e-11*crate*cse1058*cse110*cse14*cse647*cse649*state.xn[0] + cse1046 + cse1050 + cse1053 - 3.0000000000000002e-15*cse695*d2g*state.xn[0];
  const amrex::Real cse1060 = cse623*std::pow(state.T, -1.8500000000000001);
  const amrex::Real cse1061 = cse138*cse626;
  const amrex::Real cse1062 = std::pow(state.T, -0.84999999999999998);
  const amrex::Real cse1063 = std::pow(state.T, -0.19999999999999998);
  const amrex::Real cse1064 = std::pow(cse615, -2);
  const amrex::Real cse1065 = cse1062*cse618;
  const amrex::Real cse1066 = std::pow(cse620, -2);
  const amrex::Real cse1067 = cse1066*cse622*(2.5500935934394033e-12*cse1063*cse1064*state.xn[2] - 6.6690584756579352e-10*cse1065);
  const amrex::Real cse1068 = 1.3659996292638161e-10*cse617;
  const amrex::Real cse1069 = std::pow(cse633, -2);
  const amrex::Real cse1070 = cse147*std::pow(cse631, 0.12299999999999994);
  const amrex::Real cse1071 = cse147*std::pow(cse631, -0.56899999999999995)*cse632;
  const amrex::Real cse1072 = cse636*cse89;
  const amrex::Real cse1073 = 1.0/cse638;
  const amrex::Real cse1074 = cse1073*cse151/std::pow(cse639, 2);
  const amrex::Real cse1075 = cse1073*cse642;
  const amrex::Real cse1076 = cse1068*cse138*cse623/std::pow(cse625, 2);
  const amrex::Real cse1077 = cse693*cse70;
  const amrex::Real cse1078 = std::pow(state.T, -0.92999999999999994);
  const amrex::Real cse1079 = 100*cse5;
  const amrex::Real cse1080 = -2.2649999999999999e-6*cse5*cse676 + cse681*(cse1079*(8.7000000000000001e-5*cse678 + 0.001281) + cse5*cse679) + (0.75*((cse682) ? (
   1.1000000000000001e-13
)
: (
   2.6233044951750002e-11*cse1078
)) + 0.25*((cse682) ? (
   3.3000000000000001e-13
)
: (
   2.7229810044920004e-11*cse1078
)))*state.xn[3];
  const amrex::Real cse1081 = 5.7927337833400002e-20*cse1029;
  const amrex::Real cse1082 = -cse1080;
  const amrex::Real cse1083 = 5.7927337833400002e-20*cse1040*cse692;
  const amrex::Real cse1084 = 0.4375*state.xn[0]/std::pow(state.T, 3.0/4.0) - 170.02500000000001*state.xn[2]/std::pow(state.T, 5.0/4.0);
  const amrex::Real cse1085 = 2.0187430200000004e-26*cse236;
  const amrex::Real cse1086 = std::pow(cse660, -3);
  const amrex::Real cse1087 = cse1030*(-cse665*cse7*(0.0040460000000000001*cse1086 + 0.0036709999999999998*cse661) - cse668*cse7*(0.0047800000000000004*cse1086 - 5.5500000000000001e-5*cse661));
  const amrex::Real cse1088 = 1.461185424e-21*cse257*state.xn[3];
  const amrex::Real cse1089 = 2*cse1040;
  const amrex::Real cse1090 = l0_co_interp1d_partial_0(state.T);
  const amrex::Real cse1091 = std::pow(cse709, -2);
  const amrex::Real cse1092 = ((cse714) ? (
   -2.872276598720128e-13*cse711*cse89*(((5.7445531974402559e-18*cse50*cse711 - gradv < 0) ? (
      0
   )
   : ((5.7445531974402559e-18*cse50*cse711 - gradv == 0) ? (
      1.0/2.0
   )
   : (
      1
   ))))*state.xn[9]/std::pow(cse712, 2)
)
: (
   0
));
  const amrex::Real cse1093 = -cse1025*cse1092 - llte_co_interp2d_partial_0(state.T, cse718);
  const amrex::Real cse1094 = cse1020*cse1092 + nhalf_co_interp2d_partial_0(state.T, cse718);
  const amrex::Real cse1095 = ((cse806) ? (
   -cse70
)
: (
   0
));
  const amrex::Real cse1096 = -1.8719999999999998e-14*cse1095*cse835 - 3.9261999999999998e-15*cse1095*cse836 - 1.53e-21*cse1095*cse837 + 4.985670872372847e-33*cse838*cse841*((cse806) ? (
   3.7599999999999998*std::pow(state.T, 2.7599999999999998)
)
: (
   0
)) - 1.0953518906603145e-26*cse839*cse841*((cse806) ? (
   -3/std::pow(state.T, 4)
)
: (
   0
)) - 2.9985055107607381e-40*cse839*((cse806) ? (
   2.1000000000000001*std::pow(state.T, 1.1000000000000001)
)
: (
   0
))/std::pow(cse840, 2);
  const amrex::Real cse1097 = ((cse806) ? (
   1
)
: (
   0
));
  const amrex::Real cse1098 = cse1097*cse834;
  const amrex::Real cse1099 = cse10*cse1098*cse843;
  const amrex::Real cse1100 = cse1098*cse816*cse846;
  const amrex::Real cse1101 = cse1098*cse818*cse848;
  const amrex::Real cse1102 = cse1098*cse8;
  const amrex::Real cse1103 = cse1098*cse820*cse849;
  const amrex::Real cse1104 = cse1098*cse810*cse866;
  const amrex::Real cse1105 = cse1098*cse812*cse868;
  const amrex::Real cse1106 = cse1098*cse814*cse844;
  const amrex::Real cse1107 = cse70*cse765;
  const amrex::Real cse1108 = cse70*cse783;
  const amrex::Real cse1109 = 65.320000000000007*cse1108;
  const amrex::Real cse1110 = cse5*cse742;
  const amrex::Real cse1111 = 0.00042999999999999999*cse678;
  const amrex::Real cse1112 = cse757*(cse1079*(0.0041900000000000001 - 3.0000000000000001e-5*cse678) - cse5*cse754) - 3.8399999999999998e-11*std::pow(state.T, -1.075)*state.xn[2] + (cse760*(cse1079*(0.00025000000000000001*cse678 + 0.00264) + cse5*cse758) + cse762*(cse1079*(cse1111 + 0.00296) + cse1110))*state.xn[3];
  const amrex::Real cse1113 = cse774*(cse1079*(0.0036900000000000001 - 6.0000000000000002e-5*cse678) - cse5*cse771) - 1.2635999999999999e-11*std::pow(state.T, -1.026)*state.xn[2] + (cse777*(cse1079*(0.00041000000000000005*cse678 + 0.0020300000000000001) + cse5*cse775) + cse780*(cse1079*(0.00058*cse678 + 0.0023700000000000001) + cse5*cse778))*state.xn[3];
  const amrex::Real cse1114 = cse741*(cse1079*(0.0075500000000000003 - 0.0016000000000000001*cse678) - cse5*cse738) + 1.0000800000000001e-14*std::pow(state.T, -0.073999999999999955)*state.xn[2] + (cse744*(cse1079*(cse1111 + 0.0088900000000000003) + cse1110) + cse747*(cse1079*(0.0052500000000000003*cse678 + 0.011979999999999999) + cse5*cse745))*state.xn[3];
  const amrex::Real cse1115 = cse1114*cse735;
  const amrex::Real cse1116 = cse1113*cse785;
  const amrex::Real cse1117 = 65.320000000000007*cse1107*cse751 + 28.360000000000003*cse1108*cse768 + cse1109*cse786 + cse1112*cse735*cse787 + (1.0/5.0)*cse1112*cse751 + (1.0/15.0)*cse1113*cse768 + cse1114*cse767*cse784 + cse1115*cse766 + (1.0/5.0)*cse1116*cse735;
  const amrex::Real cse1118 = 136.68000000000001*cse1107;
  const amrex::Real cse1119 = cse794*cse795;
  const amrex::Real cse1120 = cse1112*cse790;
  const amrex::Real cse1121 = (3.0/5.0)*cse795;
  const amrex::Real cse1122 = cse1112*cse789;
  const amrex::Real cse1123 = cse1109*cse798 + cse1113*cse1119 + (1.0/5.0)*cse1113*cse798 + cse1114*cse1119 + cse1115*cse787 + cse1118*cse796 + cse1118*cse797 + cse1120*cse1121 + cse1121*cse1122;
  const amrex::Real cse1124 = cse1007*(-cse1113*cse791*cse793 - cse1114*cse785 - 1.0/3.0*cse1114*cse792 - cse1116 - cse1117 - cse1120 - cse1122 - cse1123 - 32.933333333333337*cse70*cse750*cse792);
  const amrex::Real cse1125 = cse70*cse925;
  const amrex::Real cse1126 = 312.30000000000001*cse70;
  const amrex::Real cse1127 = cse1126*cse902;
  const amrex::Real cse1128 = cse5*cse879;
  const amrex::Real cse1129 = 0.00046000000000000001*cse678;
  const amrex::Real cse1130 = 8.6290000000000003e-7*cse5;
  const amrex::Real cse1131 = -cse1130*cse981 + cse906*(cse1079*(cse1129 + 0.00231) + cse1128) + cse924*((cse829) ? (
   cse151*(cse151*(cse5*cse915 + cse5*cse916) + cse5*cse917) + cse5*cse918
)
: (
   cse151*(cse151*(cse5*cse919 + cse5*cse920) + cse5*cse921) + cse5*cse922
)) + (cse909*(cse1079*(0.00031*cse678 + 0.00263) + cse5*cse907) + cse912*(cse1079*(0.00064000000000000005*cse678 + 0.00071999999999999994) + cse5*cse910))*state.xn[3];
  const amrex::Real cse1132 = -1.4381666666666666e-6*cse5*cse950 + cse932*(cse1079*(0.00057000000000000009*cse678 + 0.00115) + cse5*cse929) + cse951*((cse829) ? (
   cse151*(cse151*(-cse5*cse941 + cse5*cse942) + cse5*cse943) + cse5*cse944
)
: (
   cse151*(cse151*(cse5*cse945 + cse5*cse946) + cse5*cse947) + cse5*cse948
)) + (cse935*(cse1079*(0.00048999999999999998*cse678 - 4.0000000000000003e-5) + cse5*cse933) + cse938*(cse1079*(0.0010199999999999999*cse678 - 0.00085000000000000006) + cse5*cse936))*state.xn[3];
  const amrex::Real cse1133 = -cse1130*cse986 + cse882*(cse1079*(cse1129 + 0.0022800000000000003) + cse1128) + cse901*((cse829) ? (
   cse151*(cse151*(cse5*cse891 + cse5*cse892) + cse5*cse893) + cse5*cse894
)
: (
   cse151*(cse151*(cse5*cse895 + cse5*cse896) + cse5*cse897) + cse5*cse898
)) + (cse885*(cse1079*(0.00038000000000000002*cse678 + 0.0016900000000000001) + cse5*cse883) + cse888*(cse1079*(0.00048000000000000001*cse678 - 0.0001) + cse5*cse886))*state.xn[3];
  const amrex::Real cse1134 = cse1131*cse927;
  const amrex::Real cse1135 = cse1133*cse955;
  const amrex::Real cse1136 = 844.16666666666686*cse1125*cse903 + cse1126*cse928*cse952 + cse1127*cse956 + (25.0/3.0)*cse1131*cse903 + cse1132*cse927*cse954 + 5*cse1132*cse928 + cse1133*cse878*cse926 + cse1134*cse953 + 5*cse1135*cse927;
  const amrex::Real cse1137 = 70.859999999999985*cse70*cse952;
  const amrex::Real cse1138 = cse964*cse965;
  const amrex::Real cse1139 = cse1132*cse959;
  const amrex::Real cse1140 = 3*cse965;
  const amrex::Real cse1141 = cse1132*cse958;
  const amrex::Real cse1142 = cse1127*cse963 + cse1131*cse1138 + cse1133*cse1138 + 5*cse1133*cse963 + cse1134*cse954 + cse1137*cse966 + cse1137*cse967 + cse1139*cse1140 + cse1140*cse1141;
  const amrex::Real cse1143 = cse992*(-64.733333333333348*cse1125*cse961 - cse1131*cse955 - 5.0/3.0*cse1131*cse961 - cse1133*cse960*cse962 - cse1135 - cse1136 - cse1139 - cse1141 - cse1142);
  const amrex::Real cse1144 = 1.7e-24*chi*cse1069*cse14*cse146*cse630*d2g*(-0.00012636182959999999*cse1070*cse111 - 0.018302846000000001*cse1071*cse111) + 5.6227499999999995e-25*chi*cse14*cse146*cse634*d2g*std::pow(state.T, -0.85299999999999998) - 1000*cse1027*cse706*std::pow(cse707, 999)*cse731 - cse1028*(-cse1024*cse1093*cse705 - cse1084*cse720 + cse1090*cse1091*cse723*cse727 + cse1090*cse1091 - cse728*(-cse1024*cse1093*cse709*cse721 - cse1090*cse720*cse721 - cse1094*cse722) - cse729*(cse1017*(cse1016*cse1092 + alpha_co_interp2d_partial_0(state.T, cse718)) + cse1023*(-cse1019*cse1094 + cse1084*cse724))) - cse1035*cse1085*cse1087 - cse1036*cse1087*cse1088 - cse1041*(-182.41999999999999*cse1077 - cse1080*cse689 - cse1082*cse1089) - cse1045*cse657*psigd_coll_interp2d_partial_1(cse14, state.T) - cse1051*(1.6000000000000001e-32*cse651*cse652*cse89 + cse654) - 1.6173435610483583e-5*cse1060*cse1061*cse616 - cse1061*cse1067*cse1068 - 2.0489994438957239e-11*cse1062*cse138*cse616*cse627 + 7.8348995947646122e-14*cse1063*cse1064*cse621*cse622*cse626*state.xn[0]*state.xn[2] - cse1076*(-0.0015792330470357991*cse1060*cse618 + 7.6502807803182096e-12*cse1063*cse1064*cse621*cse622*state.xn[2] - 2.0007175426973804e-9*cse1065*cse623 - 1.3338116951315871e-8*cse1067*cse619) - 5.2835524837844135e-18*cse1077*cse692 - cse1080*cse1081 - cse1082*cse1083 - cse1084*cse733 - 2.1789607199999999e-11*cse139 - 7.1777529600000012e-12*cse521 - 7.1777529600000012e-12*cse522 - cse646*(-1.8989*cse1072*cse1074 + cse5*cse640) - cse646*(-0.043256499999999996*cse1072*cse1073*cse639 + 0.84999999999999998*cse1072*cse1075) - (cse1011*cse1124 + cse1012*cse1124 + cse1117*cse802 + cse1123*cse803)*state.xn[11] - ((cse833) ? (
   0
)
: (
   cse1096*cse876 + cse842*cse977*(-cse1096*cse874 - cse842*cse976*(-cse852*(4.8223900769399082*cse1097*cse8*cse834 + 3.0182298984217999*cse1097*cse820*cse834*cse849 - 3.5529549287336835*cse1099 - 0.3872755400043702*cse1100 - 1.3735579540080118*cse1101) - cse856*(0.0708789387907936*cse1099 + 3.7035619079275177*cse1100 - 4.6974781513675152*cse1101 + 3.6184459289309552*cse1102 - 1.6316107607322889*cse1103) - cse858*(-3.7541549062629067*cse1099 + 2.2094886994529301*cse1100 - 1.528565035159452*cse1101 + 5.0409049417480247*cse1102 + 2.0057552335975877*cse1103) - ((cse682) ? (
      cse870*(445.51869310442481*cse1099 + 11324.985706577943*cse1100 + 6766.9891340104432*cse1101 + 38.719649225812766*cse1102 + 2370.6849681533818*cse1103 + 1300.6686834345674*cse1104 + 5869.3152370465659*cse1105 + 11077.448644792414*cse1106)
   )
   : (
      cse871*(4.297112944704045*cse1099 - 101.40241318979159*cse1100 - 43.540996231705549*cse1101 + 3.8689780091986448*cse1102 + 27.91190909651122*cse1103 - 117.34186576967058*cse1104 + 103.76942484394611*cse1105 + 123.18901581604325*cse1106)
   ))*state.xn[2] - ((cse860) ? (
      cse7*cse859*(116.29033200000001*cse1099 + 19.239805*cse1100 + 80.639324000000002*cse1101 + 37.383713*cse1102 + 145.968309*cse1103)
   )
   : ((cse862) ? (
      cse861*(8.2184944748967013*cse1097*cse8*cse834 - 52.189748993977005*cse1099 - 48.951834264235494*cse1100 - 196.44057098336626*cse1101 - 192.38155095558542*cse1103)
   )
   : (
      cse863*(-17.135767342440825*cse1099 + 17.889115159724135*cse1100 - 50.756388852554174*cse1101 + 10.695627721640689*cse1102 + 41.010708268606813*cse1103)
   )))*state.xn[0]))
))*state.xn[3] - ((cse973) ? (
   cse1136*cse971 + cse1142*cse972 + cse1143*cse995 + cse1143*cse996
)
: (
   0
))*state.xn[7];
  const amrex::Real cse1145 = cse1144*cse193;
  const amrex::Real cse1146 = 7.1777529600000012e-12*cse202;
  const amrex::Real cse1147 = 7.1777529600000012e-12*cse529;
  const amrex::Real cse1148 = cse724*cse726;
  const amrex::Real cse1149 = cse1028*(-cse1026 - cse729*(cse1015*cse1016*cse1017 - cse1021*cse1148));
  const amrex::Real cse1150 = -cse1034*cse1036 + cse28*cse670*cse671;
  const amrex::Real cse1151 = std::pow(state.T, 0.29999999999999999);
  const amrex::Real cse1152 = cse641*cse643;
  const amrex::Real cse1153 = std::pow(cse635, -2);
  const amrex::Real cse1154 = cse1153*cse147;
  const amrex::Real cse1155 = 9.0599999999999997e-6*cse76;
  const amrex::Real cse1156 = 9.7199999999999998e-11*cse769;
  const amrex::Real cse1157 = 5.1199999999999999e-10*cse752;
  const amrex::Real cse1158 = 1.08e-14*cse736;
  const amrex::Real cse1159 = cse1000*cse1158 + cse1001*cse1158 + cse1156*cse786 + cse1157*cse999 + 1.024e-10*cse751*cse752 + 3.2399999999999999e-11*cse768*cse769;
  const amrex::Real cse1160 = 3.0719999999999998e-10*cse752;
  const amrex::Real cse1161 = 4.8599999999999998e-10*cse769;
  const amrex::Real cse1162 = cse1005*cse1158 + cse1005*cse1161 + cse1156*cse798 + cse1158*cse999 + cse1160*cse796 + cse1160*cse797;
  const amrex::Real cse1163 = cse1007*(-cse1009*cse1161 - cse1157*cse789 - cse1157*cse790 - cse1158*cse785 - cse1159 - cse1161*cse785 - cse1162 - 3.6000000000000001e-15*cse736*cse792);
  const amrex::Real cse1164 = cse76*cse923;
  const amrex::Real cse1165 = cse76*cse949;
  const amrex::Real cse1166 = 8.6289999999999999e-6*cse76*cse899;
  const amrex::Real cse1167 = 2.8763333333333333e-6*cse1165;
  const amrex::Real cse1168 = 1.7258000000000001e-6*cse76;
  const amrex::Real cse1169 = cse1168*cse899;
  const amrex::Real cse1170 = cse1168*cse923;
  const amrex::Real cse1171 = 1.4381666666666668e-5*cse1164*cse903 + 1.4381666666666666e-5*cse1165*cse928 + cse1166*cse956 + cse1167*cse987 + cse1169*cse982 + cse1170*cse984;
  const amrex::Real cse1172 = 8.6289999999999999e-6*cse1165;
  const amrex::Real cse1173 = cse1166*cse963 + cse1169*cse990 + cse1170*cse987 + cse1170*cse990 + cse1172*cse966 + cse1172*cse967;
  const amrex::Real cse1174 = cse1168*cse955;
  const amrex::Real cse1175 = cse992*(-2.8763333333333337e-6*cse1164*cse961 - cse1167*cse958 - cse1167*cse959 - cse1169*cse993 - cse1171 - cse1173 - cse1174*cse899 - cse1174*cse923);
  const amrex::Real cse1176 = cse1049*cse1152;
  const amrex::Real cse1177 = 3.7996253111508318e-29*cse1176;
  const amrex::Real cse1178 = cse1052*(4*state.xn[0] + 4*state.xn[10] + 4*state.xn[14] + 4*state.xn[15] + 12*state.xn[16] + 4*state.xn[1] + 8*state.xn[3] + 8*state.xn[4]);
  const amrex::Real cse1179 = -cse684;
  const amrex::Real cse1180 = cse1043*cse163;
  const amrex::Real cse1181 = cse1045*(200.0*cse1042 + 200.0*cse1180);
  const amrex::Real cse1182 = ((cse697) ? (
   0
)
: ((cse698) ? (
   3*cse1047
)
: ((cse699) ? (
   (8.0/3.0)*cse1047
)
: ((cse700) ? (
   (2.0/3.0)*cse1047
)
: (
   0
)))));
  const amrex::Real cse1183 = cse1055*cse648;
  const amrex::Real cse1184 = 1.0*cse1056*cse1057*cse215 - 1.0*cse1183;
  const amrex::Real cse1185 = cse240*cse663 + cse242*cse666;
  const amrex::Real cse1186 = 2*cse1031 - cse1036*(cse1185 - cse664);
  const amrex::Real cse1187 = ((cse714) ? (
   0
)
: (
   cse1013*cse717
));
  const amrex::Real cse1188 = cse1020*cse1187;
  const amrex::Real cse1189 = -cse1024*cse1025*cse1187*cse705 + cse728*(cse1024*cse1025*cse1187*cse709*cse721 - cse1188*cse722);
  const amrex::Real cse1190 = cse735*(9.4999999999999995e-12*cse776 + 3.3450000000000005e-11*cse779);
  const amrex::Real cse1191 = cse1000*cse748 + cse1001*cse748 + cse1190*cse785 + cse751*(7.4500000000000008e-12*cse759 + 2.0549999999999999e-11*cse761) + cse763*cse999 + cse768*(3.1666666666666666e-12*cse776 + 1.1150000000000002e-11*cse779);
  const amrex::Real cse1192 = cse795*(2.2349999999999999e-11*cse759 + 6.1649999999999989e-11*cse761);
  const amrex::Real cse1193 = cse1005*cse748 + cse1005*cse781 + cse1190*cse789 + cse1192*cse789 + cse1192*cse790 + cse748*cse999;
  const amrex::Real cse1194 = cse1007*(-cse1009*cse781 - cse1191 - cse1193 - cse748*cse785 - cse763*cse789 - cse763*cse790 - cse781*cse785 - cse792*(1.7499999999999999e-13*cse743 + 7.5000000000000004e-13*cse746));
  const amrex::Real cse1195 = cse927*(2.5874999999999996e-10*cse884 + 1.075e-10*cse887);
  const amrex::Real cse1196 = cse1195*cse955 + cse889*cse982 + cse903*(9.2500000000000011e-10*cse908 + 3.6458333333333334e-10*cse911) + cse913*cse984 + cse928*(2.6624999999999999e-10*cse934 + 8.3749999999999995e-11*cse937) + cse939*cse987;
  const amrex::Real cse1197 = cse965*(1.5974999999999999e-10*cse934 + 5.0250000000000001e-11*cse937);
  const amrex::Real cse1198 = cse1195*cse958 + cse1197*cse958 + cse1197*cse959 + cse889*cse990 + cse913*cse987 + cse913*cse990;
  const amrex::Real cse1199 = cse992*(-cse1196 - cse1198 - cse889*cse955 - cse889*cse993 - cse913*cse955 - cse939*cse958 - cse939*cse959 - cse961*(1.8500000000000001e-10*cse908 + 7.291666666666666e-11*cse911));
  const amrex::Real cse1200 = cse147*cse634*d2g;
  const amrex::Real cse1201 = cse1200*cse630;
  const amrex::Real cse1202 = 3.2043540000000001e-12*crate*state.xn[3];
  const amrex::Real cse1203 = 4.22974728e-11*cse110*cse14*cse189*cse647*cse649;
  const amrex::Real cse1204 = -cse1036*cse1185 + 2*cse28*cse670*cse671;
  const amrex::Real cse1205 = cse1030*cse1088;
  const amrex::Real cse1206 = cse1030*cse1085*cse14;
  const amrex::Real cse1207 = cse66*((cse833) ? (
   0
)
: (
   cse857*cse978
))*state.xn[3];
  const amrex::Real cse1208 = ((cse714) ? (
   cse713
)
: (
   0
));
  const amrex::Real cse1209 = cse1020*cse1208;
  const amrex::Real cse1210 = cse66*(-cse1046 + cse1048*cse1202 - cse1050 - cse1053 + cse1058*cse1203 - cse1146 - cse1147 - cse1149 + cse1150*cse1205 + cse1150*cse1206 + 1.7e-24*cse1200*cse630 + cse515*cse695);
  const amrex::Real cse1211 = -cse1036*(3*cse1032 + 3*cse1033) + 3*cse28*cse670*cse671;
  const amrex::Real cse1212 = ((cse714) ? (
   0
)
: (
   2635052181.6465044*cse1014
));
  const amrex::Real cse1213 = cse1020*cse1212;
  jac(1, 1) = -cse1 - cse14*cse58 + 0.66666666666670005*cse177*cse63*cse65*cse69 - cse191 + cse3*state.xn[17] + 2*cse4*cse49 + 2*cse57*state.xn[3] - cse59 + 2*cse62*state.xn[0]*state.xn[3];
  jac(1, 2) = cse108 + cse192*cse195 + cse207 + cse211;
  jac(1, 3) = cse195*cse214 + cse213*state.xn[16] + cse218*cse219 + cse224 + cse231;
  jac(1, 4) = cse195*cse260 + cse205*cse253 - cse219*cse261 + cse233 + cse235 + cse237 + 2*cse247 + 2*cse248 + cse249*cse250 + cse251*state.xn[17] + 1.1399999999999999e-10*cse257 + cse259*state.xn[8] + cse265 + cse272;
  jac(1, 5) = cse195*cse281 + cse205*cse280 + cse237 + cse273 + cse275 + 2*cse279 + cse284;
  jac(1, 6) = cse195*cse285;
  jac(1, 7) = cse195*cse286 + cse290;
  jac(1, 8) = cse195*cse293 + cse292;
  jac(1, 9) = cse195*cse294 + cse259*state.xn[3];
  jac(1, 10) = 0.66666666666670005*cse177*cse296*cse63*cse65 - cse295;
  jac(1, 11) = cse195*cse297 + cse206 + cse299;
  jac(1, 12) = cse195*cse305 + cse300 + cse301 + cse302 + cse304;
  jac(1, 13) = cse195*cse306;
  jac(1, 14) = cse195*cse307;
  jac(1, 15) = cse195*cse312 + cse207 + cse308 + cse309 + cse311;
  jac(1, 16) = cse195*cse314 + cse207 + cse313 + cse317;
  jac(1, 17) = cse195*cse324 + cse205*cse323 + cse213*state.xn[2] - 1.8000000000000002e-14*cse236 + 2*cse322 + cse330;
  jac(1, 18) = cse195*cse334 + cse251*state.xn[3] + cse331 + cse333;
  jac(1, 19) = cse195;
  jac(2, 1) = cse191 + cse335 + cse338*cse69;
  jac(2, 2) = cse192*cse338 - cse209 - cse210 + cse340 + cse341;
  jac(2, 3) = 0.66666666666670005*cse214*cse337*cse63*cse65 - cse230;
  jac(2, 4) = cse233 + cse260*cse338 + cse270 + cse342;
  jac(2, 5) = cse274 + cse281*cse338 + cse283 + cse342;
  jac(2, 6) = cse285*cse338;
  jac(2, 7) = cse286*cse338 + cse287;
  jac(2, 8) = cse293*cse338;
  jac(2, 9) = cse294*cse338;
  jac(2, 10) = cse296*cse338;
  jac(2, 11) = cse297*cse338 + cse341;
  jac(2, 12) = cse305*cse338 + cse343;
  jac(2, 13) = cse306*cse338;
  jac(2, 14) = cse307*cse338;
  jac(2, 15) = cse312*cse338 + cse341;
  jac(2, 16) = cse314*cse338 + cse341;
  jac(2, 17) = cse324*cse338 + cse328 - cse329;
  jac(2, 18) = cse334*cse338 + cse344;
  jac(2, 19) = cse338;
  jac(3, 1) = -cse181 + cse184 + cse186 + cse190 - cse347 + 0.66666666666670005*cse435*cse63*cse65*cse69 - cse451;
  jac(3, 2) = 0.66666666666670005*cse192*cse435*cse63*cse65 - cse211 - cse454;
  jac(3, 3) = 0.66666666666670005*cse214*cse435*cse63*cse65 - cse222 - cse231 - cse455 - cse457 + cse459*state.xn[8] - cse460 - cse462 - cse467;
  jac(3, 4) = 0.66666666666670005*cse260*cse435*cse63*cse65 - cse272 - cse469 + cse470 + cse471 - cse479 - cse486 - cse490 - cse491;
  jac(3, 5) = 0.66666666666670005*cse281*cse435*cse63*cse65 - cse284 - cse469 - cse478 - cse485 + cse492;
  jac(3, 6) = cse285*cse494 + cse493;
  jac(3, 7) = 0.66666666666670005*cse286*cse435*cse63*cse65 - cse495;
  jac(3, 8) = cse293*cse494 + cse498 + cse500;
  jac(3, 9) = 0.66666666666670005*cse294*cse435*cse63*cse65 + cse459*state.xn[2] - cse501 - cse502;
  jac(3, 10) = cse295 + cse296*cse494;
  jac(3, 11) = 0.66666666666670005*cse297*cse435*cse63*cse65 - cse299 - cse453;
  jac(3, 12) = 0.66666666666670005*cse305*cse435*cse63*cse65 - cse504;
  jac(3, 13) = cse306*cse494 + cse505;
  jac(3, 14) = 0.66666666666670005*cse307*cse435*cse63*cse65 - cse506;
  jac(3, 15) = 0.66666666666670005*cse312*cse435*cse63*cse65 - cse454;
  jac(3, 16) = 0.66666666666670005*cse314*cse435*cse63*cse65 - cse454;
  jac(3, 17) = 0.66666666666670005*cse324*cse435*cse63*cse65 - cse330 - cse507 - cse508 - cse510 - cse511 - cse512 - cse513;
  jac(3, 18) = 0.66666666666670005*cse334*cse435*cse63*cse65 - cse514;
  jac(3, 19) = cse494;
  jac(4, 1) = cse1 + 3.0000000000000002e-15*cse163 + cse183 + cse347 + cse515 - cse516 - cse518 - cse519 + cse528*cse69;
  jac(4, 2) = 0.66666666666670005*cse192*cse527*cse63*cse65 - cse530;
  jac(4, 3) = cse214*cse528 + cse531*state.xn[16] + cse532 + cse533;
  jac(4, 4) = -cse13*cse249 - cse203 - cse247 - cse248 + 0.66666666666670005*cse260*cse527*cse63*cse65 - cse481 - cse534 - cse535 - cse537 - cse538 - cse540 - cse541;
  jac(4, 5) = -cse203 - cse273 - cse279 + 0.66666666666670005*cse281*cse527*cse63*cse65 - cse542 - cse543;
  jac(4, 6) = cse285*cse528;
  jac(4, 7) = 0.66666666666670005*cse286*cse527*cse63*cse65 - cse545;
  jac(4, 8) = cse293*cse528 + cse546;
  jac(4, 9) = 0.66666666666670005*cse294*cse527*cse63*cse65 - cse501;
  jac(4, 10) = cse296*cse528 + cse547;
  jac(4, 11) = 0.66666666666670005*cse297*cse527*cse63*cse65 - cse530;
  jac(4, 12) = cse305*cse528 + cse504;
  jac(4, 13) = cse306*cse528;
  jac(4, 14) = cse307*cse528;
  jac(4, 15) = -cse311 + 0.66666666666670005*cse312*cse527*cse63*cse65 - cse530;
  jac(4, 16) = 0.66666666666670005*cse314*cse527*cse63*cse65 - cse530;
  jac(4, 17) = -cse322 + cse324*cse528 + cse510 + cse531*state.xn[2] + cse548 + cse549 - cse550 + cse552;
  jac(4, 18) = 0.66666666666670005*cse334*cse527*cse63*cse65 - cse514;
  jac(4, 19) = cse528;
  jac(5, 1) = -cse183 + cse346 + cse553*cse69;
  jac(5, 2) = cse192*cse553 + cse452;
  jac(5, 3) = cse214*cse553;
  jac(5, 4) = -cse235 + cse260*cse553 + cse536 + cse538;
  jac(5, 5) = -cse273 + cse281*cse553 + cse543;
  jac(5, 6) = cse285*cse553;
  jac(5, 7) = cse286*cse553 + cse544;
  jac(5, 8) = cse293*cse553;
  jac(5, 9) = cse294*cse553;
  jac(5, 10) = cse296*cse553;
  jac(5, 11) = cse297*cse553 + cse452;
  jac(5, 12) = cse305*cse553;
  jac(5, 13) = cse306*cse553;
  jac(5, 14) = cse307*cse553;
  jac(5, 15) = cse312*cse553 + cse452;
  jac(5, 16) = cse314*cse553 + cse452;
  jac(5, 17) = cse324*cse553 + cse509;
  jac(5, 18) = cse334*cse553;
  jac(5, 19) = cse553;
  jac(6, 1) = cse439 + cse556*cse69;
  jac(6, 2) = cse192*cse556 + cse439;
  jac(6, 3) = cse214*cse556 + cse460;
  jac(6, 4) = cse260*cse556 + cse557;
  jac(6, 5) = cse281*cse556 + cse469;
  jac(6, 6) = 0.66666666666670005*cse285*cse555*cse63*cse65 - cse493;
  jac(6, 7) = cse286*cse556 + cse290 + cse544 + cse559;
  jac(6, 8) = cse293*cse556;
  jac(6, 9) = cse294*cse556;
  jac(6, 10) = cse296*cse556 + cse560;
  jac(6, 11) = cse297*cse556 + cse439;
  jac(6, 12) = cse305*cse556;
  jac(6, 13) = cse306*cse556;
  jac(6, 14) = cse307*cse556;
  jac(6, 15) = cse312*cse556 + cse439;
  jac(6, 16) = cse313 + cse314*cse556 + cse439;
  jac(6, 17) = cse324*cse556 + cse507;
  jac(6, 18) = cse334*cse556;
  jac(6, 19) = cse556;
  jac(7, 1) = -cse439 + 0.66666666666670005*cse561*cse63*cse65*cse69;
  jac(7, 2) = 0.66666666666670005*cse192*cse561*cse63*cse65 - cse439;
  jac(7, 3) = 0.66666666666670005*cse214*cse561*cse63*cse65 - cse460;
  jac(7, 4) = 0.66666666666670005*cse260*cse561*cse63*cse65 - cse557;
  jac(7, 5) = 0.66666666666670005*cse281*cse561*cse63*cse65 - cse469;
  jac(7, 6) = cse285*cse562 + cse493;
  jac(7, 7) = 0.66666666666670005*cse286*cse561*cse63*cse65 - cse545 - cse559 + cse563*state.xn[15];
  jac(7, 8) = cse293*cse562;
  jac(7, 9) = cse294*cse562;
  jac(7, 10) = cse296*cse562 - cse560;
  jac(7, 11) = 0.66666666666670005*cse297*cse561*cse63*cse65 - cse439;
  jac(7, 12) = cse305*cse562;
  jac(7, 13) = cse306*cse562;
  jac(7, 14) = cse307*cse562;
  jac(7, 15) = 0.66666666666670005*cse312*cse561*cse63*cse65 - cse439;
  jac(7, 16) = cse314*cse562 - cse439 + cse563*state.xn[6];
  jac(7, 17) = 0.66666666666670005*cse324*cse561*cse63*cse65 - cse507;
  jac(7, 18) = cse334*cse562;
  jac(7, 19) = cse562;
  jac(8, 1) = cse1 + cse566*cse69 + cse570;
  jac(8, 2) = cse192*cse566 + cse570;
  jac(8, 3) = cse214*cse566 + cse455 + cse458*state.xn[8];
  jac(8, 4) = cse260*cse566 + cse483 + cse572 + cse573*state.xn[8] + cse576;
  jac(8, 5) = cse281*cse566 + cse576;
  jac(8, 6) = cse285*cse566;
  jac(8, 7) = cse286*cse566;
  jac(8, 8) = -cse292 + 0.66666666666670005*cse293*cse565*cse63*cse65 + cse497*state.xn[16] - cse500;
  jac(8, 9) = cse294*cse566 + cse458*state.xn[2] + cse502 + cse573*state.xn[3];
  jac(8, 10) = cse296*cse566 + cse577;
  jac(8, 11) = cse297*cse566 + cse570;
  jac(8, 12) = cse305*cse566;
  jac(8, 13) = cse306*cse566;
  jac(8, 14) = cse307*cse566;
  jac(8, 15) = cse312*cse566 + cse570 + cse578;
  jac(8, 16) = cse314*cse566 - cse315 + cse570;
  jac(8, 17) = cse324*cse566 + cse497*state.xn[7] + cse511 - cse579;
  jac(8, 18) = cse334*cse566;
  jac(8, 19) = cse566;
  jac(9, 1) = -cse445 + 0.66666666666670005*cse582*cse63*cse65*cse69;
  jac(9, 2) = 0.66666666666670005*cse192*cse582*cse63*cse65 - cse445;
  jac(9, 3) = cse214*cse583 - cse455 + cse459*state.xn[8];
  jac(9, 4) = 0.66666666666670005*cse260*cse582*cse63*cse65 - cse486;
  jac(9, 5) = 0.66666666666670005*cse281*cse582*cse63*cse65 - cse485;
  jac(9, 6) = cse285*cse583;
  jac(9, 7) = cse286*cse583 + cse558;
  jac(9, 8) = cse293*cse583 + cse500;
  jac(9, 9) = cse294*cse583 + cse459*state.xn[2] - cse501 - cse502 + cse586;
  jac(9, 10) = cse296*cse583 + cse560;
  jac(9, 11) = 0.66666666666670005*cse297*cse582*cse63*cse65 - cse445;
  jac(9, 12) = cse305*cse583;
  jac(9, 13) = cse306*cse583;
  jac(9, 14) = cse307*cse583;
  jac(9, 15) = 0.66666666666670005*cse312*cse582*cse63*cse65 - cse445;
  jac(9, 16) = 0.66666666666670005*cse314*cse582*cse63*cse65 - cse445 - cse587;
  jac(9, 17) = 0.66666666666670005*cse324*cse582*cse63*cse65 - cse511;
  jac(9, 18) = cse334*cse583;
  jac(9, 19) = cse583;
  jac(10, 1) = cse568 + cse589*cse69;
  jac(10, 2) = cse192*cse589 + cse568;
  jac(10, 3) = cse214*cse589 + cse226;
  jac(10, 4) = cse260*cse589 - cse572 + cse574;
  jac(10, 5) = cse281*cse589 + cse574;
  jac(10, 6) = cse285*cse589;
  jac(10, 7) = cse286*cse589 - cse558;
  jac(10, 8) = cse292 + cse293*cse589;
  jac(10, 9) = cse294*cse589;
  jac(10, 10) = 0.66666666666670005*cse296*cse588*cse63*cse65 - cse590 - cse591;
  jac(10, 11) = cse297*cse589 + cse298 + cse568;
  jac(10, 12) = cse300 + cse305*cse589;
  jac(10, 13) = cse306*cse589;
  jac(10, 14) = cse307*cse589;
  jac(10, 15) = cse312*cse589 + cse592;
  jac(10, 16) = cse314*cse589 + cse315 + cse568;
  jac(10, 17) = cse324*cse589 - cse551 + cse579;
  jac(10, 18) = cse334*cse589;
  jac(10, 19) = cse589;
  jac(11, 1) = cse594*cse69;
  jac(11, 2) = cse192*cse594;
  jac(11, 3) = 0.66666666666670005*cse214*cse593*cse63*cse65 - cse226;
  jac(11, 4) = cse260*cse594;
  jac(11, 5) = cse281*cse594;
  jac(11, 6) = cse285*cse594;
  jac(11, 7) = cse286*cse594;
  jac(11, 8) = cse293*cse594;
  jac(11, 9) = cse294*cse594 + cse584*state.xn[15];
  jac(11, 10) = cse296*cse594 + cse590;
  jac(11, 11) = 0.66666666666670005*cse297*cse593*cse63*cse65 - cse298;
  jac(11, 12) = cse305*cse594;
  jac(11, 13) = cse306*cse594;
  jac(11, 14) = cse307*cse594;
  jac(11, 15) = cse312*cse594;
  jac(11, 16) = cse314*cse594 + cse584*state.xn[8];
  jac(11, 17) = cse324*cse594 + cse551;
  jac(11, 18) = cse334*cse594;
  jac(11, 19) = cse594;
  jac(12, 1) = cse335 + cse569 + cse598*cse69;
  jac(12, 2) = cse192*cse598 + cse340 + cse569;
  jac(12, 3) = cse214*cse598 + cse222 - cse464 + cse465;
  jac(12, 4) = cse260*cse598 + cse491 + cse539 + cse572 + cse575;
  jac(12, 5) = 0.66666666666670005*cse281*cse597*cse63*cse65 - cse574;
  jac(12, 6) = cse285*cse598;
  jac(12, 7) = cse286*cse598 + cse558;
  jac(12, 8) = cse293*cse598;
  jac(12, 9) = cse294*cse598;
  jac(12, 10) = cse296*cse598 + cse591;
  jac(12, 11) = 0.66666666666670005*cse297*cse597*cse63*cse65 - cse568;
  jac(12, 12) = cse301 + cse305*cse598 + cse343 - cse503 + cse599;
  jac(12, 13) = cse306*cse598;
  jac(12, 14) = cse307*cse598;
  jac(12, 15) = 0.66666666666670005*cse312*cse597*cse63*cse65 - cse592;
  jac(12, 16) = cse314*cse598 + cse316 + cse569;
  jac(12, 17) = 0.66666666666670005*cse324*cse597*cse63*cse65 - cse508 - cse579;
  jac(12, 18) = cse332 + cse334*cse598 + cse344;
  jac(12, 19) = cse598;
  jac(13, 1) = cse450 + cse601*cse69;
  jac(13, 2) = cse192*cse601 + cse450;
  jac(13, 3) = cse214*cse601 + cse462;
  jac(13, 4) = cse260*cse601 + cse479;
  jac(13, 5) = cse281*cse601 + cse478;
  jac(13, 6) = cse285*cse601;
  jac(13, 7) = cse286*cse601;
  jac(13, 8) = cse293*cse601;
  jac(13, 9) = cse294*cse601;
  jac(13, 10) = cse296*cse601;
  jac(13, 11) = cse297*cse601 + cse450;
  jac(13, 12) = cse305*cse601;
  jac(13, 13) = 0.66666666666670005*cse306*cse600*cse63*cse65 - cse505;
  jac(13, 14) = cse307*cse601 + cse506;
  jac(13, 15) = cse312*cse601 + cse450;
  jac(13, 16) = cse314*cse601 + cse450;
  jac(13, 17) = cse324*cse601 + cse512;
  jac(13, 18) = cse334*cse601;
  jac(13, 19) = cse601;
  jac(14, 1) = -cse450 + 0.66666666666670005*cse602*cse63*cse65*cse69;
  jac(14, 2) = 0.66666666666670005*cse192*cse602*cse63*cse65 - cse450;
  jac(14, 3) = 0.66666666666670005*cse214*cse602*cse63*cse65 - cse462;
  jac(14, 4) = cse260*cse603 + cse474 - cse476 - cse477;
  jac(14, 5) = 0.66666666666670005*cse281*cse602*cse63*cse65 - cse478;
  jac(14, 6) = cse285*cse603;
  jac(14, 7) = cse286*cse603;
  jac(14, 8) = cse293*cse603;
  jac(14, 9) = cse294*cse603;
  jac(14, 10) = cse296*cse603;
  jac(14, 11) = 0.66666666666670005*cse297*cse602*cse63*cse65 - cse450;
  jac(14, 12) = cse305*cse603;
  jac(14, 13) = cse306*cse603 + cse505;
  jac(14, 14) = 0.66666666666670005*cse307*cse602*cse63*cse65 - cse506;
  jac(14, 15) = 0.66666666666670005*cse312*cse602*cse63*cse65 - cse450;
  jac(14, 16) = 0.66666666666670005*cse314*cse602*cse63*cse65 - cse450;
  jac(14, 17) = 0.66666666666670005*cse324*cse602*cse63*cse65 - cse512;
  jac(14, 18) = cse334*cse603;
  jac(14, 19) = cse603;
  jac(15, 1) = -cse1 + 0.66666666666670005*cse604*cse63*cse65*cse69;
  jac(15, 2) = cse192*cse605;
  jac(15, 3) = cse214*cse605;
  jac(15, 4) = cse260*cse605 + cse606*state.xn[8];
  jac(15, 5) = cse281*cse605;
  jac(15, 6) = cse285*cse605;
  jac(15, 7) = cse286*cse605;
  jac(15, 8) = cse293*cse605 + cse546;
  jac(15, 9) = cse294*cse605 + cse606*state.xn[3];
  jac(15, 10) = cse296*cse605;
  jac(15, 11) = cse297*cse605;
  jac(15, 12) = cse305*cse605 + cse599;
  jac(15, 13) = cse306*cse605;
  jac(15, 14) = cse307*cse605;
  jac(15, 15) = -cse308 + 0.66666666666670005*cse312*cse604*cse63*cse65 - cse578;
  jac(15, 16) = cse314*cse605;
  jac(15, 17) = cse324*cse605 + cse548;
  jac(15, 18) = cse334*cse605;
  jac(15, 19) = cse605;
  jac(16, 1) = cse608*cse69;
  jac(16, 2) = cse192*cse608;
  jac(16, 3) = cse214*cse608 + cse467;
  jac(16, 4) = cse260*cse608 + cse490;
  jac(16, 5) = cse281*cse608;
  jac(16, 6) = cse285*cse608;
  jac(16, 7) = cse286*cse608 + cse563*state.xn[15];
  jac(16, 8) = -cse292 + 0.66666666666670005*cse293*cse607*cse63*cse65;
  jac(16, 9) = cse294*cse608 + cse586;
  jac(16, 10) = cse296*cse608;
  jac(16, 11) = cse297*cse608;
  jac(16, 12) = -cse301 + cse305*cse608 + cse503;
  jac(16, 13) = cse306*cse608;
  jac(16, 14) = cse307*cse608;
  jac(16, 15) = cse312*cse608;
  jac(16, 16) = 0.66666666666670005*cse314*cse607*cse63*cse65 - cse317 + cse563*state.xn[6] - cse587;
  jac(16, 17) = cse324*cse608 + cse508;
  jac(16, 18) = cse333 + cse334*cse608;
  jac(16, 19) = cse608;
  jac(17, 1) = cse610*cse69;
  jac(17, 2) = cse192*cse610;
  jac(17, 3) = 0.66666666666670005*cse214*cse609*cse63*cse65 - cse457 - cse533;
  jac(17, 4) = cse260*cse610 + cse540;
  jac(17, 5) = cse273 + cse281*cse610;
  jac(17, 6) = cse285*cse610;
  jac(17, 7) = cse286*cse610;
  jac(17, 8) = cse293*cse610 + cse498;
  jac(17, 9) = cse294*cse610;
  jac(17, 10) = cse296*cse610 - cse547;
  jac(17, 11) = cse297*cse610;
  jac(17, 12) = 0.66666666666670005*cse305*cse609*cse63*cse65 - cse504;
  jac(17, 13) = cse306*cse610;
  jac(17, 14) = cse307*cse610;
  jac(17, 15) = cse312*cse610;
  jac(17, 16) = cse314*cse610;
  jac(17, 17) = 0.66666666666670005*cse324*cse609*cse63*cse65 - cse513 - cse552;
  jac(17, 18) = cse334*cse610;
  jac(17, 19) = cse610;
  jac(18, 1) = cse3*state.xn[17] + cse612*cse69;
  jac(18, 2) = cse108 + cse192*cse612;
  jac(18, 3) = cse214*cse612 + cse532;
  jac(18, 4) = 0.66666666666670005*cse260*cse611*cse63*cse65 - cse541;
  jac(18, 5) = cse281*cse612;
  jac(18, 6) = cse285*cse612;
  jac(18, 7) = cse286*cse612 + cse289;
  jac(18, 8) = cse293*cse612;
  jac(18, 9) = cse294*cse612;
  jac(18, 10) = cse296*cse612;
  jac(18, 11) = cse297*cse612;
  jac(18, 12) = cse302 + cse305*cse612;
  jac(18, 13) = cse306*cse612;
  jac(18, 14) = cse307*cse612;
  jac(18, 15) = cse312*cse612;
  jac(18, 16) = cse313 + cse314*cse612;
  jac(18, 17) = cse324*cse612;
  jac(18, 18) = cse331 - cse332 - cse333 + cse334*cse612;
  jac(18, 19) = cse612;
  jac(19, 1) = cse1145*cse69 + cse66*(1.461185424e-21*chi*cse1030*cse1037*cse255*cse256*state.xn[3] + 1.602177e-12*crate*cse650 - cse1028*(-cse1026 - cse704*cse720 - cse729*(cse1015*cse1018 + cse1023*(-cse1019*cse1021 + cse704*cse724))) - 4.39089220777172e-29*cse1029*cse680 + 2.0187430200000004e-26*cse1030*cse1037*cse14*d2g*state.xn[0] + 4.39089220777172e-29*cse1039*cse680*cse685*cse687*cse691*state.xn[8] - cse1041*(cse1038*cse1040 - cse1038*cse688) - cse1059 + 3.0000000000000002e-15*cse14*cse695*d2g - 7.1777529600000012e-12*cse518 - 7.1777529600000012e-12*cse519 - cse613 - cse619*cse628 - cse704*cse733 - cse734 + (-cse1002*cse802 - cse1006*cse803 - cse1010*cse1011 - cse1010*cse1012)*state.xn[11] - ((cse833) ? (
   0
)
: (
   cse864*cse978
))*state.xn[3] - ((cse973) ? (
   cse971*cse988 + cse972*cse991 + cse994*cse995 + cse994*cse996
)
: (
   0
))*state.xn[7]) - 1.6737729999999998e-24*cse975;
  jac(19, 2) = cse1145*cse192 + cse66*(1.461185424e-21*chi*cse1030*cse1150*cse255*cse256*state.xn[3] + 2.0187430200000004e-26*cse1030*cse1150*cse14*d2g*state.xn[0] - cse1059 - cse1146 - cse1147 - cse1149 - ((cse833) ? (
   0
)
: (
   cse855*cse978
))*state.xn[3]) - 1.6728620616289998e-24*cse975;
  jac(19, 3) = cse1145*cse214 + cse66*(1.7e-24*chi*cse1069*cse14*cse146*cse630*d2g*(0.00025272365919999998*cse1070*cse217 + 0.036605692000000002*cse1071*cse217) + 4.22974728e-11*crate*cse110*cse14*cse647*cse649*(cse1054*cse15 - 0.5*cse1055*cse1057)*state.xn[0] - cse1028*(-cse1022*cse703*cse729 - cse703*cse720) - 2.6241084038530201e-25*cse1029*cse76 + 2.6241084038530201e-25*cse1039*cse685*cse687*cse691*cse76*state.xn[8] - cse1041*(cse1040*cse1155 - cse1155*cse688) + 6.0732876035249664e-19*cse1064*cse1066*cse1151*cse622*cse626*state.xn[0]*state.xn[2] - 7.2151085033443149e-29*cse1074*cse1152*cse1154*cse163*cse50*state.xn[2] - cse1076*(5.9301787935659931e-17*cse1064*cse1066*cse1151*cse622*state.xn[2] - cse617*cse624) - 2.1789607199999999e-11*cse229 - cse629*state.xn[0] - cse645 - cse646*(0.086512999999999993*chi*cse1073*cse1153*cse146*cse50*cse639 - cse1075*cse1154*cse637) - cse703*cse733 + (-cse1011*cse1163 - cse1012*cse1163 - cse1159*cse802 - cse1162*cse803)*state.xn[11] - ((cse833) ? (
   0
)
: (
   cse872*cse978
))*state.xn[3] - ((cse973) ? (
   cse1171*cse971 + cse1173*cse972 + cse1175*cse995 + cse1175*cse996
)
: (
   0
))*state.xn[7]) - 9.1093837099999994e-28*cse975;
  jac(19, 4) = cse1145*cse260 + cse66*(1.461185424e-21*chi*cse1030*cse1186*cse255*cse256*state.xn[3] + 3.3999999999999999e-24*chi*cse146*cse630*cse634*d2g + 5.6999999999999997e-11*chi*cse255*cse256*cse675 + 4.22974728e-11*crate*cse110*cse1184*cse14*cse647*cse649*state.xn[0] + 3.2043540000000001e-12*crate*cse1182*state.xn[3] + 3.2043540000000001e-12*crate*cse701 - cse1028*(-cse1189 - cse720 - cse729*(cse1018*cse1187 + cse1023*(-cse1019*cse1188 + cse724))) + 2.0187430200000004e-26*cse1030*cse1186*cse14*d2g*state.xn[0] - cse1041*(-cse1089*cse1179 - cse684*cse689) - cse1081*cse684 - cse1083*cse1179 - cse1177 - cse1178 - cse1181 - 7.1777529600000012e-12*cse247 - 7.1777529600000012e-12*cse248 - 1.4355505920000002e-11*cse249*state.xn[3] - 7.1777529600000012e-12*cse535 + 6.0000000000000005e-15*cse695*d2g*state.xn[0] - cse733 - cse877 + (-cse1011*cse1194 - cse1012*cse1194 - cse1191*cse802 - cse1193*cse803)*state.xn[11] - ((cse833) ? (
   0
)
: (
   cse851*cse978
))*state.xn[3] - ((cse973) ? (
   cse1196*cse971 + cse1198*cse972 + cse1199*cse995 + cse1199*cse996
)
: (
   0
))*state.xn[7]) - 3.3475459999999996e-24*cse975;
  jac(19, 5) = cse1145*cse281 + cse66*(-cse1028*(-cse1189 - cse729*(cse1016*cse1017*cse1187 - cse1148*cse1188)) - cse1177 - cse1178 - cse1181 + cse1182*cse1202 + cse1184*cse1203 + 3.3999999999999999e-24*cse1201 + cse1204*cse1205 + cse1204*cse1206 - 7.1777529600000012e-12*cse279 - 7.1777529600000012e-12*cse542 + cse59*cse695) - 3.3466350616289996e-24*cse975;
  jac(19, 6) = 0.66666666666670005*cse1144*cse285*cse65 - cse1207 - 6.6464729999999996e-24*cse975;
  jac(19, 7) = 0.66666666666670005*cse1144*cse286*cse65 - cse1207 - 6.6455620616289996e-24*cse975;
  jac(19, 8) = 0.66666666666670005*cse1144*cse293*cse65 - cse66*cse974 - 1.9944730000000003e-23*cse975;
  jac(19, 9) = 0.66666666666670005*cse1144*cse294*cse65 - cse66*cse691*cse694 - 1.9943819061629003e-23*cse975;
  jac(19, 10) = cse1145*cse296 + cse66*(-cse1028*(cse1024*cse1025*cse1208*cse705 - cse728*(cse1024*cse1025*cse1208*cse709*cse721 - cse1209*cse722) - cse729*(cse1016*cse1017*cse1208 - cse1148*cse1209)) - cse705*cse732) - 4.6512360000000008e-23*cse975;
  jac(19, 11) = cse1145*cse297 + cse1210 - 4.8185222061629005e-23*cse975;
  jac(19, 12) = cse1145*cse305 - cse66*cse804 - 2.6567630000000005e-23*cse975;
  jac(19, 13) = cse1145*cse306 - 4.6637070000000001e-23*cse975;
  jac(19, 14) = cse1145*cse307 - 4.6636159061629004e-23*cse975;
  jac(19, 15) = cse1145*cse312 + cse1210 - 2.1618503000000003e-23*cse975;
  jac(19, 16) = cse1145*cse314 + cse1210 - 2.8241403000000005e-23*cse975;
  jac(19, 17) = cse1145*cse324 + cse66*(-cse1028*(cse1024*cse1025*cse1212*cse705 - cse728*(cse1024*cse1025*cse1212*cse709*cse721 - cse1213*cse722) - cse729*(cse1016*cse1017*cse1212 - cse1148*cse1213)) - cse1045*(300.0*cse1042 + 300.0*cse1180) - cse1052*(6*state.xn[0] + 6*state.xn[10] + 6*state.xn[14] + 6*state.xn[15] + 18*state.xn[16] + 6*state.xn[1] + 12*state.xn[3] + 12*state.xn[4]) - 5.6994379667262474e-29*cse1176 + 5.0999999999999995e-24*cse1201 + cse1202*((cse697) ? (
   0
)
: ((cse698) ? (
   (9.0/2.0)*cse1047
)
: ((cse699) ? (
   4*cse1047
)
: ((cse700) ? (
   cse1047
)
: (
   0
))))) + cse1203*(1.5*cse1056*cse1057*cse215 - 1.5*cse1183) + cse1205*cse1211 + cse1206*cse1211 - 7.1777529600000012e-12*cse322 + cse549*cse695 - 7.1777529600000012e-12*cse550) - 5.0204080616289998e-24*cse975;
  jac(19, 18) = cse1145*cse334 - 2.6566719061629008e-23*cse975;
  jac(19, 19) = cse1145;
  // $$JAFF END
}

AMREX_INLINE
void actual_rhs_init() {}

#endif
