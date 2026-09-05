#ifndef actual_rhs_H
#define actual_rhs_H

#include <AMReX_Array.H>
#include <AMReX_REAL.H>
#include <cmath>
#include <gow_interp.H>

#include <actual_network.H>
#include <burn_type.H>
#include <extern_parameters.H>
#include <jacobian_utilities.H>

using namespace ArrayUtil;
using namespace network_rp;

AMREX_GPU_HOST_DEVICE AMREX_INLINE void
actual_rhs(const burn_t &state, amrex::Array1D<amrex::Real, 1, neqs> &ydot) {

  using namespace Rates;
  const amrex::Real ncol_H2 = state.aux[0];
	const amrex::Real ncol_CO = state.aux[1];
	const amrex::Real ncol_C = state.aux[2];
	const amrex::Real ncol_H = state.aux[3];
	const amrex::Real d2g = network_rp::dust2gas_ratio;
	const amrex::Real gradv = 3.0 * 3.0e-14;
	const amrex::Real vdisp = 3.0 * 1.0e5 / pow(2, 0.5);


  // $JAFF REPEAT idx, rhs, cse IN rhses $[REPLACE radeden\[\s*(\d+)\s*\] state.rn[\1] REPLACE photden\[\s*(\d+)\s*\] state.rn[\1] REPLACE rflux\[\s*(\d+)\s*\] state.rn[2*\1+1] REPLACE nden\[\s*(\d+)\s*\] state.xn[\1] REPLACE tgas state.T REPLACE c_hat state.c_hat  RADIATION True]$
  const amrex::Real cse0 = 6.3999999999999996e-10*state.xn[0]*state.xn[4];
  const amrex::Real cse1 = -cse0;
  const amrex::Real cse2 = std::sqrt(state.T);
  const amrex::Real cse3 = cse2/state.xn[2];
  const amrex::Real cse4 = 0.00011320754716981131*cse3*state.xn[3];
  const amrex::Real cse5 = std::min(10000000000.0, cse4);
  const amrex::Real cse6 = 1.0/(cse5 + 1);
  const amrex::Real cse7 = cse6*state.xn[3]*state.xn[17];
  const amrex::Real cse8 = 1.6000000000000001e-9*cse7;
  const amrex::Real cse9 = cse5*cse8;
  const amrex::Real cse10 = 1.99e-9*std::pow(state.T, -0.19)*state.xn[11]*state.xn[16];
  const amrex::Real cse11 = cse10*cse6;
  const amrex::Real cse12 = 3.5000000000000002e-11*state.xn[11]*state.xn[15];
  const amrex::Real cse13 = state.c_hat*state.rn[0];
  const amrex::Real cse14 = 2.1301056248450629e-18*state.xn[15];
  const amrex::Real cse15 = cse13*cse14;
  const amrex::Real cse16 = 1.0/state.T;
  const amrex::Real cse17 = 7.9513696852843427e-10*std::pow(state.T, -0.33900000000000002)*std::exp(0.11*cse16)*state.xn[7]*state.xn[15];
  const amrex::Real cse18 = M_LN10;
  const amrex::Real cse19 = 1.0/cse18;
  const amrex::Real cse20 = 2*state.xn[3];
  const amrex::Real cse21 = cse20 + state.xn[0] + state.xn[1] + 2*state.xn[4] + state.xn[10] + state.xn[14] + state.xn[15] + 3*state.xn[16];
  const amrex::Real cse22 = std::log((1.0/10000.0)*state.T);
  const amrex::Real cse23 = std::pow(cse18, -2);
  const amrex::Real cse24 = std::pow(cse22, 2)*cse23;
  const amrex::Real cse25 = 1.0/cse21;
  const amrex::Real cse26 = cse25*state.xn[3];
  const amrex::Real cse27 = cse25*state.xn[0];
  const amrex::Real cse28 = std::pow(10, 0.41599999999999998*cse19*cse22 + 0.32700000000000001*cse24 - 3)*cse27 + std::pow(10, 1.3*cse19*cse22 - 1.6200000000000001*cse24 - 4.8449999999999998)*cse26;
  const amrex::Real cse29 = cse21*((cse28 < 1.0e-50) ? (
   1.0e-50
)
: (
   cse28
));
  const amrex::Real cse30 = cse19/(cse29 + 1);
  const amrex::Real cse31 = cse29*cse30;
  const amrex::Real cse32 = state.T > 700.0;
  const amrex::Real cse33 = ((cse32) ? (
   std::pow(10, cse30*std::log(5.9959999999999997e-30*std::pow(state.T, 4.1881000000000004)*std::pow(6.7610000000000002e-6*state.T + 1, -5.6881000000000004)*std::exp(-54657.400000000001*cse16)) + cse31*std::log(1.3000000000000001e-9*std::exp(-53300*cse16)))
)
: (
   0.0
))*std::pow(state.xn[3], 2);
  const amrex::Real cse34 = (7.5e-10*std::pow(state.T, -0.45800000000000002) + 4.9899999999999997e-11*std::pow(state.T, 0.40500000000000003))*state.xn[0]*state.xn[17];
  const amrex::Real cse35 = ((cse32) ? (
   std::pow(10, cse30*std::log(6.6699999999999996e-12*cse2*std::exp(-63590*cse16 - 1)) + cse31*std::log(3.5199999999999998e-9*std::exp(-43900*cse16)))
)
: (
   0.0
))*state.xn[0];
  const amrex::Real cse36 = std::pow(state.T, -0.52000000000000002);
  const amrex::Real cse37 = cse36*state.xn[2]*state.xn[16];
  const amrex::Real cse38 = cse21*d2g;
  const amrex::Real cse39 = (4.0000000000000001e-10*std::pow(state.T, 0.0066899999999999998) + 1.1000000000000001e-11*std::pow(state.T, 0.51700000000000002))*std::exp(-227*cse16)*state.xn[1]*state.xn[11];
  const amrex::Real cse40 = std::max(1.0e-35, 2.0000000000000002e-15*ncol_H2);
  const amrex::Real cse41 = 0.035000000000000003*std::pow(cse40 + 1, -0.5)*std::exp(-0.00084999999999999995*std::sqrt(cse40 + 1.0)) + 0.96499999999999997*std::pow(cse40/std::max(9.9999999999999995e-21, 1.4142135623730953e-5*vdisp) + 1.0, -2.0);
  const amrex::Real cse42 = cse13*cse41;
  const amrex::Real cse43 = cse42*state.xn[3];
  const amrex::Real cse44 = std::pow(state.T, -1.3);
  const amrex::Real cse45 = std::exp(-23*cse16);
  const amrex::Real cse46 = cse44*cse45*state.xn[3]*state.xn[8];
  const amrex::Real cse47 = 7.0000000000000004e-11*state.xn[11]*state.xn[14];
  const amrex::Real cse48 = 3.7026980154779576e-18*state.xn[14];
  const amrex::Real cse49 = cse13*cse48;
  const amrex::Real cse50 = cse47 + cse49;
  const amrex::Real cse51 = (26.218500000000002*std::sqrt(3)*std::sqrt(cse16) + 0.62)*state.xn[15];
  const amrex::Real cse52 = 1.3500000000000001e-9*cse51*state.xn[6];
  const amrex::Real cse53 = state.xn[3]*state.xn[6];
  const amrex::Real cse54 = 1.2599999999999999e-13*cse53*std::exp(-22.5*cse16);
  const amrex::Real cse55 = cse52 + cse54;
  const amrex::Real cse56 = 2.8142984982406731e-11*std::pow(state.T, 0.26000000000000001)*state.xn[0]*state.xn[14];
  const amrex::Real cse57 = 1.7599999999999999e-9*std::pow(state.T, 0.042000000000000003)*std::exp(-2.145922746781116e-5*state.T)*state.xn[3]*state.xn[4];
  const amrex::Real cse58 = -cse56 + cse57;
  const amrex::Real cse59 = crate*state.xn[9];
  const amrex::Real cse60 = 6.5199999999999996*cse59;
  const amrex::Real cse61 = crate*(2.2999999999999998*cse26 + 1.5*cse27);
  const amrex::Real cse62 = cse61*state.xn[0];
  const amrex::Real cse63 = std::log(8.6173000000000003e-5*state.T);
  const amrex::Real cse64 = state.xn[0]*state.xn[2];
  const amrex::Real cse65 = cse64*((cse32) ? (
   5.9078447946382801e-70*std::pow(state.T, 13.536555999999999)*std::exp(-2.0391498499999999e-6*std::pow(cse63, 8) + 0.000111954395*std::pow(cse63, 7) - 0.00263197617*std::pow(cse63, 6) + 0.034825597700000002*std::pow(cse63, 5) - 0.28770560000000001*std::pow(cse63, 4) + 1.56315498*std::pow(cse63, 3) - 5.7393287500000003*std::pow(cse63, 2))
)
: (
   0.0
));
  const amrex::Real cse66 = 1.05849862170431e-5*std::pow(state.T, -0.64000000000000001)*state.xn[2]*state.xn[10];
  const amrex::Real cse67 = 4.8813573300130879e-6*std::pow(cse16, 1.5)*std::pow(114.81344434488582*std::pow(cse16, 0.40699999999999997) + 1, -2.242)*state.xn[1]*state.xn[2];
  const amrex::Real cse68 = cse13*cse3;
  const amrex::Real cse69 = 8.9181112054588774e-9*cse68;
  const amrex::Real cse70 = std::log(state.T);
  const amrex::Real cse71 = 1.2250000000000001e-11*cse38*state.xn[1]/(6.5247635594176241e-17*std::pow(cse68, 1.3779999999999999)*(508.69999999999999*std::pow(cse69, -1.102e-5*cse70 - 0.4723)*std::pow(state.T, 0.015859999999999999) + 1) + 1);
  const amrex::Real cse72 = -cse60 - cse62 - cse65 + cse66 + cse67 + cse71;
  const amrex::Real cse73 = cse34 - cse39;
  const amrex::Real cse74 = cse20*cse61;
  const amrex::Real cse75 = 1.4595000290776864e-10*std::pow(state.T, -0.62)*state.xn[2]*state.xn[13];
  const amrex::Real cse76 = 2.1212529374176215e-17*state.xn[12];
  const amrex::Real cse77 = cse13*cse76;
  const amrex::Real cse78 = 2.1659999999999999e-12*cse38*state.xn[13]/(4.6666665696897683e-23*std::pow(cse68, 1.8740000000000001)*(43750.0*std::pow(cse69, -7.538e-5*cse70 - 0.89639999999999997)*std::pow(state.T, 1.635e-6) + 1) + 1);
  const amrex::Real cse79 = 16800.0*crate*cse26*state.xn[12];
  const amrex::Real cse80 = (1.0537934368240927e-9*std::pow(cse16, 0.00231) + std::pow(state.T, -1.5)*(0.00015100000000000001*std::exp(-8110.0*cse16) + 1.31e-7*std::exp(-26.600000000000001*cse16) + 3.4e-8*std::exp(-7.6200000000000001*cse16) + 6.9699999999999997e-9*std::exp(-1.3799999999999999*cse16)))*state.xn[7]*state.xn[16];
  const amrex::Real cse81 = cse25*state.xn[2];
  const amrex::Real cse82 = ((cse81 < 1.0e-50) ? (
   10000000000.0
)
: (
   cse4
));
  const amrex::Real cse83 = cse10*cse82/(cse82 + 1);
  const amrex::Real cse84 = cse11 + 1.3007021411393064e-6*cse37 + cse80 + cse83;
  const amrex::Real cse85 = cse19*cse70;
  const amrex::Real cse86 = -1.1000000000000001*crate*state.xn[5] + 5.5720000000000001e-12*cse38*state.xn[6]/(2.1474449357543363e-19*std::pow(cse68, 1.512)*(5115.0*std::pow(cse69, -5.4939999999999998e-7*cse70 - 0.49559999999999998)*std::pow(state.T, 3.9029999999999999e-7) + 1) + 1) + 9.9999999999999994e-12*std::pow(state.T, -0.5)*(cse85*(cse85*(0.044330000000000001*cse85 - 0.28520000000000001) - 1.6759999999999999) + 11.19)*state.xn[2]*state.xn[6];
  const amrex::Real cse87 = crate*state.xn[7];
  const amrex::Real cse88 = 3.8500000000000001*cse87;
  const amrex::Real cse89 = 0.15970000000000001*std::exp(-49550.0*cse16);
  const amrex::Real cse90 = ((1.6339999999999999e-6*std::exp(-15230.0*cse16) + 9.7930000000000004e-9*std::exp(-73.799999999999997*cse16) + 6.3460000000000002e-9*std::exp(-12.17*cse16))/std::pow(state.T, 3.0/2.0) + 2.4460185352936309e-10*std::pow(0.00071740367932898903*cse2 + 1, -cse89 - 1.7848999999999999)*std::pow(12.244387999457523*cse2 + 1, cse89 - 0.21509999999999996)/cse2)*state.xn[2]*state.xn[8];
  const amrex::Real cse91 = 4.5579999999999993e-11*cse38*state.xn[8]/(5.063534655349809e-12*std::pow(cse68, 1.1279999999999999)*(433.10000000000002*std::pow(cse69, -0.00013329999999999999*cse70 - 0.81200000000000006)*std::pow(state.T, 0.04845) + 1) + 1);
  const amrex::Real cse92 = 1040.0*cse26*cse87;
  const amrex::Real cse93 = 2.8636309100363227e-18*interp_8_shielding_H2(ncol_H2)*interp_8_shielding_self(ncol_C)*state.xn[7];
  const amrex::Real cse94 = cse13*cse93;
  const amrex::Real cse95 = -cse88 + cse90 + cse91 - cse92 - cse94;
  const amrex::Real cse96 = 3.2999999999999996e-13*cse46;
  const amrex::Real cse97 = cse8 + cse9;
  const amrex::Real cse98 = cse96 + cse97;
  const amrex::Real cse99 = -cse80;
  const amrex::Real cse100 = -cse83;
  const amrex::Real cse101 = 1.6999999999999999e-9*state.xn[9]*state.xn[16];
  const amrex::Real cse102 = cse35*state.xn[3];
  const amrex::Real cse103 = 6.2169970539621192e-19*cse42;
  const amrex::Real cse104 = 7.2000000000000002e-15*cse53;
  const amrex::Real cse105 = cse1 + cse104 + cse74;
  const amrex::Real cse106 = -cse57;
  const amrex::Real cse107 = 1.6000000000000001e-9*state.xn[6]*state.xn[9];
  const amrex::Real cse108 = cse104 + cse107 + cse55 + cse86;
  const amrex::Real cse109 = cse56 + cse99;
  const amrex::Real cse110 = 2.8416883703341718e-18*interp_10_shielding_H2(ncol_H2)*interp_10_shielding_self(ncol_CO)*state.xn[9];
  const amrex::Real cse111 = cse110*cse13;
  const amrex::Real cse112 = cse111 + cse49;
  const amrex::Real cse113 = 184.0*cse26*cse59;
  const amrex::Real cse114 = cse113 - cse17;
  const amrex::Real cse115 = 9.1500000000000005e-10*cse51*state.xn[8];
  const amrex::Real cse116 = cse101 + cse60 - cse66;
  const amrex::Real cse117 = cse107 + cse111 - cse47;
  const amrex::Real cse118 = cse100 + cse12 + cse15;
  const amrex::Real cse119 = -cse75 + cse77 - cse78 + cse79;
  const amrex::Real cse120 = std::pow(state.T, 0.14999999999999999)/(0.00088240812266906664*std::pow(state.T, 0.65000000000000002) + 1);
  const amrex::Real cse121 = cse120*state.xn[2];
  const amrex::Real cse122 = std::exp(-118400.0*cse16)/(4.4460389837719569e-9*cse121 + 626500000.0);
  const amrex::Real cse123 = cse69 + 50;
  const amrex::Real cse124 = std::log(8.9181112054588774e-9*cse13*cse2/(state.xn[2] + 1.0e-50) + 50);
  const amrex::Real cse125 = std::sqrt(cse16)*state.xn[2];
  const amrex::Real cse126 = (1.0/100.0)*state.T;
  const amrex::Real cse127 = std::log(cse126);
  const amrex::Real cse128 = state.T < 500;
  const amrex::Real cse129 = std::pow(state.T, 0.070000000000000007);
  const amrex::Real cse130 = 4.5299999999999998e-6*cse125 + 7.5799999999999997e-10*std::pow(cse126, 0.0086999999999999994*cse127 + 0.12809999999999999)*state.xn[0] + (0.75*((cse128) ? (
   1.1000000000000001e-13*state.T + 5.3300000000000002e-10
)
: (
   3.74757785025e-10*cse129
)) + 0.25*((cse128) ? (
   3.3000000000000001e-13*state.T + 4.4299999999999997e-10
)
: (
   3.8899728635600003e-10*cse129
)))*state.xn[3];
  const amrex::Real cse131 = cse130*std::exp(-91.209999999999994*cse16)/(cse130 + 2.3e-6);
  const amrex::Real cse132 = 0.001*state.T;
  const amrex::Real cse133 = cse132 + 1;
  const amrex::Real cse134 = std::pow(cse133, -2);
  const amrex::Real cse135 = 1.0/cse133;
  const amrex::Real cse136 = cse25/(std::pow(10, -2.3900000000000001*cse134 + 0.055500000000000001*cse135 - 11.06)*cse27 + std::pow(10, -2.0230000000000001*cse134 - 3.6709999999999998*cse135 - 11.08)*cse26);
  const amrex::Real cse137 = std::pow(state.T, 1.0/4.0);
  const amrex::Real cse138 = 1.75*cse137*state.xn[0] + state.xn[3] + 680.10000000000002*state.xn[2]/cse137;
  const amrex::Real cse139 = l0_co_interp1d(state.T);
  const amrex::Real cse140 = 1.0/cse139;
  const amrex::Real cse141 = ((ncol_H <= 0) ? (
   100000.0*state.xn[9]/std::max(gradv, 5.7445531974402559e-18*M_SQRT2*cse2)
)
: (
   1756701454.4310031*ncol_CO/std::sqrt(std::max(1.0000000000000001e-30, cse25*ncol_H))
));
  const amrex::Real cse142 = 1.0/llte_co_interp2d(state.T, cse141);
  const amrex::Real cse143 = nhalf_co_interp2d(state.T, cse141);
  const amrex::Real cse144 = cse19*std::log(cse21);
  const amrex::Real cse145 = 2.0*cse26*((cse144 < 2) ? (
   10.0
)
: ((cse144 < 4) ? (
   (3.0/2.0)*cse144 + 7
)
: ((cse144 < 7) ? (
   (4.0/3.0)*cse144 + 23.0/3.0
)
: ((cse144 < 10) ? (
   (1.0/3.0)*cse144 + 44.0/3.0
)
: (
   18.0
))))) + cse27*(26.399999999999999*std::sqrt(cse81/(cse81 + 0.070000000000000007)) + 6.5);
  const amrex::Real cse146 = 0.042999999999999997*cse127;
  const amrex::Real cse147 = 4.34e-10*std::pow(cse126, 0.755 - 0.16*cse127)*state.xn[0] + 1.08e-14*std::pow(state.T, 0.92600000000000005)*state.xn[2] + (2.2500000000000003e-12*std::pow(cse126, 0.52500000000000002*cse127 + 1.198) + 5.2499999999999998e-13*std::pow(cse126, cse146 + 0.88900000000000001))*state.xn[3];
  const amrex::Real cse148 = 3.5700000000000001e-10*std::pow(cse126, 0.41899999999999998 - 0.0030000000000000001*cse127)*state.xn[0] + 5.1199999999999999e-10*std::pow(state.T, -0.074999999999999997)*state.xn[2] + (3.7250000000000001e-11*std::pow(cse126, 0.025000000000000001*cse127 + 0.26400000000000001) + 1.0274999999999999e-10*std::pow(cse126, cse146 + 0.29599999999999999))*state.xn[3];
  const amrex::Real cse149 = (1.0/5.0)*std::exp(-326.60000000000002*cse16);
  const amrex::Real cse150 = 3.1899999999999998e-10*std::pow(cse126, 0.36899999999999999 - 0.0060000000000000001*cse127)*state.xn[0] + 4.8599999999999998e-10*std::pow(state.T, -0.025999999999999999)*state.xn[2] + (4.7499999999999998e-11*std::pow(cse126, 0.041000000000000002*cse127 + 0.20300000000000001) + 1.6725000000000002e-10*std::pow(cse126, 0.058000000000000003*cse127 + 0.23699999999999999))*state.xn[3];
  const amrex::Real cse151 = cse148 + 8.9099999999999997e-5;
  const amrex::Real cse152 = cse149*cse150;
  const amrex::Real cse153 = cse147*cse148*cse149 + (1.0/15.0)*cse147*cse150*std::exp(-425.40000000000003*cse16) + cse151*cse152;
  const amrex::Real cse154 = cse147 + 1.7499999999999998e-5;
  const amrex::Real cse155 = cse150 + 1.34e-10;
  const amrex::Real cse156 = (3.0/5.0)*cse148*std::exp(-227.80000000000001*cse16);
  const amrex::Real cse157 = cse152*cse154 + cse154*cse156 + cse155*cse156;
  const amrex::Real cse158 = 1.0/((1.0/3.0)*cse147*cse155*std::exp(-98.800000000000011*cse16) + cse151*cse154 + cse151*cse155 + cse153 + cse157);
  const amrex::Real cse159 = state.T < 100.0;
  const amrex::Real cse160 = state.T < 6000.0;
  const amrex::Real cse161 = state.T < 10;
  const amrex::Real cse162 = std::log(cse132);
  const amrex::Real cse163 = cse162*cse19;
  const amrex::Real cse164 = std::pow(cse18, -8);
  const amrex::Real cse165 = std::pow(cse162, 8)*cse164;
  const amrex::Real cse166 = std::pow(cse18, -7);
  const amrex::Real cse167 = std::pow(cse162, 7)*cse166;
  const amrex::Real cse168 = std::pow(cse18, -6);
  const amrex::Real cse169 = std::pow(cse162, 6)*cse168;
  const amrex::Real cse170 = std::pow(cse18, -5);
  const amrex::Real cse171 = std::pow(cse162, 5)*cse170;
  const amrex::Real cse172 = std::pow(cse18, -4);
  const amrex::Real cse173 = std::pow(cse162, 4)*cse172;
  const amrex::Real cse174 = std::pow(cse18, -3);
  const amrex::Real cse175 = std::pow(cse162, 3);
  const amrex::Real cse176 = cse174*cse175;
  const amrex::Real cse177 = std::pow(cse162, 2)*cse23;
  const amrex::Real cse178 = std::pow(10.0, 16.815729999999999*cse163 + 70.609154000000004*cse165 + 364.14445999999998*cse167 + 801.81246999999996*cse169 + 983.67575999999997*cse171 + 734.71650999999997*cse173 + 343.1918*cse176 + 96.743155000000002*cse177 - 21.928795999999998)*state.xn[2];
  const amrex::Real cse179 = state.xn[5] + state.xn[6];
  const amrex::Real cse180 = std::pow(10.0, 1.5714710999999999*cse163 + 0.32168730000000001*cse171 - 0.51002221000000003*cse173 - 0.23619984999999999*cse176 + 0.015391166*cse177 - 22.089523)*state.xn[1] + std::pow(10.0, 2.1892372*cse163 + 0.19191374999999999*cse171 - 0.16596184*cse173 + 0.29036281000000003*cse176 - 0.81520437999999995*cse177 - 23.689236999999999)*cse179 + std::pow(10.0, 2.0943374000000001*cse162*cse19 - 0.033638326000000003*cse171 - 0.14913216000000001*cse173 + 0.43693353000000001*cse174*cse175 - 0.77151435999999995*cse177 - 23.962112000000001)*state.xn[3];
  const amrex::Real cse181 = std::pow(10, 37.383713*cse163 + 3.8479610000000002*cse171 + 20.159831000000001*cse173 + 48.656103000000002*cse176 + 58.145166000000003*cse177 - 16.818342000000001)*state.xn[0] + cse180;
  const amrex::Real cse182 = std::pow(10.0, 1.6802758*cse163 - 6.3701156000000001*cse165 + 6.4380698000000001*cse167 + 8.9167182999999994*cse169 - 8.8077017000000009*cse171 - 4.7274035999999997*cse173 + 4.0406627000000004*cse176 + 0.93310621999999999*cse177 - 22.921188999999998)*state.xn[2];
  const amrex::Real cse183 = state.T < 1000.0;
  const amrex::Real cse184 = std::pow(10.0, 3.5692468000000002*cse162*cse19 - 4.2519023000000002*cse171 - 21.328264000000001*cse173 - 27.850082*cse176 - 11.33286*cse177 - 24.311209000000002)*state.xn[0] + cse180;
  const amrex::Real cse185 = std::pow(10.0, 4.6450521*cse163 + 1.5538288*cse171 - 5.5108049000000001*cse173 + 5.9369081000000001*cse176 - 3.7209846*cse177 - 24.311209000000002)*state.xn[0] + cse180;
  const amrex::Real cse186 = std::pow(10.0, -0.62120210234156448*cse170 - 1.5370575959566981*cse172 + 2.5133588394664677*cse174 + 3.7525488682084647*cse19 - 2.4768712409536775*cse23 - 23.962112000000001)*state.xn[3] + std::pow(10.0, 3.5440891133599637*cse170 - 1.7105157385968939*cse172 + 1.6702447513374001*cse174 + 3.9225864834863131*cse19 - 2.6171337683481011*cse23 - 23.689236999999999)*cse179 + std::pow(10.0, 5.9406293599919788*cse170 - 5.2566362076906961*cse172 - 1.3586848802337363*cse174 + 2.8156982240432273*cse19 + 0.049411830040524525*cse23 - 22.089523)*state.xn[1] + std::pow(10.0, 28.69470131298657*cse170 - 56.79810800956944*cse172 + 34.150687525022562*cse174 + 8.3228160852326614*cse19 - 11.945856385319289*cse23 - 24.311209000000002)*state.xn[0];
  const amrex::Real cse187 = ((cse160) ? (
   cse16
)
: (
   0.00016666666666666666
));
  const amrex::Real cse188 = 1.6e-18*std::exp(-11700.0*cse187) + 6.7e-19*std::exp(-5860.0*cse187) + 3.0e-24*std::exp(-510.0*cse187) + 4.985670872372847e-33*((cse160) ? (
   std::pow(state.T, 3.7599999999999998)
)
: (
   160638152785302.53
))*std::exp(-2197000.0*((cse160) ? (
   std::pow(state.T, -3)
)
: (
   4.6296296296296294e-12
)))/(6.0142468035272636e-8*((cse160) ? (
   std::pow(state.T, 2.1000000000000001)
)
: (
   85924621.118090734
)) + 1.0);
  const amrex::Real cse189 = std::log(0.001*((cse160) ? (
   state.T
)
: (
   6000.0
)));
  const amrex::Real cse190 = cse170*std::pow(cse189, 5);
  const amrex::Real cse191 = cse172*std::pow(cse189, 4);
  const amrex::Real cse192 = std::pow(cse189, 3);
  const amrex::Real cse193 = std::pow(cse189, 2)*cse23;
  const amrex::Real cse194 = cse189*cse19;
  const amrex::Real cse195 = cse174*cse192;
  const amrex::Real cse196 = cse164*std::pow(cse189, 8);
  const amrex::Real cse197 = cse166*std::pow(cse189, 7);
  const amrex::Real cse198 = cse168*std::pow(cse189, 6);
  const amrex::Real cse199 = 0.045999999999999999*cse127;
  const amrex::Real cse200 = 1.7258000000000001e-6*cse125;
  const amrex::Real cse201 = 8.9000000000000003e-11*std::pow(cse126, cse199 + 0.22800000000000001)*state.xn[0] + cse200*std::exp(((cse183) ? (
   cse70*(cse70*(cse70*(0.0070527699999999999*cse70 - 0.11133800000000001) + 0.69763799999999998) - 1.3074300000000001) - 7.6973500000000001
)
: (
   cse70*(cse70*(cse70*(0.093813800000000003*cse70 - 3.0328300000000001) + 36.180300000000003) - 187.47399999999999) + 350.60899999999998
))) + (5.1749999999999996e-11*std::pow(cse126, 0.037999999999999999*cse127 + 0.16900000000000001) + 2.15e-11*std::pow(cse126, 0.048000000000000001*cse127 - 0.01))*state.xn[3];
  const amrex::Real cse202 = 2.6400000000000002e-10*std::pow(cse126, cse199 + 0.23100000000000001)*state.xn[0] + cse200*std::exp(((cse183) ? (
   cse70*(cse70*(cse70*(0.00235272*cse70 - 0.041816600000000002) + 0.35826400000000003) - 0.57443) - 7.4386999999999999
)
: (
   cse70*(cse70*(cse70*(0.097857299999999994*cse70 - 3.1926800000000002) + 38.504899999999999) - 202.19300000000001) + 386.18599999999998
))) + (1.11e-10*std::pow(cse126, 0.031*cse127 + 0.26300000000000001) + 4.3749999999999998e-11*std::pow(cse126, 0.064000000000000001*cse127 + 0.071999999999999995))*state.xn[3];
  const amrex::Real cse203 = 2.8763333333333333e-6*cse125*std::exp(((cse183) ? (
   cse70*(cse70*(cse70*(-0.00065632500000000003*cse70 - 0.015089200000000001) + 0.36118400000000001) - 0.77378199999999997) - 9.2514099999999999
)
: (
   cse70*(cse70*(cse70*(0.10508000000000001*cse70 - 3.4762) + 42.594999999999999) - 227.91300000000001) + 444.60000000000002
))) + 1.26e-10*std::pow(cse126, 0.057000000000000002*cse127 + 0.115)*state.xn[0] + (5.3249999999999997e-11*std::pow(cse126, 0.049000000000000002*cse127 - 0.0040000000000000001) + 1.675e-11*std::pow(cse126, 0.10199999999999999*cse127 - 0.085000000000000006))*state.xn[3];
  const amrex::Real cse204 = 5*std::exp(-62.460000000000001*cse16);
  const amrex::Real cse205 = cse203 + 7.8800000000000004e-8;
  const amrex::Real cse206 = cse201*cse204;
  const amrex::Real cse207 = (25.0/3.0)*cse201*cse202*std::exp(-101.30000000000001*cse16) + cse202*cse203*cse204 + cse205*cse206;
  const amrex::Real cse208 = cse202 + 2.65e-7;
  const amrex::Real cse209 = cse201 + 1.81e-14;
  const amrex::Real cse210 = 3*cse203*std::exp(-23.619999999999997*cse16);
  const amrex::Real cse211 = cse206*cse208 + cse208*cse210 + cse209*cse210;
  const amrex::Real cse212 = 1.0/((5.0/3.0)*cse202*cse209*std::exp(-38.840000000000003*cse16) + cse205*cse208 + cse205*cse209 + cse207 + cse211);
  const amrex::Real cse213 = 1.3890778379475547e-21*Zd*cse21;
  const amrex::Real cse214 = state.c_hat*state.rn[2*0+1];

  ydot(1) = cse1 + cse11 + cse12 + cse15 + cse17 + cse20*cse35 + 2*cse33 - cse34 + 2.9935562710997168e-6*cse37 - 6.0000000000000005e-15*cse38*state.xn[0] + cse39 + 1.2433994107924238e-18*cse43 + 4.2899999999999994e-13*cse46 + cse50 + cse55 + cse58 + 3.2000000000000001e-9*cse7 + cse72 + cse9;
  ydot(2) = cse0 + cse54 + cse62 + cse65 - cse67 - cse71 + cse73;
  ydot(3) = -cse72 + cse74 - cse75 + cse77 - cse78 + cse79 - cse84 - cse86 - cse95 - cse98;
  ydot(4) = -cse100 + cse101 - cse102 - cse103*state.xn[3] - cse105 + cse11 + 3.0000000000000002e-15*cse21*d2g*state.xn[0] - cse33 + 4.5427507615910105e-7*cse36*state.xn[2]*state.xn[16] - cse54 - cse58 - cse98 - cse99;
  ydot(5) = cse105 + cse106;
  ydot(6) = cse108;
  ydot(7) = -cse108;
  ydot(8) = cse109 + cse112 + cse114 + 9.8999999999999995e-14*cse46 + cse95;
  ydot(9) = cse107 - cse115 + cse88 - cse90 - cse91 + cse92 + cse94 - cse96;
  ydot(10) = -cse114 - cse116 - cse117;
  ydot(11) = cse115 + cse116;
  ydot(12) = cse113 + cse117 + cse118 + cse73 + cse8;
  ydot(13) = -cse119;
  ydot(14) = cse119;
  ydot(15) = -cse109 + 2.3099999999999997e-13*cse44*cse45*state.xn[3]*state.xn[8] - cse50;
  ydot(16) = -cse115 - cse118 - cse17 - cse52 + cse9;
  ydot(17) = -cse101 - cse106 - cse84;
  ydot(18) = cse52 - cse73 - cse97;
  ydot(19) = 8.9181112054588776e-33*state.c_hat*cse21*d2g*(2.25*std::pow(state.T, 0.14699999999999999) + 5.2199999999999998)*state.rn[0]/(0.049959999999999997*std::pow(cse123, 0.43099999999999999)*(0.0043*std::pow(cse123, 0.69199999999999995) + 1) + 1) + 6.2169970539621192e-19*state.c_hat*cse41*(6.4087080000000003e-13 + 2.5634832e-11/(cse136*(cse103 + 1.9999999999999999e-7) + 1))*state.xn[3]*state.rn[0] + 1.602177e-12*crate*cse145*state.xn[0] + 3.2043540000000001e-12*crate*cse145*state.xn[3] - 7.1777529600000012e-12*cse102 - 1.3659996292638161e-10*cse120*cse122*cse64/(1.3338116951315871e-8*cse121*cse122 + 1) - 5.7927337833400002e-20*cse131*state.xn[8]/(2*cse131 + 1) - cse138*std::pow(1 - std::exp(-state.T), 1000)*state.xn[9]/(cse138*cse142 + cse140*std::pow(cse138/cse143, alpha_co_interp2d(state.T, cse141))*(-cse139*cse142*cse143 + 1) + cse140) + 3.0000000000000002e-15*cse21*d2g*(3.2043540000000002e-13 + 6.7291434000000005e-12/(cse136*(3.0000000000000002e-15*cse38/state.xn[0] + 1.9999999999999999e-7) + 1))*state.xn[0] - 7.1777529600000012e-12*cse33 - 1.8998126555754159e-29*cse38*std::pow(state.T, 0.45350000000000001 + 2.234/cse124)*std::exp(cse124*(1.4419999999999999 - 0.050889999999999998*cse124))*state.xn[2] - 2.1789607199999999e-11*cse65 - (2.3872025442750958e-19*cse153*cse158 + 2.8023005140020001e-18*cse157*cse158)*state.xn[11] - ((((cse128 && cse159 && cse160) ? (
   cse161 || cse178 + cse181 < 1.0e-100
)
: (
   ((cse159 && cse160) ? (
      cse161 || cse181 + cse182 < 1.0e-100
   )
   : (
      ((cse128 && cse160 && cse183) ? (
         cse161 || cse178 + cse184 < 1.0e-100
      )
      : (
         ((cse160 && cse183) ? (
            cse161 || cse182 + cse184 < 1.0e-100
         )
         : (
            ((cse128 && cse160) ? (
               cse161 || cse178 + cse185 < 1.0e-100
            )
            : (
               ((cse160) ? (
                  cse161 || cse182 + cse185 < 1.0e-100
               )
               : (
                  ((cse128) ? (
                     cse161 || std::pow(10.0, 7500.646099484944*cse164 + 21588.957206746247*cse166 + 26530.843406310876*cse168 + 18165.631967965237*cse170 + 7572.4886742758581*cse172 + 1974.131269262874*cse174 + 30.129743459482278*cse19 + 310.58441786958315*cse23 - 21.928795999999998)*state.xn[2] + cse186 < 1.0e-100
                  )
                  : (
                     cse161 || std::pow(10.0, -676.68255490510751*cse164 + 381.69251128040054*cse166 + 295.04162851880631*cse168 - 162.65264843551881*cse170 - 48.723840736518795*cse172 + 23.242975457496804*cse174 + 3.0106500755647456*cse19 + 2.9956460707652877*cse23 - 22.921188999999998)*state.xn[2] + cse186 < 1.0e-100
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
   cse188/(cse188/(std::pow(10.0, 0.19191374999999999*cse190 - 0.16596184*cse191 - 0.81520437999999995*cse193 + 2.1892372*cse194 + 0.29036281000000003*cse195 - 23.689236999999999)*cse179 + std::pow(10.0, 0.32168730000000001*cse190 - 0.51002221000000003*cse191 + 0.015391166*cse193 + 1.5714710999999999*cse194 - 0.23619984999999999*cse195 - 22.089523)*state.xn[1] + std::pow(10.0, 0.43693353000000001*cse174*cse192 + 2.0943374000000001*cse189*cse19 - 0.033638326000000003*cse190 - 0.14913216000000001*cse191 - 0.77151435999999995*cse193 - 23.962112000000001)*state.xn[3] + ((cse128) ? (
      std::pow(10.0, 983.67575999999997*cse190 + 734.71650999999997*cse191 + 96.743155000000002*cse193 + 16.815729999999999*cse194 + 343.1918*cse195 + 70.609154000000004*cse196 + 364.14445999999998*cse197 + 801.81246999999996*cse198 - 21.928795999999998)
   )
   : (
      std::pow(10.0, -8.8077017000000009*cse190 - 4.7274035999999997*cse191 + 0.93310621999999999*cse193 + 1.6802758*cse194 + 4.0406627000000004*cse195 - 6.3701156000000001*cse196 + 6.4380698000000001*cse197 + 8.9167182999999994*cse198 - 22.921188999999998)
   ))*state.xn[2] + ((((cse160) ? (
      cse159
   )
   : (
      false
   ))) ? (
      std::pow(10, 3.8479610000000002*cse190 + 20.159831000000001*cse191 + 58.145166000000003*cse193 + 37.383713*cse194 + 48.656103000000002*cse195 - 16.818342000000001)
   )
   : ((((cse160) ? (
      cse183
   )
   : (
      false
   ))) ? (
      std::pow(10.0, 3.5692468000000002*cse189*cse19 - 4.2519023000000002*cse190 - 21.328264000000001*cse191 - 11.33286*cse193 - 27.850082*cse195 - 24.311209000000002)
   )
   : (
      std::pow(10.0, 1.5538288*cse190 - 5.5108049000000001*cse191 - 3.7209846*cse193 + 4.6450521*cse194 + 5.9369081000000001*cse195 - 24.311209000000002)
   )))*state.xn[0]) + 1)
))*state.xn[3] - ((state.T < 1000000.0) ? (
   1.4210469458259591e-21*cse207*cse212 + 2.5697412351439998e-22*cse211*cse212
)
: (
   0.0
))*state.xn[7] - std::min(100.0*d2g*psigd_coll_interp2d(cse21, state.T), 3.2000000000000002e-32*cse2*cse38*(state.T - 2.1544346900318838*std::pow(0.00045101070015594175*cse13 + 4.1397668473788896, 1.0/6.0)));
  ydot(20) = -cse112 - cse13*cse213 - cse15 - 4.5072610463162814e-18*cse43 - cse77 - cse94;
  ydot(21) = -cse110*cse214 - cse14*cse214 - cse213*cse214 - 4.5072610463162814e-18*cse214*cse41*state.xn[3] - cse214*cse48 - cse214*cse76 - cse214*cse93;
  // $JAFF END
  // $JAFF SUB nspec
  ydot(19) /= state.rho;
  // $JAFF END
}

template <class MatrixType>
AMREX_GPU_HOST_DEVICE AMREX_INLINE void actual_jac(const burn_t &state,
                                                   MatrixType &jac) {
jac.zero();
const amrex::Real ncol_H2 = state.aux[0];
	const amrex::Real ncol_CO = state.aux[1];
	const amrex::Real ncol_C = state.aux[2];
	const amrex::Real ncol_H = state.aux[3];
	const amrex::Real d2g = network_rp::dust2gas_ratio;
	const amrex::Real gradv = 3.0 * 3.0e-14;
	const amrex::Real vdisp = 3.0 * 1.0e5 / pow(2, 0.5);


  // $JAFF REPEAT idx, expr, cse IN jacobian $[REPLACE radeden\[\s*(\d+)\s*\] state.rn[\1] REPLACE photden\[\s*(\d+)\s*\] state.rn[\1] REPLACE rflux\[\s*(\d+)\s*\] state.rn[2*\1+1] REPLACE nden\[\s*(\d+)\s*\] state.xn[\1] REPLACE tgas state.T REPLACE c_hat state.c_hat RADIATION True USE_DEDT True]$
  const amrex::Real cse0 = 2.8142984982406731e-11*std::pow(state.T, 0.26000000000000001);
  const amrex::Real cse1 = cse0*state.xn[14];
  const amrex::Real cse2 = std::sqrt(state.T);
  const amrex::Real cse3 = 1.0/state.T;
  const amrex::Real cse4 = 63590*cse3 + 1;
  const amrex::Real cse5 = std::exp(-cse4);
  const amrex::Real cse6 = std::log(6.6699999999999996e-12*cse2*cse5);
  const amrex::Real cse7 = M_LN10;
  const amrex::Real cse8 = 1.0/cse7;
  const amrex::Real cse9 = 2*state.xn[3];
  const amrex::Real cse10 = cse9 + state.xn[0] + state.xn[1] + state.xn[10] + state.xn[14] + state.xn[15] + 3*state.xn[16] + 2*state.xn[4];
  const amrex::Real cse11 = 1.0/cse10;
  const amrex::Real cse12 = std::log((1.0/10000.0)*state.T);
  const amrex::Real cse13 = std::pow(cse7, -2);
  const amrex::Real cse14 = std::pow(cse12, 2)*cse13;
  const amrex::Real cse15 = std::pow(10, 1.3*cse12*cse8 - 1.6200000000000001*cse14 - 4.8449999999999998);
  const amrex::Real cse16 = cse11*cse15*state.xn[3];
  const amrex::Real cse17 = std::pow(10, 0.41599999999999998*cse12*cse8 + 0.32700000000000001*cse14 - 3);
  const amrex::Real cse18 = cse11*cse17*state.xn[0];
  const amrex::Real cse19 = cse16 + cse18;
  const amrex::Real cse20 = cse19 < 1.0e-50;
  const amrex::Real cse21 = ((cse20) ? (
   1.0e-50
)
: (
   cse19
));
  const amrex::Real cse22 = cse10*cse21;
  const amrex::Real cse23 = cse22 + 1;
  const amrex::Real cse24 = cse8/cse23;
  const amrex::Real cse25 = std::log(3.5199999999999998e-9*std::exp(-43900*cse3));
  const amrex::Real cse26 = cse22*cse24;
  const amrex::Real cse27 = std::pow(10, cse24*cse6 + cse25*cse26);
  const amrex::Real cse28 = state.T > 700.0;
  const amrex::Real cse29 = ((cse28) ? (
   cse27
)
: (
   0.0
));
  const amrex::Real cse30 = std::pow(state.xn[3], 2);
  const amrex::Real cse31 = std::log(1.3000000000000001e-9*std::exp(-53300*cse3));
  const amrex::Real cse32 = cse21*cse24;
  const amrex::Real cse33 = cse31*cse32;
  const amrex::Real cse34 = std::pow(cse10, 2);
  const amrex::Real cse35 = 1.0/cse34;
  const amrex::Real cse36 = cse35*state.xn[3];
  const amrex::Real cse37 = cse15*cse36;
  const amrex::Real cse38 = cse35*state.xn[0];
  const amrex::Real cse39 = cse17*cse38;
  const amrex::Real cse40 = cse37 + cse39;
  const amrex::Real cse41 = cse10*((cse20) ? (
   0
)
: (
   cse11*cse17 - cse40
));
  const amrex::Real cse42 = cse24*cse41;
  const amrex::Real cse43 = std::pow(state.T, 4.1881000000000004);
  const amrex::Real cse44 = 6.7610000000000002e-6*state.T + 1;
  const amrex::Real cse45 = std::pow(cse44, 5.6881000000000004);
  const amrex::Real cse46 = 54657.400000000001*cse3;
  const amrex::Real cse47 = std::exp(-cse46);
  const amrex::Real cse48 = cse47/cse45;
  const amrex::Real cse49 = std::log(5.9959999999999997e-30*cse43*cse48);
  const amrex::Real cse50 = cse8/std::pow(cse23, 2);
  const amrex::Real cse51 = cse50*(-cse21 - cse41);
  const amrex::Real cse52 = cse22*cse51;
  const amrex::Real cse53 = std::pow(10, cse24*cse49 + cse26*cse31);
  const amrex::Real cse54 = cse53*cse7;
  const amrex::Real cse55 = ((cse28) ? (
   cse54*(cse31*cse42 + cse31*cse52 + cse33 + cse49*cse51)
)
: (
   0
));
  const amrex::Real cse56 = 6.0000000000000005e-15*d2g;
  const amrex::Real cse57 = cse56*state.xn[0];
  const amrex::Real cse58 = cse25*cse32;
  const amrex::Real cse59 = cse27*cse7;
  const amrex::Real cse60 = ((cse28) ? (
   cse59*(cse25*cse42 + cse25*cse52 + cse51*cse6 + cse58)
)
: (
   0
));
  const amrex::Real cse61 = 6.3999999999999996e-10*state.xn[4];
  const amrex::Real cse62 = 7.5e-10*std::pow(state.T, -0.45800000000000002) + 4.9899999999999997e-11*std::pow(state.T, 0.40500000000000003);
  const amrex::Real cse63 = cse62*state.xn[17];
  const amrex::Real cse64 = state.c_hat*state.rn[0];
  const amrex::Real cse65 = 1.0/state.xn[2];
  const amrex::Real cse66 = cse2*cse65;
  const amrex::Real cse67 = cse64*cse66;
  const amrex::Real cse68 = 8.9181112054588774e-9*cse67;
  const amrex::Real cse69 = std::log(state.T);
  const amrex::Real cse70 = 1.102e-5*cse69 + 0.4723;
  const amrex::Real cse71 = std::pow(cse68, -cse70);
  const amrex::Real cse72 = cse71*std::pow(state.T, 0.015859999999999999);
  const amrex::Real cse73 = 508.69999999999999*cse72;
  const amrex::Real cse74 = cse73 + 1;
  const amrex::Real cse75 = std::pow(cse67, 1.3779999999999999);
  const amrex::Real cse76 = 6.5247635594176241e-17*cse75;
  const amrex::Real cse77 = cse74*cse76 + 1;
  const amrex::Real cse78 = 1.0/cse77;
  const amrex::Real cse79 = 1.2250000000000001e-11*cse78*d2g;
  const amrex::Real cse80 = cse79*state.xn[1];
  const amrex::Real cse81 = -cse80;
  const amrex::Real cse82 = crate*(1.5*cse11*state.xn[0] + 2.2999999999999998*cse11*state.xn[3]);
  const amrex::Real cse83 = 2.2999999999999998*cse36 + 1.5*cse38;
  const amrex::Real cse84 = 1.5*cse11 - cse83;
  const amrex::Real cse85 = crate*state.xn[0];
  const amrex::Real cse86 = cse84*cse85;
  const amrex::Real cse87 = std::log(8.6173000000000003e-5*state.T);
  const amrex::Real cse88 = std::pow(cse87, 2);
  const amrex::Real cse89 = std::pow(cse87, 3);
  const amrex::Real cse90 = std::pow(cse87, 4);
  const amrex::Real cse91 = std::pow(cse87, 5);
  const amrex::Real cse92 = std::pow(cse87, 6);
  const amrex::Real cse93 = std::pow(cse87, 7);
  const amrex::Real cse94 = std::exp(-2.0391498499999999e-6*std::pow(cse87, 8) - 5.7393287500000003*cse88 + 1.56315498*cse89 - 0.28770560000000001*cse90 + 0.034825597700000002*cse91 - 0.00263197617*cse92 + 0.000111954395*cse93);
  const amrex::Real cse95 = 5.9078447946382801e-70*cse94*std::pow(state.T, 13.536555999999999);
  const amrex::Real cse96 = ((cse28) ? (
   cse95
)
: (
   0.0
));
  const amrex::Real cse97 = cse96*state.xn[2];
  const amrex::Real cse98 = cse61 + cse63 + cse81 + cse82 + cse86 + cse97;
  const amrex::Real cse99 = std::exp(-227*cse3);
  const amrex::Real cse100 = cse99*(4.0000000000000001e-10*std::pow(state.T, 0.0066899999999999998) + 1.1000000000000001e-11*std::pow(state.T, 0.51700000000000002));
  const amrex::Real cse101 = cse100*state.xn[11];
  const amrex::Real cse102 = std::pow(cse3, 1.5);
  const amrex::Real cse103 = 114.81344434488582*std::pow(cse3, 0.40699999999999997) + 1;
  const amrex::Real cse104 = std::pow(cse103, -2.242);
  const amrex::Real cse105 = 4.8813573300130879e-6*cse102*cse104;
  const amrex::Real cse106 = -cse83;
  const amrex::Real cse107 = cse106*cse85;
  const amrex::Real cse108 = -cse107 + cse80;
  const amrex::Real cse109 = cse10*cse79 + cse105*state.xn[2] + cse108;
  const amrex::Real cse110 = cse101 + cse109;
  const amrex::Real cse111 = cse10*((cse20) ? (
   0
)
: (
   -cse40
));
  const amrex::Real cse112 = cse111*cse24;
  const amrex::Real cse113 = cse50*(-cse111 - cse21);
  const amrex::Real cse114 = cse113*cse22;
  const amrex::Real cse115 = cse30*((cse28) ? (
   cse54*(cse112*cse31 + cse113*cse49 + cse114*cse31 + cse33)
)
: (
   0
));
  const amrex::Real cse116 = -cse57;
  const amrex::Real cse117 = ((cse28) ? (
   cse59*(cse112*cse25 + cse113*cse6 + cse114*cse25 + cse58)
)
: (
   0
));
  const amrex::Real cse118 = cse9*state.xn[0];
  const amrex::Real cse119 = 2*cse115 + cse116 + cse117*cse118;
  const amrex::Real cse120 = std::pow(state.xn[2], 2);
  const amrex::Real cse121 = 1.0/cse120;
  const amrex::Real cse122 = 0.00011320754716981131*cse66;
  const amrex::Real cse123 = cse122*state.xn[3];
  const amrex::Real cse124 = std::min(10000000000.0, cse123);
  const amrex::Real cse125 = cse124 + 1;
  const amrex::Real cse126 = std::pow(cse125, -2);
  const amrex::Real cse127 = (((cse123 - 10000000000.0 > 0) ? (
   0
)
: ((cse123 - 10000000000.0 == 0) ? (
   1.0/2.0
)
: (
   1
))));
  const amrex::Real cse128 = 2.2528301886792451e-13*std::pow(state.T, 0.31);
  const amrex::Real cse129 = cse121*cse126*cse127*cse128*state.xn[11]*state.xn[16]*state.xn[3];
  const amrex::Real cse130 = std::pow(state.T, -0.52000000000000002);
  const amrex::Real cse131 = 2.9935562710997168e-6*cse130;
  const amrex::Real cse132 = cse121*cse2;
  const amrex::Real cse133 = cse127*state.xn[17];
  const amrex::Real cse134 = cse133*cse30;
  const amrex::Real cse135 = cse132*cse134;
  const amrex::Real cse136 = 3.6226415094339619e-13*cse126;
  const amrex::Real cse137 = 1.0/cse125;
  const amrex::Real cse138 = 1.811320754716981e-13*cse137;
  const amrex::Real cse139 = cse135*cse138;
  const amrex::Real cse140 = 1.811320754716981e-13*cse126;
  const amrex::Real cse141 = cse135*cse140;
  const amrex::Real cse142 = cse124*cse141;
  const amrex::Real cse143 = -cse139 + cse142;
  const amrex::Real cse144 = 1.05849862170431e-5*std::pow(state.T, -0.64000000000000001);
  const amrex::Real cse145 = cse144*state.xn[10];
  const amrex::Real cse146 = cse96*state.xn[0];
  const amrex::Real cse147 = cse74*cse75;
  const amrex::Real cse148 = 8.9911241848774851e-17*cse147;
  const amrex::Real cse149 = -cse70;
  const amrex::Real cse150 = 3.319147222675745e-14*cse149*cse72*cse75;
  const amrex::Real cse151 = d2g*state.xn[1];
  const amrex::Real cse152 = 1.2250000000000001e-11*cse10*cse151/std::pow(cse77, 2);
  const amrex::Real cse153 = cse105*state.xn[1] - cse146 + cse152*(cse148*cse65 + cse150*cse65);
  const amrex::Real cse154 = cse145 + cse153;
  const amrex::Real cse155 = std::exp(-22.5*cse3);
  const amrex::Real cse156 = 1.2599999999999999e-13*cse155;
  const amrex::Real cse157 = cse156*state.xn[6];
  const amrex::Real cse158 = std::exp(-2.145922746781116e-5*state.T);
  const amrex::Real cse159 = cse158*state.xn[4];
  const amrex::Real cse160 = std::pow(state.T, 0.042000000000000003);
  const amrex::Real cse161 = 1.7599999999999999e-9*cse160;
  const amrex::Real cse162 = cse159*cse161;
  const amrex::Real cse163 = d2g*state.xn[0];
  const amrex::Real cse164 = -1.2000000000000001e-14*cse163;
  const amrex::Real cse165 = cse29*state.xn[0];
  const amrex::Real cse166 = 2*cse21;
  const amrex::Real cse167 = cse24*cse31;
  const amrex::Real cse168 = cse166*cse167;
  const amrex::Real cse169 = cse35*cse9;
  const amrex::Real cse170 = cse15*cse169 + 2*cse39;
  const amrex::Real cse171 = cse10*((cse20) ? (
   0
)
: (
   cse11*cse15 - cse170
));
  const amrex::Real cse172 = cse50*(-cse166 - cse171);
  const amrex::Real cse173 = cse172*cse22;
  const amrex::Real cse174 = cse30*((cse28) ? (
   cse54*(cse167*cse171 + cse168 + cse172*cse49 + cse173*cse31)
)
: (
   0
));
  const amrex::Real cse175 = ((cse28) ? (
   cse53
)
: (
   0.0
));
  const amrex::Real cse176 = cse175*state.xn[3];
  const amrex::Real cse177 = 3.2000000000000001e-9*cse137;
  const amrex::Real cse178 = cse24*cse25;
  const amrex::Real cse179 = cse166*cse178;
  const amrex::Real cse180 = ((cse28) ? (
   cse59*(cse171*cse178 + cse172*cse6 + cse173*cse25 + cse179)
)
: (
   0
));
  const amrex::Real cse181 = std::max(1.0e-35, 2.0000000000000002e-15*ncol_H2);
  const amrex::Real cse182 = 0.035000000000000003*std::pow(cse181 + 1, -0.5)*std::exp(-0.00084999999999999995*std::sqrt(cse181 + 1.0)) + 0.96499999999999997*std::pow(cse181/std::max(9.9999999999999995e-21, 1.4142135623730953e-5*vdisp) + 1.0, -2.0);
  const amrex::Real cse183 = 1.2433994107924238e-18*cse182;
  const amrex::Real cse184 = std::exp(-23*cse3);
  const amrex::Real cse185 = cse184*std::pow(state.T, -1.3);
  const amrex::Real cse186 = 4.2899999999999994e-13*cse185;
  const amrex::Real cse187 = cse133*cse66*state.xn[3];
  const amrex::Real cse188 = 1.6000000000000001e-9*cse137;
  const amrex::Real cse189 = cse188*state.xn[17];
  const amrex::Real cse190 = cse140*cse187;
  const amrex::Real cse191 = cse124*cse189 - cse124*cse190 + cse138*cse187;
  const amrex::Real cse192 = cse151*cse78;
  const amrex::Real cse193 = 2.4500000000000003e-11*cse192;
  const amrex::Real cse194 = 4.5999999999999996*cse36 + 3.0*cse38;
  const amrex::Real cse195 = 2.2999999999999998*cse11 - cse194;
  const amrex::Real cse196 = cse195*cse85;
  const amrex::Real cse197 = cse126*cse127*cse65*state.xn[11]*state.xn[16];
  const amrex::Real cse198 = cse128*cse197;
  const amrex::Real cse199 = cse193 - cse196 - cse198;
  const amrex::Real cse200 = cse10*((cse20) ? (
   0
)
: (
   -cse170
));
  const amrex::Real cse201 = cse50*(-cse166 - cse200);
  const amrex::Real cse202 = cse201*cse22;
  const amrex::Real cse203 = cse30*((cse28) ? (
   cse54*(cse167*cse200 + cse168 + cse201*cse49 + cse202*cse31)
)
: (
   0
));
  const amrex::Real cse204 = ((cse28) ? (
   cse59*(cse178*cse200 + cse179 + cse201*cse6 + cse202*cse25)
)
: (
   0
));
  const amrex::Real cse205 = cse158*cse161*state.xn[3];
  const amrex::Real cse206 = 6.3999999999999996e-10*state.xn[0];
  const amrex::Real cse207 = cse205 - cse206;
  const amrex::Real cse208 = -cse194;
  const amrex::Real cse209 = cse208*cse85;
  const amrex::Real cse210 = cse193 - cse209;
  const amrex::Real cse211 = cse156*state.xn[3];
  const amrex::Real cse212 = 26.218500000000002*std::sqrt(3)*std::sqrt(cse3) + 0.62;
  const amrex::Real cse213 = 1.3500000000000001e-9*cse212;
  const amrex::Real cse214 = cse213*state.xn[15];
  const amrex::Real cse215 = cse211 + cse214;
  const amrex::Real cse216 = std::exp(0.11*cse3);
  const amrex::Real cse217 = cse216*state.xn[15];
  const amrex::Real cse218 = 7.9513696852843427e-10*std::pow(state.T, -0.33900000000000002);
  const amrex::Real cse219 = cse217*cse218;
  const amrex::Real cse220 = 6.5199999999999996*crate;
  const amrex::Real cse221 = cse144*state.xn[2];
  const amrex::Real cse222 = cse108 + cse221;
  const amrex::Real cse223 = cse137*state.xn[16];
  const amrex::Real cse224 = 1.99e-9*std::pow(state.T, -0.19);
  const amrex::Real cse225 = cse223*cse224;
  const amrex::Real cse226 = 3.5000000000000002e-11*state.xn[15];
  const amrex::Real cse227 = 7.0000000000000004e-11*state.xn[14];
  const amrex::Real cse228 = cse100*state.xn[1];
  const amrex::Real cse229 = cse227 + cse228;
  const amrex::Real cse230 = 7.0000000000000004e-11*state.xn[11];
  const amrex::Real cse231 = 3.7026980154779576e-18*cse64;
  const amrex::Real cse232 = cse0*state.xn[0];
  const amrex::Real cse233 = -cse232;
  const amrex::Real cse234 = cse108 + cse119;
  const amrex::Real cse235 = cse213*state.xn[6];
  const amrex::Real cse236 = cse216*cse218*state.xn[7];
  const amrex::Real cse237 = 2.1301056248450629e-18*cse64;
  const amrex::Real cse238 = cse237 + 3.5000000000000002e-11*state.xn[11];
  const amrex::Real cse239 = cse235 + cse236 + cse238;
  const amrex::Real cse240 = 3*cse21;
  const amrex::Real cse241 = cse10*((cse20) ? (
   0
)
: (
   -3*cse37 - 3*cse39
));
  const amrex::Real cse242 = cse50*(-cse240 - cse241);
  const amrex::Real cse243 = cse22*cse242;
  const amrex::Real cse244 = cse30*((cse28) ? (
   cse54*(cse167*cse240 + cse167*cse241 + cse242*cse49 + cse243*cse31)
)
: (
   0
));
  const amrex::Real cse245 = ((cse28) ? (
   cse59*(cse178*cse240 + cse178*cse241 + cse242*cse6 + cse243*cse25)
)
: (
   0
));
  const amrex::Real cse246 = cse224*state.xn[11];
  const amrex::Real cse247 = cse137*cse246;
  const amrex::Real cse248 = -6.8999999999999995*cse36 - 4.5*cse38;
  const amrex::Real cse249 = cse248*cse85;
  const amrex::Real cse250 = 3.6750000000000002e-11*cse192;
  const amrex::Real cse251 = cse247 - cse249 + cse250;
  const amrex::Real cse252 = cse62*state.xn[0];
  const amrex::Real cse253 = cse188*state.xn[3];
  const amrex::Real cse254 = cse124*cse253;
  const amrex::Real cse255 = std::pow(state.T, -2);
  const amrex::Real cse256 = 2.8349999999999999e-12*cse155*cse255*state.xn[3]*state.xn[6];
  const amrex::Real cse257 = -cse256;
  const amrex::Real cse258 = 7.3171760954257497e-12*std::pow(state.T, -0.73999999999999999)*state.xn[0]*state.xn[14];
  const amrex::Real cse259 = std::sqrt(cse3);
  const amrex::Real cse260 = std::sqrt(3)*cse259*cse3*state.xn[15];
  const amrex::Real cse261 = 1.7697487500000004e-8*cse260*state.xn[6];
  const amrex::Real cse262 = cse255*cse26;
  const amrex::Real cse263 = cse13*cse3;
  const amrex::Real cse264 = cse12*cse263;
  const amrex::Real cse265 = cse3*cse8;
  const amrex::Real cse266 = ((cse20) ? (
   0
)
: (
   cse16*cse7*(-3.2400000000000002*cse264 + 1.3*cse3*cse8) + cse18*cse7*(0.65400000000000003*cse264 + 0.41599999999999998*cse265)
));
  const amrex::Real cse267 = cse10*cse266;
  const amrex::Real cse268 = cse267*cse50;
  const amrex::Real cse269 = cse21*cse266*cse34*cse50;
  const amrex::Real cse270 = ((cse28) ? (
   cse54*(cse167*cse267 + 1.6677785190126752e+29*cse24*cse45*(-2.3058963562359999e-34*cse43*std::pow(cse44, -6.6881000000000004)*cse47 + 3.2772577039999999e-25*cse48*std::pow(state.T, 2.1881000000000004) + 2.51118476e-29*cse48*std::pow(state.T, 3.1881000000000004))*std::exp(cse46)/cse43 + 53300.0*cse262 - cse268*cse49 - cse269*cse31)
)
: (
   0
));
  const amrex::Real cse271 = state.xn[0]*state.xn[17]*(-3.435e-10*std::pow(state.T, -1.458) + 2.0209499999999999e-11*std::pow(state.T, -0.59499999999999997));
  const amrex::Real cse272 = 1.0/cse2;
  const amrex::Real cse273 = std::pow(state.T, -3.0/2.0);
  const amrex::Real cse274 = ((cse28) ? (
   cse59*(cse178*cse267 + 149925037481.25937*cse24*cse272*(3.3349999999999998e-12*cse272*cse5 + 4.2414529999999999e-7*cse273*cse5)*std::exp(cse4) - cse25*cse269 + 43900.0*cse262 - cse268*cse6)
)
: (
   0
));
  const amrex::Real cse275 = std::pow(state.T, -1.52)*state.xn[16]*state.xn[2];
  const amrex::Real cse276 = cse99*state.xn[1]*state.xn[11]*(2.6759999999999999e-12*std::pow(state.T, -0.99331000000000003) + 5.6870000000000003e-12*std::pow(state.T, -0.48299999999999998));
  const amrex::Real cse277 = std::pow(state.T, -2.2999999999999998);
  const amrex::Real cse278 = cse184*state.xn[3]*state.xn[8];
  const amrex::Real cse279 = cse277*cse278;
  const amrex::Real cse280 = cse159*state.xn[3];
  const amrex::Real cse281 = 3.7768240343347639e-14*cse160*cse280;
  const amrex::Real cse282 = 7.3920000000000002e-11*cse280*std::pow(state.T, -0.95799999999999996);
  const amrex::Real cse283 = std::pow(state.T, -3.2999999999999998);
  const amrex::Real cse284 = 227*cse101*cse255*state.xn[1];
  const amrex::Real cse285 = cse272*cse65;
  const amrex::Real cse286 = cse134*cse285;
  const amrex::Real cse287 = 6.7743911789075847e-6*std::pow(state.T, -1.6400000000000001)*state.xn[10]*state.xn[2];
  const amrex::Real cse288 = cse217*state.xn[7];
  const amrex::Real cse289 = 8.7465066538127767e-11*cse288*std::pow(state.T, -2.339) + 2.6955143233113923e-10*cse288*std::pow(state.T, -1.339);
  const amrex::Real cse290 = cse287 + cse289;
  const amrex::Real cse291 = state.xn[0]*state.xn[2];
  const amrex::Real cse292 = cse291*((cse28) ? (
   7.9971871901929577e-69*cse94*std::pow(state.T, 12.536555999999999) + cse95*(-11.478657500000001*cse3*cse87 + 4.6894649399999997*cse3*cse88 - 1.1508224*cse3*cse89 + 0.1741279885*cse3*cse90 - 0.015791857020000001*cse3*cse91 + 0.00078368076500000001*cse3*cse92 - 1.6313198799999999e-5*cse3*cse93)
)
: (
   0
));
  const amrex::Real cse293 = cse3*state.xn[1]*state.xn[2];
  const amrex::Real cse294 = 7.3220359950196314e-6*cse102*cse104*cse293;
  const amrex::Real cse295 = 0.0005114031087522854*std::pow(cse103, -3.242)*cse293*std::pow(cse3, 1.907);
  const amrex::Real cse296 = 0.5*cse3;
  const amrex::Real cse297 = cse3*std::log(cse68);
  const amrex::Real cse298 = cse152*(-4.4955620924387425e-17*cse147*cse3 - cse76*(8.0679819999999989*cse71*std::pow(state.T, -0.98414000000000001) + cse73*(cse149*cse296 - 1.102e-5*cse297)));
  const amrex::Real cse299 = 9.0566037735849048e-14*cse286;
  const amrex::Real cse300 = cse137*cse299;
  const amrex::Real cse301 = cse126*cse299;
  const amrex::Real cse302 = cse124*cse301;
  const amrex::Real cse303 = -cse300 + cse302;
  const amrex::Real cse304 = 3.7810000000000002e-10*std::pow(state.T, -1.1899999999999999)*state.xn[11];
  const amrex::Real cse305 = 1.1264150943396225e-13*cse197*std::pow(state.T, -0.68999999999999995)*state.xn[3] + cse223*cse304;
  const amrex::Real cse306 = cse292 + cse294 - cse295 - cse298 + cse303 + cse305;
  const amrex::Real cse307 = 0.66666666666670005*(1.6737729999999998e-24*state.xn[0] + 1.6728620616289998e-24*state.xn[1] + 9.1093837099999994e-28*state.xn[2] + 3.3475459999999996e-24*state.xn[3] + 3.3466350616289996e-24*state.xn[4] + 6.6464729999999996e-24*state.xn[5] + 6.6455620616289996e-24*state.xn[6] + 1.9944730000000003e-23*state.xn[7] + 1.9943819061629003e-23*state.xn[8] + 4.6512360000000008e-23*state.xn[9] + 4.8185222061629005e-23*state.xn[10] + 2.6567630000000005e-23*state.xn[11] + 4.6637070000000001e-23*state.xn[12] + 4.6636159061629004e-23*state.xn[13] + 2.1618503000000003e-23*state.xn[14] + 2.8241403000000005e-23*state.xn[15] + 5.0204080616289998e-24*state.xn[16] + 2.6566719061629008e-23*state.xn[17])/(1.380649e-16*state.xn[0] + 1.380649e-16*state.xn[1] + 1.380649e-16*state.xn[2] + 1.380649e-16*state.xn[3] + 1.380649e-16*state.xn[4] + 1.380649e-16*state.xn[5] + 1.380649e-16*state.xn[6] + 1.380649e-16*state.xn[7] + 1.380649e-16*state.xn[8] + 1.380649e-16*state.xn[9] + 1.380649e-16*state.xn[10] + 1.380649e-16*state.xn[11] + 1.380649e-16*state.xn[12] + 1.380649e-16*state.xn[13] + 1.380649e-16*state.xn[14] + 1.380649e-16*state.xn[15] + 1.380649e-16*state.xn[16] + 1.380649e-16*state.xn[17]);
  const amrex::Real cse308 = 3.7026980154779576e-18*state.c_hat;
  const amrex::Real cse309 = cse308*state.xn[14];
  const amrex::Real cse310 = 2.1301056248450629e-18*state.c_hat;
  const amrex::Real cse311 = cse310*state.xn[15];
  const amrex::Real cse312 = state.c_hat*state.xn[3];
  const amrex::Real cse313 = 1.0/state.rn[0];
  const amrex::Real cse314 = cse152*(-cse148*cse313 - cse150*cse313);
  const amrex::Real cse315 = -cse193;
  const amrex::Real cse316 = cse107 + cse81;
  const amrex::Real cse317 = -cse271 + cse276 + cse284;
  const amrex::Real cse318 = crate*cse9;
  const amrex::Real cse319 = cse318*cse84;
  const amrex::Real cse320 = 5.4939999999999998e-7*cse69 + 0.49559999999999998;
  const amrex::Real cse321 = std::pow(cse68, -cse320);
  const amrex::Real cse322 = cse321*std::pow(state.T, 3.9029999999999999e-7);
  const amrex::Real cse323 = 5115.0*cse322;
  const amrex::Real cse324 = cse323 + 1;
  const amrex::Real cse325 = std::pow(cse67, 1.512);
  const amrex::Real cse326 = 2.1474449357543363e-19*cse325;
  const amrex::Real cse327 = cse324*cse326 + 1;
  const amrex::Real cse328 = 1.0/cse327;
  const amrex::Real cse329 = cse328*d2g*state.xn[6];
  const amrex::Real cse330 = 5.5720000000000001e-12*cse329;
  const amrex::Real cse331 = crate*cse36;
  const amrex::Real cse332 = cse331*state.xn[7];
  const amrex::Real cse333 = 0.00013329999999999999*cse69 + 0.81200000000000006;
  const amrex::Real cse334 = std::pow(cse68, -cse333);
  const amrex::Real cse335 = cse334*std::pow(state.T, 0.04845);
  const amrex::Real cse336 = 433.10000000000002*cse335;
  const amrex::Real cse337 = cse336 + 1;
  const amrex::Real cse338 = std::pow(cse67, 1.1279999999999999);
  const amrex::Real cse339 = 5.063534655349809e-12*cse338;
  const amrex::Real cse340 = cse337*cse339 + 1;
  const amrex::Real cse341 = 1.0/cse340;
  const amrex::Real cse342 = cse341*d2g*state.xn[8];
  const amrex::Real cse343 = 1040.0*cse332 + 4.5579999999999993e-11*cse342;
  const amrex::Real cse344 = cse331*state.xn[12];
  const amrex::Real cse345 = 7.538e-5*cse69 + 0.89639999999999997;
  const amrex::Real cse346 = std::pow(cse68, -cse345);
  const amrex::Real cse347 = cse346*std::pow(state.T, 1.635e-6);
  const amrex::Real cse348 = 43750.0*cse347;
  const amrex::Real cse349 = cse348 + 1;
  const amrex::Real cse350 = std::pow(cse67, 1.8740000000000001);
  const amrex::Real cse351 = 4.6666665696897683e-23*cse350;
  const amrex::Real cse352 = cse349*cse351 + 1;
  const amrex::Real cse353 = 1.0/cse352;
  const amrex::Real cse354 = cse353*d2g*state.xn[13];
  const amrex::Real cse355 = 16800.0*cse344 + 2.1659999999999999e-12*cse354;
  const amrex::Real cse356 = cse330 + cse343 + cse355;
  const amrex::Real cse357 = cse106*cse318;
  const amrex::Real cse358 = cse356 - cse357;
  const amrex::Real cse359 = cse69*cse8;
  const amrex::Real cse360 = 0.044330000000000001*cse359 - 0.28520000000000001;
  const amrex::Real cse361 = cse359*cse360 - 1.6759999999999999;
  const amrex::Real cse362 = cse359*cse361 + 11.19;
  const amrex::Real cse363 = cse362*state.xn[6];
  const amrex::Real cse364 = 9.9999999999999994e-12*std::pow(state.T, -0.5);
  const amrex::Real cse365 = cse324*cse325;
  const amrex::Real cse366 = 3.2469367428605564e-19*cse365;
  const amrex::Real cse367 = -cse320;
  const amrex::Real cse368 = 1.0984180846383429e-15*cse322*cse325*cse367;
  const amrex::Real cse369 = cse10*d2g;
  const amrex::Real cse370 = 5.5720000000000001e-12*cse369;
  const amrex::Real cse371 = cse370*state.xn[6]/std::pow(cse327, 2);
  const amrex::Real cse372 = cse363*cse364 + cse371*(cse366*cse65 + cse368*cse65);
  const amrex::Real cse373 = std::exp(-15230.0*cse3);
  const amrex::Real cse374 = std::exp(-73.799999999999997*cse3);
  const amrex::Real cse375 = std::exp(-12.17*cse3);
  const amrex::Real cse376 = 1.6339999999999999e-6*cse373 + 9.7930000000000004e-9*cse374 + 6.3460000000000002e-9*cse375;
  const amrex::Real cse377 = 0.00071740367932898903*cse2 + 1;
  const amrex::Real cse378 = std::exp(-49550.0*cse3);
  const amrex::Real cse379 = 0.15970000000000001*cse378;
  const amrex::Real cse380 = cse379 + 1.7848999999999999;
  const amrex::Real cse381 = 12.244387999457523*cse2 + 1;
  const amrex::Real cse382 = 0.21509999999999996 - cse379;
  const amrex::Real cse383 = std::pow(cse377, -cse380)*std::pow(cse381, -cse382);
  const amrex::Real cse384 = 2.4460185352936309e-10*cse272*cse383;
  const amrex::Real cse385 = cse273*cse376 + cse384;
  const amrex::Real cse386 = cse337*cse338;
  const amrex::Real cse387 = 5.7116670912345843e-12*cse386;
  const amrex::Real cse388 = -cse333;
  const amrex::Real cse389 = 2.1930168592320025e-9*cse335*cse338*cse388;
  const amrex::Real cse390 = 4.5579999999999993e-11*cse369;
  const amrex::Real cse391 = cse390*state.xn[8]/std::pow(cse340, 2);
  const amrex::Real cse392 = cse385*state.xn[8] + cse391*(cse387*cse65 + cse389*cse65);
  const amrex::Real cse393 = 1.4595000290776864e-10*std::pow(state.T, -0.62);
  const amrex::Real cse394 = cse349*cse350;
  const amrex::Real cse395 = 8.745333151598626e-23*cse394;
  const amrex::Real cse396 = -cse345;
  const amrex::Real cse397 = 2.0416666242392735e-18*cse347*cse350*cse396;
  const amrex::Real cse398 = 2.1659999999999999e-12*cse369;
  const amrex::Real cse399 = cse398*state.xn[13]/std::pow(cse352, 2);
  const amrex::Real cse400 = cse393*state.xn[13] + cse399*(cse395*cse65 + cse397*cse65);
  const amrex::Real cse401 = cse141 + cse143;
  const amrex::Real cse402 = 1.3007021411393064e-6*cse130;
  const amrex::Real cse403 = cse11*state.xn[2];
  const amrex::Real cse404 = cse403 < 1.0e-50;
  const amrex::Real cse405 = ((cse404) ? (
   0
)
: (
   -0.00011320754716981131*cse132*state.xn[3]
));
  const amrex::Real cse406 = ((cse404) ? (
   10000000000.0
)
: (
   cse123
));
  const amrex::Real cse407 = cse406 + 1;
  const amrex::Real cse408 = 1.0/cse407;
  const amrex::Real cse409 = cse246*state.xn[16];
  const amrex::Real cse410 = cse408*cse409;
  const amrex::Real cse411 = cse405*cse410;
  const amrex::Real cse412 = cse406*cse409/std::pow(cse407, 2);
  const amrex::Real cse413 = cse405*cse412;
  const amrex::Real cse414 = cse411 - cse413;
  const amrex::Real cse415 = cse129 + cse414;
  const amrex::Real cse416 = cse402*state.xn[16] + cse415;
  const amrex::Real cse417 = 1.1144e-11*cse329;
  const amrex::Real cse418 = 2*cse82;
  const amrex::Real cse419 = cse195*cse318;
  const amrex::Real cse420 = crate*cse11;
  const amrex::Real cse421 = 16800.0*cse420;
  const amrex::Real cse422 = 33600.0*cse344 + 4.3319999999999997e-12*cse354;
  const amrex::Real cse423 = -cse421*state.xn[12] + cse422;
  const amrex::Real cse424 = 3.2999999999999996e-13*cse185;
  const amrex::Real cse425 = cse424*state.xn[8];
  const amrex::Real cse426 = 1040.0*cse420;
  const amrex::Real cse427 = -cse426*state.xn[7];
  const amrex::Real cse428 = 2080.0*cse332 + 9.1159999999999986e-11*cse342;
  const amrex::Real cse429 = cse425 + cse427 + cse428;
  const amrex::Real cse430 = ((cse404) ? (
   0
)
: (
   cse122
));
  const amrex::Real cse431 = cse410*cse430;
  const amrex::Real cse432 = cse412*cse430;
  const amrex::Real cse433 = cse431 - cse432;
  const amrex::Real cse434 = cse191 + cse433;
  const amrex::Real cse435 = cse189 - cse190;
  const amrex::Real cse436 = cse208*cse318;
  const amrex::Real cse437 = -cse436;
  const amrex::Real cse438 = 1.1000000000000001*crate;
  const amrex::Real cse439 = cse364*state.xn[2];
  const amrex::Real cse440 = cse328*cse370 + cse362*cse439;
  const amrex::Real cse441 = std::pow(cse3, 0.00231);
  const amrex::Real cse442 = std::pow(state.T, -1.5);
  const amrex::Real cse443 = std::exp(-8110.0*cse3);
  const amrex::Real cse444 = std::exp(-26.600000000000001*cse3);
  const amrex::Real cse445 = std::exp(-7.6200000000000001*cse3);
  const amrex::Real cse446 = std::exp(-1.3799999999999999*cse3);
  const amrex::Real cse447 = 0.00015100000000000001*cse443 + 1.31e-7*cse444 + 3.4e-8*cse445 + 6.9699999999999997e-9*cse446;
  const amrex::Real cse448 = 1.0537934368240927e-9*cse441 + cse442*cse447;
  const amrex::Real cse449 = cse448*state.xn[16];
  const amrex::Real cse450 = -cse449;
  const amrex::Real cse451 = 2.8636309100363227e-18*interp_8_shielding_H2(ncol_H2)*interp_8_shielding_self(ncol_C);
  const amrex::Real cse452 = cse451*cse64;
  const amrex::Real cse453 = 3.8500000000000001*crate + cse426*state.xn[3] + cse452;
  const amrex::Real cse454 = cse424*state.xn[3];
  const amrex::Real cse455 = cse341*cse390 + cse385*state.xn[2];
  const amrex::Real cse456 = cse454 + cse455;
  const amrex::Real cse457 = cse406*cse408;
  const amrex::Real cse458 = cse457*state.xn[16];
  const amrex::Real cse459 = cse224*cse458;
  const amrex::Real cse460 = cse225 + cse459;
  const amrex::Real cse461 = -cse460;
  const amrex::Real cse462 = 2.1212529374176215e-17*cse64;
  const amrex::Real cse463 = cse421*state.xn[3] + cse462;
  const amrex::Real cse464 = cse353*cse398 + cse393*state.xn[2];
  const amrex::Real cse465 = -cse464;
  const amrex::Real cse466 = -cse108 - cse358;
  const amrex::Real cse467 = cse246*cse457;
  const amrex::Real cse468 = cse448*state.xn[7];
  const amrex::Real cse469 = cse402*state.xn[2];
  const amrex::Real cse470 = cse248*cse318;
  const amrex::Real cse471 = -cse470;
  const amrex::Real cse472 = 1.6715999999999999e-11*cse329;
  const amrex::Real cse473 = 3120.0*cse332 + 1.3673999999999998e-10*cse342;
  const amrex::Real cse474 = 50400.0*cse344 + 6.4979999999999992e-12*cse354;
  const amrex::Real cse475 = cse253 + cse254;
  const amrex::Real cse476 = -cse475;
  const amrex::Real cse477 = state.xn[16]*state.xn[7]*(-2.4342628390636543e-12*cse3*cse441 + cse442*(1.2246100000000002*cse255*cse443 + 3.4846000000000001e-6*cse255*cse444 + 2.5908000000000001e-7*cse255*cse445 + 9.6185999999999982e-9*cse255*cse446) - 1.5*cse447*std::pow(state.T, -2.5));
  const amrex::Real cse478 = -cse477;
  const amrex::Real cse479 = 6.7636511339243933e-7*cse275;
  const amrex::Real cse480 = cse304*cse458;
  const amrex::Real cse481 = ((cse404) ? (
   0
)
: (
   5.6603773584905654e-5*cse285*state.xn[3]
));
  const amrex::Real cse482 = cse410*cse481;
  const amrex::Real cse483 = -cse482;
  const amrex::Real cse484 = cse412*cse481;
  const amrex::Real cse485 = 7913.1350000000002*cse255*cse378;
  const amrex::Real cse486 = state.xn[2]*state.xn[8]*(-1.2230092676468155e-10*cse273*cse383 + cse273*(0.024885819999999999*cse255*cse373 + 7.2272340000000005e-7*cse255*cse374 + 7.7230820000000007e-8*cse255*cse375) - 3.0/2.0*cse376/std::pow(state.T, 5.0/2.0) + cse384*(-0.00035870183966449452*cse272*cse380/cse377 - cse485*std::log(cse377)) + cse384*(-6.1221939997287613*cse272*cse382/cse381 + cse485*std::log(cse381)));
  const amrex::Real cse487 = 9.0489001802816564e-11*std::pow(state.T, -1.6200000000000001)*state.xn[13]*state.xn[2];
  const amrex::Real cse488 = 4.9999999999999997e-12*cse363*cse442*state.xn[2];
  const amrex::Real cse489 = cse278*cse283;
  const amrex::Real cse490 = 7.5899999999999998e-12*cse489;
  const amrex::Real cse491 = cse439*state.xn[6]*(cse265*cse361 + cse359*(0.044330000000000001*cse263*cse69 + cse265*cse360));
  const amrex::Real cse492 = cse399*(-4.372666575799313e-23*cse3*cse394 - cse351*(0.071531250000000005*cse346*std::pow(state.T, -0.99999836499999994) + cse348*(cse296*cse396 - 7.538e-5*cse297)));
  const amrex::Real cse493 = cse371*(-1.6234683714302782e-19*cse3*cse365 - cse326*(0.0019963845000000001*cse321*std::pow(state.T, -0.99999960970000001) + cse323*(cse296*cse367 - 5.4939999999999998e-7*cse297)));
  const amrex::Real cse494 = cse391*(-2.8558335456172922e-12*cse3*cse386 - cse339*(20.983695000000001*cse334*std::pow(state.T, -0.95155000000000001) + cse336*(cse296*cse388 - 0.00013329999999999999*cse297)));
  const amrex::Real cse495 = 2.1212529374176215e-17*state.c_hat;
  const amrex::Real cse496 = cse495*state.xn[12];
  const amrex::Real cse497 = cse399*(-cse313*cse395 - cse313*cse397);
  const amrex::Real cse498 = cse371*(-cse313*cse366 - cse313*cse368);
  const amrex::Real cse499 = state.c_hat*cse451*state.xn[7];
  const amrex::Real cse500 = cse391*(-cse313*cse387 - cse313*cse389);
  const amrex::Real cse501 = -cse499 + cse500;
  const amrex::Real cse502 = -3.0000000000000002e-15*d2g*state.xn[0];
  const amrex::Real cse503 = cse29*state.xn[3];
  const amrex::Real cse504 = cse30*cse55;
  const amrex::Real cse505 = -cse1;
  const amrex::Real cse506 = state.xn[0]*state.xn[3];
  const amrex::Real cse507 = cse506*cse60;
  const amrex::Real cse508 = cse319 - cse61;
  const amrex::Real cse509 = cse117*cse506;
  const amrex::Real cse510 = cse115 + cse357 + cse502 + cse509;
  const amrex::Real cse511 = -cse510;
  const amrex::Real cse512 = 4.5427507615910105e-7*cse130;
  const amrex::Real cse513 = 6.2169970539621192e-19*cse182;
  const amrex::Real cse514 = cse513*cse64;
  const amrex::Real cse515 = cse180*cse506;
  const amrex::Real cse516 = 7.2000000000000002e-15*state.xn[6];
  const amrex::Real cse517 = cse157 + cse516;
  const amrex::Real cse518 = cse418 + cse419;
  const amrex::Real cse519 = cse162 + cse198 - cse431 + cse432;
  const amrex::Real cse520 = cse191 + cse435;
  const amrex::Real cse521 = cse204*cse506;
  const amrex::Real cse522 = 7.2000000000000002e-15*state.xn[3];
  const amrex::Real cse523 = 1.6999999999999999e-9*state.xn[16];
  const amrex::Real cse524 = 9.0000000000000011e-15*cse163;
  const amrex::Real cse525 = cse245*cse506;
  const amrex::Real cse526 = 1.6999999999999999e-9*state.xn[9];
  const amrex::Real cse527 = cse247 + cse467 + cse468 + cse526;
  const amrex::Real cse528 = -cse258;
  const amrex::Real cse529 = cse270*cse30;
  const amrex::Real cse530 = cse274*cse506;
  const amrex::Real cse531 = -4.2899999999999994e-13*cse184*cse277*state.xn[3]*state.xn[8] + cse490;
  const amrex::Real cse532 = cse300 - cse302;
  const amrex::Real cse533 = -cse281 + cse282;
  const amrex::Real cse534 = cse305 + cse478 + cse480 + cse483 + cse484 + cse533;
  const amrex::Real cse535 = cse312*cse513;
  const amrex::Real cse536 = cse417 + cse517;
  const amrex::Real cse537 = 1.6000000000000001e-9*state.xn[9];
  const amrex::Real cse538 = cse215 + cse440 + cse522 + cse537;
  const amrex::Real cse539 = 1.6000000000000001e-9*state.xn[6];
  const amrex::Real cse540 = cse235 + cse330;
  const amrex::Real cse541 = -cse261;
  const amrex::Real cse542 = cse256 - cse488 + cse491 + cse493 + cse541;
  const amrex::Real cse543 = -cse330;
  const amrex::Real cse544 = cse331*state.xn[9];
  const amrex::Real cse545 = 184.0*cse544;
  const amrex::Real cse546 = -cse545;
  const amrex::Real cse547 = cse343 + cse546;
  const amrex::Real cse548 = 184.0*cse420;
  const amrex::Real cse549 = cse548*state.xn[9];
  const amrex::Real cse550 = 9.8999999999999995e-14*cse185;
  const amrex::Real cse551 = 368.0*cse544;
  const amrex::Real cse552 = -cse551;
  const amrex::Real cse553 = cse428 + cse552;
  const amrex::Real cse554 = 2.8416883703341718e-18*interp_10_shielding_H2(ncol_H2)*interp_10_shielding_self(ncol_CO);
  const amrex::Real cse555 = cse554*cse64;
  const amrex::Real cse556 = cse548*state.xn[3] + cse555;
  const amrex::Real cse557 = cse231 + cse232;
  const amrex::Real cse558 = 552.0*cse544;
  const amrex::Real cse559 = cse486 + cse494;
  const amrex::Real cse560 = state.c_hat*cse554*state.xn[9];
  const amrex::Real cse561 = -cse343;
  const amrex::Real cse562 = 9.1500000000000005e-10*cse212;
  const amrex::Real cse563 = cse562*state.xn[15];
  const amrex::Real cse564 = cse562*state.xn[8];
  const amrex::Real cse565 = 1.1994963750000002e-8*cse260*state.xn[8];
  const amrex::Real cse566 = -cse565;
  const amrex::Real cse567 = -cse549 + cse551;
  const amrex::Real cse568 = cse220 + cse523;
  const amrex::Real cse569 = cse539 + cse556;
  const amrex::Real cse570 = cse230 + cse545;
  const amrex::Real cse571 = -cse63;
  const amrex::Real cse572 = -cse226 + cse459;
  const amrex::Real cse573 = -cse480 + cse482 - cse484;
  const amrex::Real cse574 = cse301 + cse317;
  const amrex::Real cse575 = cse311 + cse560;
  const amrex::Real cse576 = -cse487 + cse492;
  const amrex::Real cse577 = cse496 - cse497;
  const amrex::Real cse578 = -cse355;
  const amrex::Real cse579 = 2.3099999999999997e-13*cse185;
  const amrex::Real cse580 = (1.0/100.0)*state.T;
  const amrex::Real cse581 = std::log(cse580);
  const amrex::Real cse582 = 0.0060000000000000001*cse581;
  const amrex::Real cse583 = std::pow(cse580, 0.36899999999999999 - cse582);
  const amrex::Real cse584 = std::pow(state.T, 0.92600000000000005);
  const amrex::Real cse585 = 1.08e-14*cse584;
  const amrex::Real cse586 = 0.16*cse581;
  const amrex::Real cse587 = std::pow(cse580, 0.755 - cse586);
  const amrex::Real cse588 = 4.34e-10*cse587;
  const amrex::Real cse589 = cse588*state.xn[0];
  const amrex::Real cse590 = 0.042999999999999997*cse581;
  const amrex::Real cse591 = 5.2499999999999998e-13*std::pow(cse580, cse590 + 0.88900000000000001);
  const amrex::Real cse592 = 0.52500000000000002*cse581;
  const amrex::Real cse593 = 2.2500000000000003e-12*std::pow(cse580, cse592 + 1.198);
  const amrex::Real cse594 = cse591 + cse593;
  const amrex::Real cse595 = cse585*state.xn[2] + cse589 + cse594*state.xn[3];
  const amrex::Real cse596 = std::exp(-425.40000000000003*cse3);
  const amrex::Real cse597 = cse595*cse596;
  const amrex::Real cse598 = std::exp(-326.60000000000002*cse3);
  const amrex::Real cse599 = std::pow(state.T, -0.025999999999999999);
  const amrex::Real cse600 = 4.8599999999999998e-10*cse599;
  const amrex::Real cse601 = 3.1899999999999998e-10*cse583;
  const amrex::Real cse602 = cse601*state.xn[0];
  const amrex::Real cse603 = 0.041000000000000002*cse581;
  const amrex::Real cse604 = 4.7499999999999998e-11*std::pow(cse580, cse603 + 0.20300000000000001);
  const amrex::Real cse605 = 0.058000000000000003*cse581;
  const amrex::Real cse606 = 1.6725000000000002e-10*std::pow(cse580, cse605 + 0.23699999999999999);
  const amrex::Real cse607 = cse604 + cse606;
  const amrex::Real cse608 = cse600*state.xn[2] + cse602 + cse607*state.xn[3];
  const amrex::Real cse609 = cse598*cse608;
  const amrex::Real cse610 = 0.0030000000000000001*cse581;
  const amrex::Real cse611 = std::pow(cse580, 0.41899999999999998 - cse610);
  const amrex::Real cse612 = 7.1400000000000007e-11*cse611;
  const amrex::Real cse613 = cse595*cse598;
  const amrex::Real cse614 = std::pow(state.T, -0.074999999999999997);
  const amrex::Real cse615 = 5.1199999999999999e-10*cse614;
  const amrex::Real cse616 = 3.5700000000000001e-10*cse611;
  const amrex::Real cse617 = cse616*state.xn[0];
  const amrex::Real cse618 = 0.025000000000000001*cse581;
  const amrex::Real cse619 = 3.7250000000000001e-11*std::pow(cse580, cse618 + 0.26400000000000001);
  const amrex::Real cse620 = 1.0274999999999999e-10*std::pow(cse580, cse590 + 0.29599999999999999);
  const amrex::Real cse621 = cse619 + cse620;
  const amrex::Real cse622 = cse615*state.xn[2] + cse617 + cse621*state.xn[3];
  const amrex::Real cse623 = cse598*cse622;
  const amrex::Real cse624 = 8.68e-11*cse587;
  const amrex::Real cse625 = cse596*cse608;
  const amrex::Real cse626 = cse622 + 8.9099999999999997e-5;
  const amrex::Real cse627 = cse598*cse626;
  const amrex::Real cse628 = 6.3800000000000002e-11*cse583;
  const amrex::Real cse629 = 2.1266666666666666e-11*cse583*cse597 + 2.8933333333333332e-11*cse587*cse625 + cse609*cse612 + cse612*cse613 + cse623*cse624 + cse627*cse628;
  const amrex::Real cse630 = cse595 + 1.7499999999999998e-5;
  const amrex::Real cse631 = cse608 + 1.34e-10;
  const amrex::Real cse632 = std::exp(-98.800000000000011*cse3);
  const amrex::Real cse633 = cse595*cse632;
  const amrex::Real cse634 = (1.0/3.0)*cse633;
  const amrex::Real cse635 = (1.0/5.0)*cse598;
  const amrex::Real cse636 = cse595*cse635;
  const amrex::Real cse637 = (1.0/15.0)*cse597;
  const amrex::Real cse638 = cse608*cse635;
  const amrex::Real cse639 = cse608*cse637 + cse622*cse636 + cse626*cse638;
  const amrex::Real cse640 = std::exp(-227.80000000000001*cse3);
  const amrex::Real cse641 = cse622*cse640;
  const amrex::Real cse642 = (3.0/5.0)*cse641;
  const amrex::Real cse643 = cse630*cse638 + cse630*cse642 + cse631*cse642;
  const amrex::Real cse644 = cse626*cse630 + cse626*cse631 + cse631*cse634 + cse639 + cse643;
  const amrex::Real cse645 = 1.0/cse644;
  const amrex::Real cse646 = 2.3872025442750958e-19*cse645;
  const amrex::Real cse647 = cse598*cse630;
  const amrex::Real cse648 = 2.1419999999999999e-10*cse611;
  const amrex::Real cse649 = cse631*cse640;
  const amrex::Real cse650 = cse630*cse640;
  const amrex::Real cse651 = 1.9139999999999998e-10*cse583*cse641 + 2.604e-10*cse587*cse641 + cse609*cse624 + cse628*cse647 + cse648*cse649 + cse648*cse650;
  const amrex::Real cse652 = 2.8023005140020001e-18*cse645;
  const amrex::Real cse653 = std::pow(cse644, -2);
  const amrex::Real cse654 = cse631*cse632;
  const amrex::Real cse655 = cse653*(-1.0633333333333333e-10*cse583*cse633 - 1.4466666666666666e-10*cse587*cse654 - cse588*cse626 - cse601*cse626 - cse616*cse630 - cse616*cse631 - cse629 - cse651);
  const amrex::Real cse656 = 2.3872025442750958e-19*cse639;
  const amrex::Real cse657 = 2.8023005140020001e-18*cse643;
  const amrex::Real cse658 = state.T < 100.0;
  const amrex::Real cse659 = state.T < 6000.0;
  const amrex::Real cse660 = state.T < 500;
  const amrex::Real cse661 = state.T < 10;
  const amrex::Real cse662 = 0.001*state.T;
  const amrex::Real cse663 = std::log(cse662);
  const amrex::Real cse664 = cse663*cse8;
  const amrex::Real cse665 = std::pow(cse7, -8);
  const amrex::Real cse666 = std::pow(cse663, 8)*cse665;
  const amrex::Real cse667 = std::pow(cse7, -7);
  const amrex::Real cse668 = std::pow(cse663, 7)*cse667;
  const amrex::Real cse669 = std::pow(cse7, -6);
  const amrex::Real cse670 = std::pow(cse663, 6)*cse669;
  const amrex::Real cse671 = std::pow(cse7, -5);
  const amrex::Real cse672 = std::pow(cse663, 5)*cse671;
  const amrex::Real cse673 = std::pow(cse7, -4);
  const amrex::Real cse674 = std::pow(cse663, 4)*cse673;
  const amrex::Real cse675 = std::pow(cse7, -3);
  const amrex::Real cse676 = std::pow(cse663, 3);
  const amrex::Real cse677 = cse675*cse676;
  const amrex::Real cse678 = cse13*std::pow(cse663, 2);
  const amrex::Real cse679 = std::pow(10.0, 16.815729999999999*cse664 + 70.609154000000004*cse666 + 364.14445999999998*cse668 + 801.81246999999996*cse670 + 983.67575999999997*cse672 + 734.71650999999997*cse674 + 343.1918*cse677 + 96.743155000000002*cse678 - 21.928795999999998)*state.xn[2];
  const amrex::Real cse680 = state.xn[5] + state.xn[6];
  const amrex::Real cse681 = std::pow(10.0, 1.5714710999999999*cse664 + 0.32168730000000001*cse672 - 0.51002221000000003*cse674 - 0.23619984999999999*cse677 + 0.015391166*cse678 - 22.089523)*state.xn[1] + std::pow(10.0, 2.1892372*cse664 + 0.19191374999999999*cse672 - 0.16596184*cse674 + 0.29036281000000003*cse677 - 0.81520437999999995*cse678 - 23.689236999999999)*cse680 + std::pow(10.0, 2.0943374000000001*cse663*cse8 - 0.033638326000000003*cse672 - 0.14913216000000001*cse674 + 0.43693353000000001*cse675*cse676 - 0.77151435999999995*cse678 - 23.962112000000001)*state.xn[3];
  const amrex::Real cse682 = std::pow(10, 37.383713*cse664 + 3.8479610000000002*cse672 + 20.159831000000001*cse674 + 48.656103000000002*cse677 + 58.145166000000003*cse678 - 16.818342000000001)*state.xn[0] + cse681;
  const amrex::Real cse683 = std::pow(10.0, 1.6802758*cse664 - 6.3701156000000001*cse666 + 6.4380698000000001*cse668 + 8.9167182999999994*cse670 - 8.8077017000000009*cse672 - 4.7274035999999997*cse674 + 4.0406627000000004*cse677 + 0.93310621999999999*cse678 - 22.921188999999998)*state.xn[2];
  const amrex::Real cse684 = state.T < 1000.0;
  const amrex::Real cse685 = std::pow(10.0, 3.5692468000000002*cse663*cse8 - 4.2519023000000002*cse672 - 21.328264000000001*cse674 - 27.850082*cse677 - 11.33286*cse678 - 24.311209000000002)*state.xn[0] + cse681;
  const amrex::Real cse686 = std::pow(10.0, 4.6450521*cse664 + 1.5538288*cse672 - 5.5108049000000001*cse674 + 5.9369081000000001*cse677 - 3.7209846*cse678 - 24.311209000000002)*state.xn[0] + cse681;
  const amrex::Real cse687 = std::pow(10.0, -11.945856385319289*cse13 + 28.69470131298657*cse671 - 56.79810800956944*cse673 + 34.150687525022562*cse675 + 8.3228160852326614*cse8 - 24.311209000000002)*state.xn[0] + std::pow(10.0, -2.6171337683481011*cse13 + 3.5440891133599637*cse671 - 1.7105157385968939*cse673 + 1.6702447513374001*cse675 + 3.9225864834863131*cse8 - 23.689236999999999)*cse680 + std::pow(10.0, -2.4768712409536775*cse13 - 0.62120210234156448*cse671 - 1.5370575959566981*cse673 + 2.5133588394664677*cse675 + 3.7525488682084647*cse8 - 23.962112000000001)*state.xn[3] + std::pow(10.0, 0.049411830040524525*cse13 + 5.9406293599919788*cse671 - 5.2566362076906961*cse673 - 1.3586848802337363*cse675 + 2.8156982240432273*cse8 - 22.089523)*state.xn[1];
  const amrex::Real cse688 = ((cse658 && cse659 && cse660) ? (
   cse661 || cse679 + cse682 < 1.0e-100
)
: (
   ((cse658 && cse659) ? (
      cse661 || cse682 + cse683 < 1.0e-100
   )
   : (
      ((cse659 && cse660 && cse684) ? (
         cse661 || cse679 + cse685 < 1.0e-100
      )
      : (
         ((cse659 && cse684) ? (
            cse661 || cse683 + cse685 < 1.0e-100
         )
         : (
            ((cse659 && cse660) ? (
               cse661 || cse679 + cse686 < 1.0e-100
            )
            : (
               ((cse659) ? (
                  cse661 || cse683 + cse686 < 1.0e-100
               )
               : (
                  ((cse660) ? (
                     cse661 || std::pow(10.0, 310.58441786958315*cse13 + 7500.646099484944*cse665 + 21588.957206746247*cse667 + 26530.843406310876*cse669 + 18165.631967965237*cse671 + 7572.4886742758581*cse673 + 1974.131269262874*cse675 + 30.129743459482278*cse8 - 21.928795999999998)*state.xn[2] + cse687 < 1.0e-100
                  )
                  : (
                     cse661 || std::pow(10.0, 2.9956460707652877*cse13 - 676.68255490510751*cse665 + 381.69251128040054*cse667 + 295.04162851880631*cse669 - 162.65264843551881*cse671 - 48.723840736518795*cse673 + 23.242975457496804*cse675 + 3.0106500755647456*cse8 - 22.921188999999998)*state.xn[2] + cse687 < 1.0e-100
                  ))
               ))
            ))
         ))
      ))
   ))
));
  const amrex::Real cse689 = std::log(0.001*((cse659) ? (
   state.T
)
: (
   6000.0
)));
  const amrex::Real cse690 = cse689*cse8;
  const amrex::Real cse691 = std::pow(cse689, 5);
  const amrex::Real cse692 = cse671*cse691;
  const amrex::Real cse693 = std::pow(cse689, 4);
  const amrex::Real cse694 = cse673*cse693;
  const amrex::Real cse695 = std::pow(cse689, 3);
  const amrex::Real cse696 = cse675*cse695;
  const amrex::Real cse697 = std::pow(cse689, 2);
  const amrex::Real cse698 = cse13*cse697;
  const amrex::Real cse699 = std::pow(10, 37.383713*cse690 + 3.8479610000000002*cse692 + 20.159831000000001*cse694 + 48.656103000000002*cse696 + 58.145166000000003*cse698 - 16.818342000000001);
  const amrex::Real cse700 = ((cse659) ? (
   cse658
)
: (
   false
));
  const amrex::Real cse701 = std::pow(10.0, 3.5692468000000002*cse689*cse8 - 4.2519023000000002*cse692 - 21.328264000000001*cse694 - 27.850082*cse696 - 11.33286*cse698 - 24.311209000000002);
  const amrex::Real cse702 = ((cse659) ? (
   cse684
)
: (
   false
));
  const amrex::Real cse703 = std::pow(10.0, 4.6450521*cse690 + 1.5538288*cse692 - 5.5108049000000001*cse694 + 5.9369081000000001*cse696 - 3.7209846*cse698 - 24.311209000000002);
  const amrex::Real cse704 = ((cse700) ? (
   cse699
)
: ((cse702) ? (
   cse701
)
: (
   cse703
)));
  const amrex::Real cse705 = std::pow(10.0, 0.43693353000000001*cse675*cse695 + 2.0943374000000001*cse689*cse8 - 0.033638326000000003*cse692 - 0.14913216000000001*cse694 - 0.77151435999999995*cse698 - 23.962112000000001);
  const amrex::Real cse706 = cse705*state.xn[3];
  const amrex::Real cse707 = std::pow(10.0, 1.5714710999999999*cse690 + 0.32168730000000001*cse692 - 0.51002221000000003*cse694 - 0.23619984999999999*cse696 + 0.015391166*cse698 - 22.089523);
  const amrex::Real cse708 = cse707*state.xn[1];
  const amrex::Real cse709 = std::pow(10.0, 2.1892372*cse690 + 0.19191374999999999*cse692 - 0.16596184*cse694 + 0.29036281000000003*cse696 - 0.81520437999999995*cse698 - 23.689236999999999);
  const amrex::Real cse710 = cse680*cse709;
  const amrex::Real cse711 = cse665*std::pow(cse689, 8);
  const amrex::Real cse712 = std::pow(cse689, 7);
  const amrex::Real cse713 = cse667*cse712;
  const amrex::Real cse714 = std::pow(cse689, 6);
  const amrex::Real cse715 = cse669*cse714;
  const amrex::Real cse716 = std::pow(10.0, 16.815729999999999*cse690 + 983.67575999999997*cse692 + 734.71650999999997*cse694 + 343.1918*cse696 + 96.743155000000002*cse698 + 70.609154000000004*cse711 + 364.14445999999998*cse713 + 801.81246999999996*cse715 - 21.928795999999998);
  const amrex::Real cse717 = std::pow(10.0, 1.6802758*cse690 - 8.8077017000000009*cse692 - 4.7274035999999997*cse694 + 4.0406627000000004*cse696 + 0.93310621999999999*cse698 - 6.3701156000000001*cse711 + 6.4380698000000001*cse713 + 8.9167182999999994*cse715 - 22.921188999999998);
  const amrex::Real cse718 = ((cse660) ? (
   cse716
)
: (
   cse717
));
  const amrex::Real cse719 = cse704*state.xn[0] + cse706 + cse708 + cse710 + cse718*state.xn[2];
  const amrex::Real cse720 = std::pow(cse719, -2);
  const amrex::Real cse721 = ((cse659) ? (
   cse3
)
: (
   0.00016666666666666666
));
  const amrex::Real cse722 = std::exp(-11700.0*cse721);
  const amrex::Real cse723 = std::exp(-5860.0*cse721);
  const amrex::Real cse724 = std::exp(-510.0*cse721);
  const amrex::Real cse725 = std::exp(-2197000.0*((cse659) ? (
   std::pow(state.T, -3)
)
: (
   4.6296296296296294e-12
)));
  const amrex::Real cse726 = cse725*((cse659) ? (
   std::pow(state.T, 3.7599999999999998)
)
: (
   160638152785302.53
));
  const amrex::Real cse727 = 6.0142468035272636e-8*((cse659) ? (
   std::pow(state.T, 2.1000000000000001)
)
: (
   85924621.118090734
)) + 1.0;
  const amrex::Real cse728 = 1.0/cse727;
  const amrex::Real cse729 = 1.6e-18*cse722 + 6.7e-19*cse723 + 3.0e-24*cse724 + 4.985670872372847e-33*cse726*cse728;
  const amrex::Real cse730 = 1.0/cse719;
  const amrex::Real cse731 = cse729*cse730 + 1;
  const amrex::Real cse732 = std::pow(cse731, -2);
  const amrex::Real cse733 = cse720*std::pow(cse729, 2)*cse732;
  const amrex::Real cse734 = 0.045999999999999999*cse581;
  const amrex::Real cse735 = std::pow(cse580, cse734 + 0.22800000000000001);
  const amrex::Real cse736 = std::pow(cse580, cse734 + 0.23100000000000001);
  const amrex::Real cse737 = 2.6400000000000002e-10*cse736;
  const amrex::Real cse738 = cse737*state.xn[0];
  const amrex::Real cse739 = 0.031*cse581;
  const amrex::Real cse740 = 1.11e-10*std::pow(cse580, cse739 + 0.26300000000000001);
  const amrex::Real cse741 = 0.064000000000000001*cse581;
  const amrex::Real cse742 = 4.3749999999999998e-11*std::pow(cse580, cse741 + 0.071999999999999995);
  const amrex::Real cse743 = cse740 + cse742;
  const amrex::Real cse744 = 0.00235272*cse69;
  const amrex::Real cse745 = cse744 - 0.041816600000000002;
  const amrex::Real cse746 = cse69*cse745 + 0.35826400000000003;
  const amrex::Real cse747 = cse69*cse746 - 0.57443;
  const amrex::Real cse748 = 0.097857299999999994*cse69;
  const amrex::Real cse749 = cse748 - 3.1926800000000002;
  const amrex::Real cse750 = cse69*cse749 + 38.504899999999999;
  const amrex::Real cse751 = cse69*cse750 - 202.19300000000001;
  const amrex::Real cse752 = std::exp(((cse684) ? (
   cse69*cse747 - 7.4386999999999999
)
: (
   cse69*cse751 + 386.18599999999998
)));
  const amrex::Real cse753 = cse259*state.xn[2];
  const amrex::Real cse754 = 1.7258000000000001e-6*cse753;
  const amrex::Real cse755 = cse752*cse754;
  const amrex::Real cse756 = cse738 + cse743*state.xn[3] + cse755;
  const amrex::Real cse757 = std::exp(-101.30000000000001*cse3);
  const amrex::Real cse758 = cse756*cse757;
  const amrex::Real cse759 = 8.9000000000000003e-11*cse735;
  const amrex::Real cse760 = cse759*state.xn[0];
  const amrex::Real cse761 = 0.037999999999999999*cse581;
  const amrex::Real cse762 = 5.1749999999999996e-11*std::pow(cse580, cse761 + 0.16900000000000001);
  const amrex::Real cse763 = 0.048000000000000001*cse581;
  const amrex::Real cse764 = 2.15e-11*std::pow(cse580, cse763 - 0.01);
  const amrex::Real cse765 = cse762 + cse764;
  const amrex::Real cse766 = 0.0070527699999999999*cse69;
  const amrex::Real cse767 = cse766 - 0.11133800000000001;
  const amrex::Real cse768 = cse69*cse767 + 0.69763799999999998;
  const amrex::Real cse769 = cse69*cse768 - 1.3074300000000001;
  const amrex::Real cse770 = 0.093813800000000003*cse69;
  const amrex::Real cse771 = cse770 - 3.0328300000000001;
  const amrex::Real cse772 = cse69*cse771 + 36.180300000000003;
  const amrex::Real cse773 = cse69*cse772 - 187.47399999999999;
  const amrex::Real cse774 = std::exp(((cse684) ? (
   cse69*cse769 - 7.6973500000000001
)
: (
   cse69*cse773 + 350.60899999999998
)));
  const amrex::Real cse775 = cse754*cse774;
  const amrex::Real cse776 = cse760 + cse765*state.xn[3] + cse775;
  const amrex::Real cse777 = cse757*cse776;
  const amrex::Real cse778 = 0.057000000000000002*cse581;
  const amrex::Real cse779 = std::pow(cse580, cse778 + 0.115);
  const amrex::Real cse780 = 1.26e-10*cse779;
  const amrex::Real cse781 = cse780*state.xn[0];
  const amrex::Real cse782 = 0.049000000000000002*cse581;
  const amrex::Real cse783 = 5.3249999999999997e-11*std::pow(cse580, cse782 - 0.0040000000000000001);
  const amrex::Real cse784 = 0.10199999999999999*cse581;
  const amrex::Real cse785 = 1.675e-11*std::pow(cse580, cse784 - 0.085000000000000006);
  const amrex::Real cse786 = cse783 + cse785;
  const amrex::Real cse787 = 0.00065632500000000003*cse69;
  const amrex::Real cse788 = -cse787 - 0.015089200000000001;
  const amrex::Real cse789 = cse69*cse788 + 0.36118400000000001;
  const amrex::Real cse790 = cse69*cse789 - 0.77378199999999997;
  const amrex::Real cse791 = 0.10508000000000001*cse69;
  const amrex::Real cse792 = cse791 - 3.4762;
  const amrex::Real cse793 = cse69*cse792 + 42.594999999999999;
  const amrex::Real cse794 = cse69*cse793 - 227.91300000000001;
  const amrex::Real cse795 = std::exp(((cse684) ? (
   cse69*cse790 - 9.2514099999999999
)
: (
   cse69*cse794 + 444.60000000000002
)));
  const amrex::Real cse796 = 2.8763333333333333e-6*cse795;
  const amrex::Real cse797 = cse753*cse796;
  const amrex::Real cse798 = cse781 + cse786*state.xn[3] + cse797;
  const amrex::Real cse799 = std::exp(-62.460000000000001*cse3);
  const amrex::Real cse800 = cse798*cse799;
  const amrex::Real cse801 = 1.32e-9*cse736;
  const amrex::Real cse802 = 6.3e-10*cse779;
  const amrex::Real cse803 = cse776*cse799;
  const amrex::Real cse804 = cse756*cse799;
  const amrex::Real cse805 = cse798 + 7.8800000000000004e-8;
  const amrex::Real cse806 = cse799*cse805;
  const amrex::Real cse807 = 4.4500000000000001e-10*cse735;
  const amrex::Real cse808 = 7.4166666666666679e-10*cse735*cse758 + 2.2000000000000003e-9*cse736*cse777 + cse800*cse801 + cse802*cse803 + cse802*cse804 + cse806*cse807;
  const amrex::Real cse809 = cse756 + 2.65e-7;
  const amrex::Real cse810 = cse776 + 1.81e-14;
  const amrex::Real cse811 = std::exp(-38.840000000000003*cse3);
  const amrex::Real cse812 = cse756*cse811;
  const amrex::Real cse813 = (5.0/3.0)*cse810;
  const amrex::Real cse814 = (25.0/3.0)*cse777;
  const amrex::Real cse815 = 5*cse799;
  const amrex::Real cse816 = cse756*cse815;
  const amrex::Real cse817 = cse776*cse815;
  const amrex::Real cse818 = cse756*cse814 + cse798*cse816 + cse805*cse817;
  const amrex::Real cse819 = std::exp(-23.619999999999997*cse3);
  const amrex::Real cse820 = cse798*cse819;
  const amrex::Real cse821 = 3*cse820;
  const amrex::Real cse822 = cse809*cse817 + cse809*cse821 + cse810*cse821;
  const amrex::Real cse823 = cse805*cse809 + cse805*cse810 + cse812*cse813 + cse818 + cse822;
  const amrex::Real cse824 = 1.0/cse823;
  const amrex::Real cse825 = 1.4210469458259591e-21*cse824;
  const amrex::Real cse826 = cse799*cse809;
  const amrex::Real cse827 = 3.7800000000000004e-10*cse779;
  const amrex::Real cse828 = cse810*cse819;
  const amrex::Real cse829 = cse809*cse819;
  const amrex::Real cse830 = 2.6700000000000001e-10*cse735*cse820 + 7.9200000000000006e-10*cse736*cse820 + cse801*cse803 + cse807*cse826 + cse827*cse828 + cse827*cse829;
  const amrex::Real cse831 = 2.5697412351439998e-22*cse824;
  const amrex::Real cse832 = std::pow(cse823, -2);
  const amrex::Real cse833 = cse810*cse811;
  const amrex::Real cse834 = cse832*(-1.4833333333333334e-10*cse735*cse812 - 4.4000000000000003e-10*cse736*cse833 - cse737*cse805 - cse759*cse805 - cse780*cse809 - cse780*cse810 - cse808 - cse830);
  const amrex::Real cse835 = 1.4210469458259591e-21*cse818;
  const amrex::Real cse836 = 2.5697412351439998e-22*cse822;
  const amrex::Real cse837 = state.T < 1000000.0;
  const amrex::Real cse838 = cse403 + 0.070000000000000007;
  const amrex::Real cse839 = 1.0/cse838;
  const amrex::Real cse840 = std::sqrt(cse403*cse839);
  const amrex::Real cse841 = 26.399999999999999*cse840;
  const amrex::Real cse842 = cse841 + 6.5;
  const amrex::Real cse843 = cse11*cse842;
  const amrex::Real cse844 = cse8*std::log(cse10);
  const amrex::Real cse845 = cse844 < 2;
  const amrex::Real cse846 = cse844 < 4;
  const amrex::Real cse847 = cse844 < 7;
  const amrex::Real cse848 = cse844 < 10;
  const amrex::Real cse849 = ((cse845) ? (
   10.0
)
: ((cse846) ? (
   (3.0/2.0)*cse844 + 7
)
: ((cse847) ? (
   (4.0/3.0)*cse844 + 23.0/3.0
)
: ((cse848) ? (
   (1.0/3.0)*cse844 + 44.0/3.0
)
: (
   18.0
)))));
  const amrex::Real cse850 = 2.0*cse11;
  const amrex::Real cse851 = cse849*cse850;
  const amrex::Real cse852 = cse843*state.xn[0] + cse851*state.xn[3];
  const amrex::Real cse853 = cse662 + 1;
  const amrex::Real cse854 = std::pow(cse853, -2);
  const amrex::Real cse855 = 1.0/cse853;
  const amrex::Real cse856 = std::pow(10, -2.0230000000000001*cse854 - 3.6709999999999998*cse855 - 11.08);
  const amrex::Real cse857 = cse11*cse856;
  const amrex::Real cse858 = cse857*state.xn[3];
  const amrex::Real cse859 = std::pow(10, -2.3900000000000001*cse854 + 0.055500000000000001*cse855 - 11.06);
  const amrex::Real cse860 = cse11*cse859;
  const amrex::Real cse861 = cse860*state.xn[0];
  const amrex::Real cse862 = cse858 + cse861;
  const amrex::Real cse863 = 1.0/cse862;
  const amrex::Real cse864 = 1.0/state.xn[0];
  const amrex::Real cse865 = 3.0000000000000002e-15*d2g;
  const amrex::Real cse866 = cse10*cse865;
  const amrex::Real cse867 = cse864*cse866 + 1.9999999999999999e-7;
  const amrex::Real cse868 = cse11*cse863*cse867 + 1;
  const amrex::Real cse869 = 3.2043540000000002e-13 + 6.7291434000000005e-12/cse868;
  const amrex::Real cse870 = cse38*cse842;
  const amrex::Real cse871 = cse35*state.xn[2];
  const amrex::Real cse872 = std::pow(cse838, -2);
  const amrex::Real cse873 = 0.5*cse872;
  const amrex::Real cse874 = cse120/std::pow(cse10, 3);
  const amrex::Real cse875 = -0.5*cse839*cse871 + cse873*cse874;
  const amrex::Real cse876 = cse65*cse838*cse841*state.xn[0];
  const amrex::Real cse877 = cse11*cse8;
  const amrex::Real cse878 = ((cse845) ? (
   0
)
: ((cse846) ? (
   (3.0/2.0)*cse877
)
: ((cse847) ? (
   (4.0/3.0)*cse877
)
: ((cse848) ? (
   (1.0/3.0)*cse877
)
: (
   0
)))));
  const amrex::Real cse879 = cse850*state.xn[3];
  const amrex::Real cse880 = cse36*cse849;
  const amrex::Real cse881 = 2.0*cse880;
  const amrex::Real cse882 = cse843 - cse870 + cse875*cse876 + cse878*cse879 - cse881;
  const amrex::Real cse883 = std::pow(state.T, 1.0/4.0);
  const amrex::Real cse884 = 1.75*cse883;
  const amrex::Real cse885 = std::exp(-state.T);
  const amrex::Real cse886 = 1 - cse885;
  const amrex::Real cse887 = std::pow(cse886, 1000);
  const amrex::Real cse888 = l0_co_interp1d(state.T);
  const amrex::Real cse889 = 1.0/cse888;
  const amrex::Real cse890 = 680.10000000000002/cse883;
  const amrex::Real cse891 = cse884*state.xn[0] + cse890*state.xn[2] + state.xn[3];
  const amrex::Real cse892 = M_SQRT2;
  const amrex::Real cse893 = std::max(gradv, 5.7445531974402559e-18*cse2*cse892);
  const amrex::Real cse894 = 100000.0/cse893;
  const amrex::Real cse895 = ncol_H <= 0;
  const amrex::Real cse896 = cse11*ncol_H;
  const amrex::Real cse897 = std::max(1.0000000000000001e-30, cse896);
  const amrex::Real cse898 = 1756701454.4310031*ncol_CO;
  const amrex::Real cse899 = ((cse895) ? (
   cse894*state.xn[9]
)
: (
   cse898/std::sqrt(cse897)
));
  const amrex::Real cse900 = llte_co_interp2d(state.T, cse899);
  const amrex::Real cse901 = 1.0/cse900;
  const amrex::Real cse902 = nhalf_co_interp2d(state.T, cse899);
  const amrex::Real cse903 = cse888*cse901;
  const amrex::Real cse904 = -cse902*cse903 + 1;
  const amrex::Real cse905 = 1.0/cse902;
  const amrex::Real cse906 = cse891*cse905;
  const amrex::Real cse907 = alpha_co_interp2d(state.T, cse899);
  const amrex::Real cse908 = std::pow(cse906, cse907);
  const amrex::Real cse909 = cse889*cse908;
  const amrex::Real cse910 = cse904*cse909;
  const amrex::Real cse911 = cse889 + cse891*cse901 + cse910;
  const amrex::Real cse912 = 1.0/cse911;
  const amrex::Real cse913 = cse887*cse912;
  const amrex::Real cse914 = cse913*state.xn[9];
  const amrex::Real cse915 = cse35*ncol_H*(((cse896 - 1.0000000000000001e-30 < 0) ? (
   0
)
: ((cse896 - 1.0000000000000001e-30 == 0) ? (
   1.0/2.0
)
: (
   1
))))/std::pow(cse897, 3.0/2.0);
  const amrex::Real cse916 = cse915*ncol_CO;
  const amrex::Real cse917 = ((cse895) ? (
   0
)
: (
   878350727.21550155*cse916
));
  const amrex::Real cse918 = alpha_co_interp2d_partial_1(state.T, cse899);
  const amrex::Real cse919 = std::log(cse906);
  const amrex::Real cse920 = cse918*cse919;
  const amrex::Real cse921 = cse891/std::pow(cse902, 2);
  const amrex::Real cse922 = nhalf_co_interp2d_partial_1(state.T, cse899);
  const amrex::Real cse923 = cse917*cse922;
  const amrex::Real cse924 = cse907/cse891;
  const amrex::Real cse925 = cse902*cse924;
  const amrex::Real cse926 = std::pow(cse900, -2);
  const amrex::Real cse927 = llte_co_interp2d_partial_1(state.T, cse899);
  const amrex::Real cse928 = -cse891*cse917*cse926*cse927 + cse909*(cse888*cse902*cse917*cse926*cse927 - cse903*cse923);
  const amrex::Real cse929 = cse891*state.xn[9];
  const amrex::Real cse930 = cse887*cse929/std::pow(cse911, 2);
  const amrex::Real cse931 = 0.0086999999999999994*cse581;
  const amrex::Real cse932 = std::pow(cse580, cse931 + 0.12809999999999999);
  const amrex::Real cse933 = 7.5799999999999997e-10*cse932*state.xn[0];
  const amrex::Real cse934 = std::pow(state.T, 0.070000000000000007);
  const amrex::Real cse935 = 0.75*((cse660) ? (
   1.1000000000000001e-13*state.T + 5.3300000000000002e-10
)
: (
   3.74757785025e-10*cse934
)) + 0.25*((cse660) ? (
   3.3000000000000001e-13*state.T + 4.4299999999999997e-10
)
: (
   3.8899728635600003e-10*cse934
));
  const amrex::Real cse936 = 4.5299999999999998e-6*cse753 + cse933 + cse935*state.xn[3];
  const amrex::Real cse937 = cse936 + 2.3e-6;
  const amrex::Real cse938 = std::exp(-91.209999999999994*cse3);
  const amrex::Real cse939 = cse938/cse937;
  const amrex::Real cse940 = 2*cse939;
  const amrex::Real cse941 = cse936*cse940 + 1;
  const amrex::Real cse942 = 1.0/cse941;
  const amrex::Real cse943 = cse939*state.xn[8];
  const amrex::Real cse944 = cse942*cse943;
  const amrex::Real cse945 = std::pow(cse868, -2);
  const amrex::Real cse946 = -cse35*cse863*cse867;
  const amrex::Real cse947 = cse864*cse865;
  const amrex::Real cse948 = std::pow(state.xn[0], 2);
  const amrex::Real cse949 = cse11*cse863;
  const amrex::Real cse950 = cse36*cse856;
  const amrex::Real cse951 = cse38*cse859;
  const amrex::Real cse952 = cse950 + cse951;
  const amrex::Real cse953 = std::pow(cse862, -2);
  const amrex::Real cse954 = cse11*cse953;
  const amrex::Real cse955 = cse954*(-cse860 + cse952);
  const amrex::Real cse956 = 1.5159999999999999e-9*cse932;
  const amrex::Real cse957 = std::pow(cse937, -2);
  const amrex::Real cse958 = cse936*cse938*cse957;
  const amrex::Real cse959 = cse936*cse939;
  const amrex::Real cse960 = 5.7927337833400002e-20*cse959*state.xn[8]/std::pow(cse941, 2);
  const amrex::Real cse961 = std::pow(state.T, 0.14999999999999999);
  const amrex::Real cse962 = 0.00088240812266906664*std::pow(state.T, 0.65000000000000002) + 1;
  const amrex::Real cse963 = 1.0/cse962;
  const amrex::Real cse964 = cse963*state.xn[2];
  const amrex::Real cse965 = cse961*cse964;
  const amrex::Real cse966 = 4.4460389837719569e-9*cse965 + 626500000.0;
  const amrex::Real cse967 = 1.0/cse966;
  const amrex::Real cse968 = std::exp(-118400.0*cse3);
  const amrex::Real cse969 = cse967*cse968;
  const amrex::Real cse970 = 1.3338116951315871e-8*cse969;
  const amrex::Real cse971 = cse965*cse970 + 1;
  const amrex::Real cse972 = 1.0/cse971;
  const amrex::Real cse973 = cse969*cse972;
  const amrex::Real cse974 = 1.3659996292638161e-10*cse973;
  const amrex::Real cse975 = cse514 + 1.9999999999999999e-7;
  const amrex::Real cse976 = cse949*cse975 + 1;
  const amrex::Real cse977 = std::pow(cse976, -2);
  const amrex::Real cse978 = cse35*cse863*cse975;
  const amrex::Real cse979 = 100.0*d2g;
  const amrex::Real cse980 = psigd_coll_interp2d(cse10, state.T);
  const amrex::Real cse981 = 0.00045101070015594175*cse64 + 4.1397668473788896;
  const amrex::Real cse982 = -2.1544346900318838*std::pow(cse981, 1.0/6.0) + state.T;
  const amrex::Real cse983 = 3.2000000000000002e-32*cse2;
  const amrex::Real cse984 = cse369*cse983;
  const amrex::Real cse985 = -cse979*cse980 + cse982*cse984;
  const amrex::Real cse986 = (((cse985 < 0) ? (
   0
)
: ((cse985 == 0) ? (
   1.0/2.0
)
: (
   1
))));
  const amrex::Real cse987 = cse986*psigd_coll_interp2d_partial_0(cse10, state.T);
  const amrex::Real cse988 = cse979*cse987;
  const amrex::Real cse989 = (((cse985 > 0) ? (
   0
)
: ((cse985 == 0) ? (
   1.0/2.0
)
: (
   1
))));
  const amrex::Real cse990 = cse982*cse989*d2g;
  const amrex::Real cse991 = cse983*cse990;
  const amrex::Real cse992 = state.xn[2] + 1.0e-50;
  const amrex::Real cse993 = 1.0/cse992;
  const amrex::Real cse994 = cse2*cse993;
  const amrex::Real cse995 = 8.9181112054588774e-9*cse994;
  const amrex::Real cse996 = cse64*cse995 + 50;
  const amrex::Real cse997 = std::log(cse996);
  const amrex::Real cse998 = 0.45350000000000001 + 2.234/cse997;
  const amrex::Real cse999 = std::pow(state.T, cse998);
  const amrex::Real cse1000 = 1.4419999999999999 - 0.050889999999999998*cse997;
  const amrex::Real cse1001 = std::exp(cse1000*cse997);
  const amrex::Real cse1002 = 1.8998126555754159e-29*cse1001*cse999;
  const amrex::Real cse1003 = d2g*state.xn[2];
  const amrex::Real cse1004 = cse1002*cse1003;
  const amrex::Real cse1005 = 2.25*std::pow(state.T, 0.14699999999999999) + 5.2199999999999998;
  const amrex::Real cse1006 = cse68 + 50;
  const amrex::Real cse1007 = 0.0043*std::pow(cse1006, 0.69199999999999995) + 1;
  const amrex::Real cse1008 = 0.049959999999999997*std::pow(cse1006, 0.43099999999999999)*cse1007 + 1;
  const amrex::Real cse1009 = 1.0/cse1008;
  const amrex::Real cse1010 = -8.9181112054588776e-33*state.c_hat*cse1005*cse1009*d2g*state.rn[0] + cse1004 - 3.0000000000000002e-15*cse869*d2g*state.xn[0] + cse988 + cse991;
  const amrex::Real cse1011 = 7.1777529600000012e-12*cse115;
  const amrex::Real cse1012 = 7.1777529600000012e-12*cse509;
  const amrex::Real cse1013 = 2.0*cse11*cse878*state.xn[3] + 26.399999999999999*cse65*cse838*cse840*cse875*state.xn[0] - cse870 - cse881;
  const amrex::Real cse1014 = cse905*cse907;
  const amrex::Real cse1015 = cse930*(-cse910*(-cse1014*cse923 + cse917*cse918*cse919) - cse928);
  const amrex::Real cse1016 = cse952*cse954;
  const amrex::Real cse1017 = -cse1016*cse867 - cse946 - cse947*cse949;
  const amrex::Real cse1018 = -cse1016*cse975 + cse35*cse863*cse975;
  const amrex::Real cse1019 = cse1002*cse369;
  const amrex::Real cse1020 = cse961*cse963;
  const amrex::Real cse1021 = std::pow(state.T, 0.29999999999999999);
  const amrex::Real cse1022 = std::pow(cse962, -2);
  const amrex::Real cse1023 = std::pow(cse966, -2);
  const amrex::Real cse1024 = std::pow(cse992, -2);
  const amrex::Real cse1025 = 1.0/cse996;
  const amrex::Real cse1026 = cse1025*cse64;
  const amrex::Real cse1027 = cse1024*cse1026*cse2;
  const amrex::Real cse1028 = cse1001*cse999;
  const amrex::Real cse1029 = cse69/std::pow(cse997, 2);
  const amrex::Real cse1030 = 3.7850082348397978e-37*cse1028*cse1029*state.xn[2];
  const amrex::Real cse1031 = std::pow(cse1008, -2);
  const amrex::Real cse1032 = std::pow(cse1006, 0.12299999999999994);
  const amrex::Real cse1033 = 1.3257751158800533e-12*cse1032;
  const amrex::Real cse1034 = cse132*cse64;
  const amrex::Real cse1035 = std::pow(cse1006, -0.56899999999999995)*cse1007;
  const amrex::Real cse1036 = 1.920315482404567e-10*cse1035;
  const amrex::Real cse1037 = 1.3659996292638161e-10*cse1020*cse291;
  const amrex::Real cse1038 = cse1037*cse969/std::pow(cse971, 2);
  const amrex::Real cse1039 = 0.5*cse11*cse839 - cse871*cse873;
  const amrex::Real cse1040 = cse1019*state.xn[2];
  const amrex::Real cse1041 = 9.0599999999999997e-6*cse259;
  const amrex::Real cse1042 = 1.024e-10*cse614;
  const amrex::Real cse1043 = 2.1600000000000004e-15*cse584;
  const amrex::Real cse1044 = 9.7199999999999998e-11*cse599;
  const amrex::Real cse1045 = cse1042*cse609 + cse1042*cse613 + cse1043*cse623 + cse1044*cse627 + 7.1999999999999997e-16*cse584*cse625 + 3.2399999999999999e-11*cse597*cse599;
  const amrex::Real cse1046 = 3.0719999999999998e-10*cse614;
  const amrex::Real cse1047 = cse1043*cse609 + cse1044*cse647 + cse1046*cse649 + cse1046*cse650 + 6.4799999999999996e-15*cse584*cse641 + 2.9159999999999995e-10*cse599*cse641;
  const amrex::Real cse1048 = cse653*(-cse1045 - cse1047 - 3.6000000000000001e-15*cse584*cse654 - cse585*cse626 - 1.6199999999999998e-10*cse599*cse633 - cse600*cse626 - cse615*cse630 - cse615*cse631);
  const amrex::Real cse1049 = cse259*cse752;
  const amrex::Real cse1050 = cse259*cse795;
  const amrex::Real cse1051 = 1.4381666666666666e-5*cse1050;
  const amrex::Real cse1052 = cse259*cse774;
  const amrex::Real cse1053 = 8.6289999999999999e-6*cse1049;
  const amrex::Real cse1054 = 8.6289999999999999e-6*cse1052;
  const amrex::Real cse1055 = 1.4381666666666668e-5*cse1049*cse777 + cse1051*cse803 + cse1051*cse804 + 1.4381666666666668e-5*cse1052*cse758 + cse1053*cse800 + cse1054*cse806;
  const amrex::Real cse1056 = 5.1774000000000006e-6*cse820;
  const amrex::Real cse1057 = 8.6289999999999999e-6*cse1050;
  const amrex::Real cse1058 = cse1049*cse1056 + cse1052*cse1056 + cse1053*cse803 + cse1054*cse826 + cse1057*cse828 + cse1057*cse829;
  const amrex::Real cse1059 = cse259*cse796;
  const amrex::Real cse1060 = 1.7258000000000001e-6*cse805;
  const amrex::Real cse1061 = cse832*(-cse1049*cse1060 - 2.8763333333333337e-6*cse1049*cse833 - cse1052*cse1060 - 2.8763333333333337e-6*cse1052*cse812 - cse1055 - cse1058 - cse1059*cse809 - cse1059*cse810);
  const amrex::Real cse1062 = cse1003*cse1028;
  const amrex::Real cse1063 = 3.7996253111508318e-29*cse1062;
  const amrex::Real cse1064 = cse2*cse990;
  const amrex::Real cse1065 = 6.4000000000000004e-32*cse1064;
  const amrex::Real cse1066 = cse987*d2g;
  const amrex::Real cse1067 = 200.0*cse1066;
  const amrex::Real cse1068 = 6.4087080000000003e-13 + 2.5634832e-11/cse976;
  const amrex::Real cse1069 = 5.7927337833400002e-20*cse942;
  const amrex::Real cse1070 = cse1069*cse943;
  const amrex::Real cse1071 = -cse935;
  const amrex::Real cse1072 = cse1069*cse958*state.xn[8];
  const amrex::Real cse1073 = 2*cse958;
  const amrex::Real cse1074 = cse169*cse856 + 2*cse951;
  const amrex::Real cse1075 = cse954*(cse1074 - cse857);
  const amrex::Real cse1076 = cse864*cse949;
  const amrex::Real cse1077 = cse1076*cse56 - 2*cse35*cse863*cse867;
  const amrex::Real cse1078 = 2*cse870;
  const amrex::Real cse1079 = cse839*cse871;
  const amrex::Real cse1080 = cse872*cse874;
  const amrex::Real cse1081 = -1.0*cse1079 + 1.0*cse1080;
  const amrex::Real cse1082 = ((cse845) ? (
   0
)
: ((cse846) ? (
   3*cse877
)
: ((cse847) ? (
   (8.0/3.0)*cse877
)
: ((cse848) ? (
   (2.0/3.0)*cse877
)
: (
   0
)))));
  const amrex::Real cse1083 = 4.0*cse880;
  const amrex::Real cse1084 = -cse1078 + cse1081*cse876 + cse1082*cse879 - cse1083 + cse851;
  const amrex::Real cse1085 = ((cse895) ? (
   0
)
: (
   cse898*cse915
));
  const amrex::Real cse1086 = cse1085*cse922;
  const amrex::Real cse1087 = -cse1085*cse891*cse926*cse927 + cse909*(cse1085*cse888*cse902*cse926*cse927 - cse1086*cse903);
  const amrex::Real cse1088 = 1.0/cse731;
  const amrex::Real cse1089 = cse622*cse635;
  const amrex::Real cse1090 = (1.0/15.0)*cse625;
  const amrex::Real cse1091 = cse607*cse626;
  const amrex::Real cse1092 = cse1089*cse594 + cse1090*cse594 + cse1091*cse635 + cse607*cse637 + cse621*cse636 + cse621*cse638;
  const amrex::Real cse1093 = cse621*cse631;
  const amrex::Real cse1094 = (3.0/5.0)*cse640;
  const amrex::Real cse1095 = cse621*cse630;
  const amrex::Real cse1096 = cse630*cse635;
  const amrex::Real cse1097 = cse1093*cse1094 + cse1094*cse1095 + cse1096*cse607 + cse594*cse638 + cse594*cse642 + cse607*cse642;
  const amrex::Real cse1098 = (1.0/3.0)*cse654;
  const amrex::Real cse1099 = cse653*(-cse1091 - cse1092 - cse1093 - cse1095 - cse1097 - cse1098*cse594 - cse594*cse626 - cse607*cse634);
  const amrex::Real cse1100 = cse798*cse815;
  const amrex::Real cse1101 = (25.0/3.0)*cse758;
  const amrex::Real cse1102 = cse765*cse805;
  const amrex::Real cse1103 = cse1100*cse743 + cse1101*cse765 + cse1102*cse815 + cse743*cse814 + cse786*cse816 + cse786*cse817;
  const amrex::Real cse1104 = cse809*cse815;
  const amrex::Real cse1105 = cse786*cse810;
  const amrex::Real cse1106 = 3*cse819;
  const amrex::Real cse1107 = cse786*cse809;
  const amrex::Real cse1108 = cse1104*cse765 + cse1105*cse1106 + cse1106*cse1107 + cse743*cse817 + cse743*cse821 + cse765*cse821;
  const amrex::Real cse1109 = cse811*cse813;
  const amrex::Real cse1110 = (5.0/3.0)*cse812;
  const amrex::Real cse1111 = cse832*(-cse1102 - cse1103 - cse1105 - cse1107 - cse1108 - cse1109*cse743 - cse1110*cse765 - cse743*cse805);
  const amrex::Real cse1112 = cse1009*cse64*d2g;
  const amrex::Real cse1113 = cse1005*cse1112;
  const amrex::Real cse1114 = cse1074*cse954;
  const amrex::Real cse1115 = cse977*state.xn[3];
  const amrex::Real cse1116 = 1.5937167502281388e-29*cse1115*cse182*cse64;
  const amrex::Real cse1117 = 2.0187430200000004e-26*cse163*cse945;
  const amrex::Real cse1118 = cse10*cse1117;
  const amrex::Real cse1119 = -cse1078 + 26.399999999999999*cse1081*cse65*cse838*cse840*state.xn[0] + 2.0*cse1082*cse11*state.xn[3] - cse1083;
  const amrex::Real cse1120 = 1.602177e-12*cse85;
  const amrex::Real cse1121 = 3.2043540000000001e-12*crate*state.xn[3];
  const amrex::Real cse1122 = -state.xn[3]*((cse688) ? (
   0
)
: (
   cse709*cse733
));
  const amrex::Real cse1123 = ((cse895) ? (
   cse894
)
: (
   0
));
  const amrex::Real cse1124 = cse1123*cse922;
  const amrex::Real cse1125 = 8.9181112054588776e-33*cse1005;
  const amrex::Real cse1126 = -cse1004 - cse1011 - cse1012 + cse1013*cse1120 + cse1013*cse1121 - cse1015 + cse1017*cse1118 + cse1018*cse1116 + cse1112*cse1125 + 3.0000000000000002e-15*cse163*cse869 - cse988 - cse991;
  const amrex::Real cse1127 = cse954*(3*cse950 + 3*cse951);
  const amrex::Real cse1128 = 2.0*cse11*state.xn[3]*((cse845) ? (
   0
)
: ((cse846) ? (
   (9.0/2.0)*cse877
)
: ((cse847) ? (
   4*cse877
)
: ((cse848) ? (
   cse877
)
: (
   0
))))) + 26.399999999999999*cse65*cse838*cse840*state.xn[0]*(-1.5*cse1079 + 1.5*cse1080) - 3*cse870 - 6.0*cse880;
  const amrex::Real cse1129 = ((cse895) ? (
   0
)
: (
   2635052181.6465044*cse916
));
  const amrex::Real cse1130 = cse1129*cse922;
  const amrex::Real cse1131 = cse969*std::pow(state.T, -1.8500000000000001);
  const amrex::Real cse1132 = cse291*cse963;
  const amrex::Real cse1133 = std::pow(state.T, -0.84999999999999998);
  const amrex::Real cse1134 = std::pow(state.T, -0.19999999999999998);
  const amrex::Real cse1135 = cse1133*cse964;
  const amrex::Real cse1136 = cse1023*cse968*(2.5500935934394033e-12*cse1022*cse1134*state.xn[2] - 6.6690584756579352e-10*cse1135);
  const amrex::Real cse1137 = cse285*cse64;
  const amrex::Real cse1138 = cse1026*cse272*cse993;
  const amrex::Real cse1139 = cse255*cse959;
  const amrex::Real cse1140 = cse3*cse753;
  const amrex::Real cse1141 = std::pow(state.T, -0.92999999999999994);
  const amrex::Real cse1142 = 100*cse3;
  const amrex::Real cse1143 = -2.2649999999999999e-6*cse1140 + cse933*(cse1142*(8.7000000000000001e-5*cse581 + 0.001281) + cse3*cse931) + state.xn[3]*(0.75*((cse660) ? (
   1.1000000000000001e-13
)
: (
   2.6233044951750002e-11*cse1141
)) + 0.25*((cse660) ? (
   3.3000000000000001e-13
)
: (
   2.7229810044920004e-11*cse1141
)));
  const amrex::Real cse1144 = 0.4375*state.xn[0]/std::pow(state.T, 3.0/4.0) - 170.02500000000001*state.xn[2]/std::pow(state.T, 5.0/4.0);
  const amrex::Real cse1145 = -cse1143;
  const amrex::Real cse1146 = std::pow(cse853, -3);
  const amrex::Real cse1147 = -cse7*cse858*(0.0040460000000000001*cse1146 + 0.0036709999999999998*cse854) - cse7*cse861*(0.0047800000000000004*cse1146 - 5.5500000000000001e-5*cse854);
  const amrex::Real cse1148 = l0_co_interp1d_partial_0(state.T);
  const amrex::Real cse1149 = std::pow(cse888, -2);
  const amrex::Real cse1150 = ((cse895) ? (
   -2.872276598720128e-13*cse272*cse892*state.xn[9]*(((5.7445531974402559e-18*cse2*cse892 - gradv < 0) ? (
      0
   )
   : ((5.7445531974402559e-18*cse2*cse892 - gradv == 0) ? (
      1.0/2.0
   )
   : (
      1
   ))))/std::pow(cse893, 2)
)
: (
   0
));
  const amrex::Real cse1151 = -cse1150*cse927 - llte_co_interp2d_partial_0(state.T, cse899);
  const amrex::Real cse1152 = cse1150*cse922 + nhalf_co_interp2d_partial_0(state.T, cse899);
  const amrex::Real cse1153 = ((cse659) ? (
   -cse255
)
: (
   0
));
  const amrex::Real cse1154 = -1.8719999999999998e-14*cse1153*cse722 - 3.9261999999999998e-15*cse1153*cse723 - 1.53e-21*cse1153*cse724 + 4.985670872372847e-33*cse725*cse728*((cse659) ? (
   3.7599999999999998*std::pow(state.T, 2.7599999999999998)
)
: (
   0
)) - 1.0953518906603145e-26*cse726*cse728*((cse659) ? (
   -3/std::pow(state.T, 4)
)
: (
   0
)) - 2.9985055107607381e-40*cse726*((cse659) ? (
   2.1000000000000001*std::pow(state.T, 1.1000000000000001)
)
: (
   0
))/std::pow(cse727, 2);
  const amrex::Real cse1155 = ((cse659) ? (
   1
)
: (
   0
));
  const amrex::Real cse1156 = cse1155*cse721;
  const amrex::Real cse1157 = cse1156*cse13*cse689;
  const amrex::Real cse1158 = cse1156*cse671*cse693;
  const amrex::Real cse1159 = cse1156*cse673*cse695;
  const amrex::Real cse1160 = cse1156*cse8;
  const amrex::Real cse1161 = cse1156*cse675*cse697;
  const amrex::Real cse1162 = cse1156*cse665*cse712;
  const amrex::Real cse1163 = cse1156*cse667*cse714;
  const amrex::Real cse1164 = cse1156*cse669*cse691;
  const amrex::Real cse1165 = 65.320000000000007*cse255;
  const amrex::Real cse1166 = cse3*cse590;
  const amrex::Real cse1167 = 0.00042999999999999999*cse581;
  const amrex::Real cse1168 = cse617*(cse1142*(0.0041900000000000001 - 3.0000000000000001e-5*cse581) - cse3*cse610) - 3.8399999999999998e-11*std::pow(state.T, -1.075)*state.xn[2] + state.xn[3]*(cse619*(cse1142*(0.00025000000000000001*cse581 + 0.00264) + cse3*cse618) + cse620*(cse1142*(cse1167 + 0.00296) + cse1166));
  const amrex::Real cse1169 = cse602*(cse1142*(0.0036900000000000001 - 6.0000000000000002e-5*cse581) - cse3*cse582) - 1.2635999999999999e-11*std::pow(state.T, -1.026)*state.xn[2] + state.xn[3]*(cse604*(cse1142*(0.00041000000000000005*cse581 + 0.0020300000000000001) + cse3*cse603) + cse606*(cse1142*(0.00058*cse581 + 0.0023700000000000001) + cse3*cse605));
  const amrex::Real cse1170 = cse589*(cse1142*(0.0075500000000000003 - 0.0016000000000000001*cse581) - cse3*cse586) + 1.0000800000000001e-14*std::pow(state.T, -0.073999999999999955)*state.xn[2] + state.xn[3]*(cse591*(cse1142*(cse1167 + 0.0088900000000000003) + cse1166) + cse593*(cse1142*(0.0052500000000000003*cse581 + 0.011979999999999999) + cse3*cse592));
  const amrex::Real cse1171 = cse1169*cse626;
  const amrex::Real cse1172 = cse1089*cse1170 + cse1090*cse1170 + cse1165*cse595*cse623 + cse1165*cse608*cse627 + cse1168*cse636 + cse1168*cse638 + cse1169*cse637 + cse1171*cse635 + 28.360000000000003*cse255*cse597*cse608;
  const amrex::Real cse1173 = 136.68000000000001*cse255*cse641;
  const amrex::Real cse1174 = cse1168*cse631;
  const amrex::Real cse1175 = cse1168*cse630;
  const amrex::Real cse1176 = cse1094*cse1174 + cse1094*cse1175 + cse1096*cse1169 + cse1165*cse609*cse630 + cse1169*cse642 + cse1170*cse638 + cse1170*cse642 + cse1173*cse630 + cse1173*cse631;
  const amrex::Real cse1177 = cse653*(-cse1098*cse1170 - cse1169*cse634 - cse1170*cse626 - cse1171 - cse1172 - cse1174 - cse1175 - cse1176 - 32.933333333333337*cse255*cse631*cse633);
  const amrex::Real cse1178 = 312.30000000000001*cse255;
  const amrex::Real cse1179 = cse3*cse734;
  const amrex::Real cse1180 = 0.00046000000000000001*cse581;
  const amrex::Real cse1181 = 8.6290000000000003e-7*cse1140;
  const amrex::Real cse1182 = -cse1181*cse752 + cse738*(cse1142*(cse1180 + 0.00231) + cse1179) + cse755*((cse684) ? (
   cse3*cse747 + cse69*(cse3*cse746 + cse69*(cse3*cse744 + cse3*cse745))
)
: (
   cse3*cse751 + cse69*(cse3*cse750 + cse69*(cse3*cse748 + cse3*cse749))
)) + state.xn[3]*(cse740*(cse1142*(0.00031*cse581 + 0.00263) + cse3*cse739) + cse742*(cse1142*(0.00064000000000000005*cse581 + 0.00071999999999999994) + cse3*cse741));
  const amrex::Real cse1183 = -1.4381666666666666e-6*cse1140*cse795 + cse781*(cse1142*(0.00057000000000000009*cse581 + 0.00115) + cse3*cse778) + cse797*((cse684) ? (
   cse3*cse790 + cse69*(cse3*cse789 + cse69*(-cse3*cse787 + cse3*cse788))
)
: (
   cse3*cse794 + cse69*(cse3*cse793 + cse69*(cse3*cse791 + cse3*cse792))
)) + state.xn[3]*(cse783*(cse1142*(0.00048999999999999998*cse581 - 4.0000000000000003e-5) + cse3*cse782) + cse785*(cse1142*(0.0010199999999999999*cse581 - 0.00085000000000000006) + cse3*cse784));
  const amrex::Real cse1184 = -cse1181*cse774 + cse760*(cse1142*(cse1180 + 0.0022800000000000003) + cse1179) + cse775*((cse684) ? (
   cse3*cse769 + cse69*(cse3*cse768 + cse69*(cse3*cse766 + cse3*cse767))
)
: (
   cse3*cse773 + cse69*(cse3*cse772 + cse69*(cse3*cse770 + cse3*cse771))
)) + state.xn[3]*(cse762*(cse1142*(0.00038000000000000002*cse581 + 0.0016900000000000001) + cse3*cse761) + cse764*(cse1142*(0.00048000000000000001*cse581 - 0.0001) + cse3*cse763));
  const amrex::Real cse1185 = cse1184*cse805;
  const amrex::Real cse1186 = cse1100*cse1182 + cse1101*cse1184 + cse1178*cse756*cse800 + cse1178*cse776*cse806 + cse1182*cse814 + cse1183*cse816 + cse1183*cse817 + cse1185*cse815 + 844.16666666666686*cse255*cse756*cse777;
  const amrex::Real cse1187 = 70.859999999999985*cse255*cse820;
  const amrex::Real cse1188 = cse1183*cse810;
  const amrex::Real cse1189 = cse1183*cse809;
  const amrex::Real cse1190 = cse1104*cse1184 + cse1106*cse1188 + cse1106*cse1189 + cse1178*cse803*cse809 + cse1182*cse817 + cse1182*cse821 + cse1184*cse821 + cse1187*cse809 + cse1187*cse810;
  const amrex::Real cse1191 = cse832*(-cse1109*cse1182 - cse1110*cse1184 - cse1182*cse805 - cse1185 - cse1186 - cse1188 - cse1189 - cse1190 - 64.733333333333348*cse255*cse810*cse812);
  const amrex::Real cse1192 = state.c_hat*cse369;
  const amrex::Real cse1193 = state.c_hat*cse66;
  const amrex::Real cse1194 = state.c_hat*cse1025;
  const amrex::Real cse1195 = Zd*cse64;
  const amrex::Real cse1196 = 1.3890778379475547e-21*cse1195;
  const amrex::Real cse1197 = -cse1196;
  const amrex::Real cse1198 = 2.7781556758951093e-21*cse1195;
  const amrex::Real cse1199 = 4.5072610463162814e-18*cse182;
  const amrex::Real cse1200 = Zd*state.c_hat;
  const amrex::Real cse1201 = 1.3890778379475547e-21*cse1200;
  const amrex::Real cse1202 = -cse10*cse1201 - cse1199*cse312 - cse309 - cse496 - cse499 - cse575;
  const amrex::Real cse1203 = cse1201*state.rn[2*0+1];
  const amrex::Real cse1204 = -cse1203;
  const amrex::Real cse1205 = cse1200*state.rn[2*0+1];
  const amrex::Real cse1206 = 2.7781556758951093e-21*cse1205;
  const amrex::Real cse1207 = state.c_hat*state.rn[2*0+1];

  jac(1, 1) = -cse1 - cse10*cse56 + 2*cse29*state.xn[3] + 2*cse30*cse55 - cse57 + 2*cse60*state.xn[0]*state.xn[3] - cse98;
  jac(1, 2) = cse110 + cse119;
  jac(1, 3) = cse129 + cse131*state.xn[16] + cse135*cse136 + cse143 + cse154;
  jac(1, 4) = cse118*cse180 - cse136*cse187 + cse157 + cse162 + cse164 + 2*cse165 + 2*cse174 + 4*cse176 + cse177*state.xn[17] + cse183*cse64 + cse186*state.xn[8] + cse191 + cse199;
  jac(1, 5) = cse118*cse204 + cse164 + 2*cse203 + cse207 + cse210;
  jac(1, 7) = cse215;
  jac(1, 8) = cse219;
  jac(1, 9) = cse186*state.xn[3];
  jac(1, 10) = -cse220;
  jac(1, 11) = cse119 + cse222;
  jac(1, 12) = cse225 + cse226 + cse229;
  jac(1, 15) = cse230 + cse231 + cse233 + cse234;
  jac(1, 16) = cse234 + cse239;
  jac(1, 17) = cse118*cse245 + cse131*state.xn[2] - 1.8000000000000002e-14*cse163 + 2*cse244 + cse251;
  jac(1, 18) = cse177*state.xn[3] - cse252 + cse254;
  jac(1, 19) = cse307*(-cse140*cse286 + 9.8669999999999991e-12*cse184*cse283*state.xn[3]*state.xn[8] - cse257 - cse258 - cse261 + 2*cse270*cse30 - cse271 + 2*cse274*state.xn[0]*state.xn[3] - 1.5566492609718528e-6*cse275 + cse276 - 5.5769999999999994e-13*cse279 - cse281 + cse282 + cse284 - cse290 - cse306);
  jac(1, 20) = cse183*cse312 + cse309 + cse311 + cse314;
  jac(2, 1) = cse98;
  jac(2, 2) = -cse110;
  jac(2, 3) = -cse153;
  jac(2, 4) = cse157 + cse196 + cse315;
  jac(2, 5) = cse206 + cse209 + cse315;
  jac(2, 7) = cse211;
  jac(2, 11) = cse316;
  jac(2, 12) = -cse228;
  jac(2, 15) = cse316;
  jac(2, 16) = cse316;
  jac(2, 17) = cse249 - cse250;
  jac(2, 18) = cse252;
  jac(2, 19) = cse307*(-cse257 + cse292 + cse294 - cse295 - cse298 - cse317);
  jac(2, 20) = -cse314;
  jac(3, 1) = cse319 - cse356 - cse80 + cse82 + cse86 + cse97;
  jac(3, 2) = -cse109 - cse358;
  jac(3, 3) = -cse154 - cse372 - cse392 - cse400 - cse401 - cse416;
  jac(3, 4) = -cse199 - cse417 + cse418 + cse419 - cse423 - cse429 - cse434 - cse435;
  jac(3, 5) = -cse210 - cse417 - cse422 - cse428 - cse437;
  jac(3, 6) = cse438;
  jac(3, 7) = -cse440;
  jac(3, 8) = cse450 + cse453;
  jac(3, 9) = -cse456;
  jac(3, 10) = cse220;
  jac(3, 11) = -cse222 - cse358;
  jac(3, 12) = cse461;
  jac(3, 13) = cse463;
  jac(3, 14) = cse465;
  jac(3, 15) = cse466;
  jac(3, 16) = cse466;
  jac(3, 17) = -cse251 - cse467 - cse468 - cse469 - cse471 - cse472 - cse473 - cse474;
  jac(3, 18) = cse476;
  jac(3, 19) = cse307*(4.2899999999999994e-13*cse279 + cse287 + cse301 + cse306 + cse478 + cse479 + cse480 + cse483 + cse484 - cse486 + cse487 + cse488 - cse490 - cse491 - cse492 - cse493 - cse494);
  jac(3, 20) = -cse314 + cse496 - cse497 - cse498 - cse501;
  jac(4, 1) = 3.0000000000000002e-15*cse10*d2g - cse502 - cse503 - cse504 - cse505 - cse507 - cse508;
  jac(4, 2) = cse511;
  jac(4, 3) = cse139 - cse141 - cse142 + cse415 + cse512*state.xn[16];
  jac(4, 4) = -cse116 - cse165 - cse174 - cse175*cse9 - cse425 - cse514 - cse515 - cse517 - cse518 - cse519 - cse520;
  jac(4, 5) = -cse116 - cse203 - cse207 - cse436 - cse521;
  jac(4, 7) = -cse211 - cse522;
  jac(4, 8) = cse449;
  jac(4, 9) = -cse454;
  jac(4, 10) = cse523;
  jac(4, 11) = cse511;
  jac(4, 12) = cse460;
  jac(4, 15) = -cse233 - cse510;
  jac(4, 16) = cse511;
  jac(4, 17) = -cse244 + cse471 + cse512*state.xn[2] + cse524 - cse525 + cse527;
  jac(4, 18) = cse476;
  jac(4, 19) = cse307*(-cse256 - 2.3622303960273254e-7*cse275 + cse301 - cse528 - cse529 - cse530 - cse531 - cse532 - cse534);
  jac(4, 20) = -cse535;
  jac(5, 1) = cse508;
  jac(5, 2) = cse357;
  jac(5, 4) = -cse162 + cse516 + cse518;
  jac(5, 5) = -cse205 - cse206 - cse437;
  jac(5, 7) = cse522;
  jac(5, 11) = cse357;
  jac(5, 15) = cse357;
  jac(5, 16) = cse357;
  jac(5, 17) = cse470;
  jac(5, 19) = -cse307*cse533;
  jac(6, 1) = cse330;
  jac(6, 2) = cse330;
  jac(6, 3) = cse372;
  jac(6, 4) = cse536;
  jac(6, 5) = cse417;
  jac(6, 6) = -cse438;
  jac(6, 7) = cse538;
  jac(6, 10) = cse539;
  jac(6, 11) = cse330;
  jac(6, 15) = cse330;
  jac(6, 16) = cse540;
  jac(6, 17) = cse472;
  jac(6, 19) = cse307*cse542;
  jac(6, 20) = cse498;
  jac(7, 1) = cse543;
  jac(7, 2) = cse543;
  jac(7, 3) = -cse372;
  jac(7, 4) = -cse536;
  jac(7, 5) = -cse417;
  jac(7, 6) = cse438;
  jac(7, 7) = -cse538;
  jac(7, 10) = -cse539;
  jac(7, 11) = cse543;
  jac(7, 15) = cse543;
  jac(7, 16) = -cse540;
  jac(7, 17) = -cse472;
  jac(7, 19) = -cse307*cse542;
  jac(7, 20) = -cse498;
  jac(8, 1) = cse1 + cse547;
  jac(8, 2) = cse547;
  jac(8, 3) = cse392;
  jac(8, 4) = cse427 + cse549 + cse550*state.xn[8] + cse553;
  jac(8, 5) = cse553;
  jac(8, 8) = -cse219 - cse449 - cse453;
  jac(8, 9) = cse455 + cse550*state.xn[3];
  jac(8, 10) = cse556;
  jac(8, 11) = cse547;
  jac(8, 15) = cse547 + cse557;
  jac(8, 16) = -cse236 + cse547;
  jac(8, 17) = -cse468 + cse473 - cse558;
  jac(8, 19) = cse307*(cse258 - 1.287e-13*cse279 + cse289 + cse478 + 2.2769999999999997e-12*cse489 + cse559);
  jac(8, 20) = cse309 + cse501 + cse560;
  jac(9, 1) = cse561;
  jac(9, 2) = cse561;
  jac(9, 3) = -cse392;
  jac(9, 4) = -cse429;
  jac(9, 5) = -cse428;
  jac(9, 7) = cse537;
  jac(9, 8) = cse453;
  jac(9, 9) = -cse456 - cse563;
  jac(9, 10) = cse539;
  jac(9, 11) = cse561;
  jac(9, 15) = cse561;
  jac(9, 16) = -cse343 - cse564;
  jac(9, 17) = -cse473;
  jac(9, 19) = cse307*(-cse531 - cse559 - cse566);
  jac(9, 20) = cse499 - cse500;
  jac(10, 1) = cse545;
  jac(10, 2) = cse545;
  jac(10, 3) = cse145;
  jac(10, 4) = cse567;
  jac(10, 5) = cse551;
  jac(10, 7) = -cse537;
  jac(10, 8) = cse219;
  jac(10, 10) = -cse568 - cse569;
  jac(10, 11) = cse221 + cse545;
  jac(10, 12) = cse227;
  jac(10, 15) = cse570;
  jac(10, 16) = cse236 + cse545;
  jac(10, 17) = -cse526 + cse558;
  jac(10, 19) = -cse290*cse307;
  jac(10, 20) = -cse560;
  jac(11, 3) = -cse145;
  jac(11, 9) = cse563;
  jac(11, 10) = cse568;
  jac(11, 11) = -cse221;
  jac(11, 16) = cse564;
  jac(11, 17) = cse526;
  jac(11, 19) = cse307*(cse287 + cse566);
  jac(12, 1) = -cse545 - cse571;
  jac(12, 2) = -cse101 - cse545;
  jac(12, 3) = cse141 - cse411 + cse413;
  jac(12, 4) = cse189 - cse190 - cse433 - cse567;
  jac(12, 5) = cse552;
  jac(12, 7) = cse537;
  jac(12, 10) = cse569;
  jac(12, 11) = cse546;
  jac(12, 12) = -cse229 - cse572;
  jac(12, 15) = -cse570;
  jac(12, 16) = cse238 + cse546;
  jac(12, 17) = -cse467 - cse558;
  jac(12, 18) = cse252 + cse253;
  jac(12, 19) = cse307*(-cse573 - cse574);
  jac(12, 20) = cse575;
  jac(13, 1) = cse355;
  jac(13, 2) = cse355;
  jac(13, 3) = cse400;
  jac(13, 4) = cse423;
  jac(13, 5) = cse422;
  jac(13, 11) = cse355;
  jac(13, 13) = -cse463;
  jac(13, 14) = cse464;
  jac(13, 15) = cse355;
  jac(13, 16) = cse355;
  jac(13, 17) = cse474;
  jac(13, 19) = cse307*cse576;
  jac(13, 20) = -cse577;
  jac(14, 1) = cse578;
  jac(14, 2) = cse578;
  jac(14, 3) = -cse400;
  jac(14, 4) = -cse423;
  jac(14, 5) = -cse422;
  jac(14, 11) = cse578;
  jac(14, 13) = cse463;
  jac(14, 14) = cse465;
  jac(14, 15) = cse578;
  jac(14, 16) = cse578;
  jac(14, 17) = -cse474;
  jac(14, 19) = -cse307*cse576;
  jac(14, 20) = cse577;
  jac(15, 1) = cse505;
  jac(15, 4) = cse579*state.xn[8];
  jac(15, 8) = cse449;
  jac(15, 9) = cse579*state.xn[3];
  jac(15, 12) = -cse227;
  jac(15, 15) = -cse230 - cse557;
  jac(15, 17) = cse468;
  jac(15, 19) = cse307*(-3.0029999999999999e-13*cse279 + cse477 + 5.3129999999999997e-12*cse489 + cse528);
  jac(15, 20) = -cse309;
  jac(16, 3) = cse143 + cse414;
  jac(16, 4) = cse434;
  jac(16, 7) = -cse214;
  jac(16, 8) = -cse219;
  jac(16, 9) = -cse563;
  jac(16, 12) = cse572;
  jac(16, 16) = -cse239 - cse564;
  jac(16, 17) = cse467;
  jac(16, 18) = cse254;
  jac(16, 19) = cse307*(cse261 + cse289 + cse532 + cse565 + cse573);
  jac(16, 20) = -cse311;
  jac(17, 3) = -cse416;
  jac(17, 4) = cse519;
  jac(17, 5) = cse205;
  jac(17, 8) = cse450;
  jac(17, 10) = -cse523;
  jac(17, 12) = cse461;
  jac(17, 17) = -cse469 - cse527;
  jac(17, 19) = cse307*(cse479 + cse534);
  jac(18, 1) = cse571;
  jac(18, 2) = cse101;
  jac(18, 3) = -cse401;
  jac(18, 4) = -cse520;
  jac(18, 7) = cse214;
  jac(18, 12) = cse228;
  jac(18, 16) = cse235;
  jac(18, 18) = -cse252 - cse475;
  jac(18, 19) = cse307*(cse303 + cse541 + cse574);
  jac(19, 1) = 1.5937167502281388e-29*state.c_hat*cse182*cse977*state.rn[0]*state.xn[3]*(-cse955*cse975 + cse978) + 1.602177e-12*crate*cse852 + 1.602177e-12*crate*cse882*state.xn[0] + 3.2043540000000001e-12*crate*cse882*state.xn[3] + 3.0000000000000002e-15*cse10*cse869*d2g + 2.0187430200000004e-26*cse10*cse945*d2g*state.xn[0]*(-cse867*cse955 - cse946 - cse949*(-cse866/cse948 + cse947)) - cse1010 - 7.1777529600000012e-12*cse503 - 7.1777529600000012e-12*cse504 - 7.1777529600000012e-12*cse507 - cse884*cse914 - cse930*(-cse884*cse901 - cse910*(cse917*cse920 + cse925*(cse884*cse905 - cse921*cse923)) - cse928) + 4.39089220777172e-29*cse932*cse936*cse938*cse942*cse957*state.xn[8] - 4.39089220777172e-29*cse932*cse944 - cse960*(-cse939*cse956 + cse956*cse958) - cse965*cse974 - 2.1789607199999999e-11*cse97 - state.xn[11]*(cse629*cse646 + cse651*cse652 + cse655*cse656 + cse655*cse657) - state.xn[3]*((cse688) ? (
   0
)
: (
   cse704*cse733
)) - state.xn[7]*((cse837) ? (
   cse808*cse825 + cse830*cse831 + cse834*cse835 + cse834*cse836
)
: (
   0
));
  jac(19, 2) = 1.5937167502281388e-29*state.c_hat*cse1018*cse182*cse977*state.rn[0]*state.xn[3] + 1.602177e-12*crate*cse1013*state.xn[0] + 3.2043540000000001e-12*crate*cse1013*state.xn[3] + 2.0187430200000004e-26*cse10*cse1017*cse945*d2g*state.xn[0] - cse1010 - cse1011 - cse1012 - cse1015 - state.xn[3]*((cse688) ? (
   0
)
: (
   cse707*cse733
));
  jac(19, 3) = 8.9181112054588776e-33*state.c_hat*cse10*cse1005*cse1031*d2g*state.rn[0]*(cse1033*cse1034 + cse1034*cse1036) + 4.22974728e-11*crate*cse1039*cse65*cse838*cse840*cse948 + 8.4594945600000001e-11*crate*cse1039*cse65*cse838*cse840*state.xn[0]*state.xn[3] - cse1019 - cse1020*cse974*state.xn[0] + 6.0732876035249664e-19*cse1021*cse1022*cse1023*cse968*cse972*state.xn[0]*state.xn[2] - cse1027*cse1030*cse369 - cse1038*(-cse1020*cse970 + 5.9301787935659931e-17*cse1021*cse1022*cse1023*cse968*state.xn[2]) - cse1040*(4.5384267924580226e-10*state.c_hat*cse1024*cse1025*cse2*cse997*state.rn[0] - 8.9181112054588774e-9*cse1000*cse1027) - 2.1789607199999999e-11*cse146 + 2.6241084038530201e-25*cse259*cse936*cse938*cse942*cse957*state.xn[8] - 2.6241084038530201e-25*cse259*cse944 - cse890*cse914 - cse930*(-cse890*cse901 - cse890*cse910*cse924) - cse960*(-cse1041*cse939 + cse1041*cse958) - state.xn[11]*(cse1045*cse646 + cse1047*cse652 + cse1048*cse656 + cse1048*cse657) - state.xn[3]*((cse688) ? (
   0
)
: (
   cse718*cse733
)) - state.xn[7]*((cse837) ? (
   cse1055*cse825 + cse1058*cse831 + cse1061*cse835 + cse1061*cse836
)
: (
   0
));
  jac(19, 4) = 1.7836222410917755e-32*state.c_hat*cse1005*cse1009*d2g*state.rn[0] + 6.2169970539621192e-19*state.c_hat*cse1068*cse182*state.rn[0] + 1.5937167502281388e-29*state.c_hat*cse182*cse977*state.rn[0]*state.xn[3]*(-cse1075*cse975 + 2*cse978) + 1.602177e-12*crate*cse1084*state.xn[0] + 3.2043540000000001e-12*crate*cse1084*state.xn[3] + 3.2043540000000001e-12*crate*cse852 + 2.0187430200000004e-26*cse10*cse945*d2g*state.xn[0]*(-cse1075*cse867 - cse1077) - cse1063 - cse1065 - cse1067 - cse1070*cse935 - cse1071*cse1072 - 7.1777529600000012e-12*cse165 - 7.1777529600000012e-12*cse174 - 1.4355505920000002e-11*cse176 - 7.1777529600000012e-12*cse515 + 6.0000000000000005e-15*cse869*d2g*state.xn[0] - cse914 - cse930*(-cse1087 - cse901 - cse910*(cse1085*cse920 + cse925*(-cse1086*cse921 + cse905))) - cse960*(-cse1071*cse1073 - cse935*cse940) - state.xn[11]*(cse1092*cse646 + cse1097*cse652 + cse1099*cse656 + cse1099*cse657) - state.xn[3]*((cse688) ? (
   0
)
: (
   cse705*cse733
)) - state.xn[7]*((cse837) ? (
   cse1103*cse825 + cse1108*cse831 + cse1111*cse835 + cse1111*cse836
)
: (
   0
)) - ((cse688) ? (
   0.0
)
: (
   cse1088*cse729
));
  jac(19, 5) = -cse1063 - cse1065 - cse1067 + 1.7836222410917755e-32*cse1113 + cse1116*(-cse1114*cse975 + 2*cse35*cse863*cse975) + cse1118*(-cse1077 - cse1114*cse867) + cse1119*cse1120 + cse1119*cse1121 - 7.1777529600000012e-12*cse203 - 7.1777529600000012e-12*cse521 + cse57*cse869 - cse930*(-cse1087 - cse910*(-cse1014*cse1086 + cse1085*cse918*cse919));
  jac(19, 6) = cse1122;
  jac(19, 7) = cse1122;
  jac(19, 8) = -((cse837) ? (
   cse818*cse825 + cse822*cse831
)
: (
   0.0
));
  jac(19, 9) = -cse1069*cse959;
  jac(19, 10) = -cse891*cse913 - cse930*(cse1123*cse891*cse926*cse927 - cse909*(cse1123*cse888*cse902*cse926*cse927 - cse1124*cse903) - cse910*(-cse1014*cse1124 + cse1123*cse918*cse919));
  jac(19, 11) = cse1126;
  jac(19, 12) = -cse639*cse646 - cse643*cse652;
  jac(19, 15) = cse1126;
  jac(19, 16) = cse1126;
  jac(19, 17) = -5.6994379667262474e-29*cse1062 - 9.6000000000000005e-32*cse1064 - 300.0*cse1066 + 2.6754333616376631e-32*cse1113 + cse1116*(-cse1127*cse975 + 3*cse35*cse863*cse975) + cse1118*(-9.0000000000000011e-15*cse1076*d2g - cse1127*cse867 + 3*cse35*cse863*cse867) + cse1120*cse1128 + cse1121*cse1128 - 7.1777529600000012e-12*cse244 + cse524*cse869 - 7.1777529600000012e-12*cse525 - cse930*(cse1129*cse891*cse926*cse927 - cse909*(cse1129*cse888*cse902*cse926*cse927 - cse1130*cse903) - cse910*(-cse1014*cse1130 + cse1129*cse918*cse919));
  jac(19, 19) = cse307*(8.9181112054588776e-33*state.c_hat*cse10*cse1005*cse1031*d2g*state.rn[0]*(-6.6288755794002664e-13*cse1032*cse1137 - 9.6015774120228348e-11*cse1035*cse1137) + 2.9496652812055237e-33*state.c_hat*cse10*cse1009*d2g*state.rn[0]*std::pow(state.T, -0.85299999999999998) + 7.8348995947646122e-14*cse1022*cse1134*cse967*cse968*cse972*state.xn[0]*state.xn[2] - cse1037*cse1136*cse972 - cse1038*(7.6502807803182096e-12*cse1022*cse1134*cse967*cse968*state.xn[2] - 0.0015792330470357991*cse1131*cse964 - 2.0007175426973804e-9*cse1135*cse969 - 1.3338116951315871e-8*cse1136*cse965) - cse1040*(4.4590556027294387e-9*cse1000*cse1138 - 2.2692133962290113e-10*cse1138*cse997) - cse1040*(-9.9615302164975657e-9*cse1029*cse1138 + cse3*cse998) - cse1070*cse1143 - cse1072*cse1145 - cse1116*cse1147*cse954*cse975 - cse1117*cse1147*cse867*cse953 - 1.6173435610483583e-5*cse1131*cse1132*cse972 - 2.0489994438957239e-11*cse1132*cse1133*cse973 - 5.2835524837844135e-18*cse1139*cse942*state.xn[8] - cse1144*cse914 - 2.1789607199999999e-11*cse292 - 7.1777529600000012e-12*cse529 - 7.1777529600000012e-12*cse530 - 1000*cse885*std::pow(cse886, 999)*cse912*cse929 - cse930*(-cse1144*cse901 + cse1148*cse1149*cse904*cse908 + cse1148*cse1149 - cse1151*cse891*cse926 - cse909*(-cse1148*cse901*cse902 - cse1151*cse888*cse902*cse926 - cse1152*cse903) - cse910*(cse919*(cse1150*cse918 + alpha_co_interp2d_partial_0(state.T, cse899)) + cse925*(cse1144*cse905 - cse1152*cse921))) - cse960*(-cse1073*cse1145 - 182.41999999999999*cse1139 - cse1143*cse940) - cse979*cse986*psigd_coll_interp2d_partial_1(cse10, state.T) - cse989*(1.6000000000000001e-32*cse272*cse369*cse982 + cse984) - state.xn[11]*(cse1172*cse646 + cse1176*cse652 + cse1177*cse656 + cse1177*cse657) - state.xn[3]*((cse688) ? (
   0
)
: (
   cse1088*cse1154 + cse729*cse732*(-cse1154*cse730 - cse720*cse729*(-cse706*(3.0182298984217999*cse1155*cse675*cse697*cse721 + 4.8223900769399082*cse1155*cse721*cse8 - 3.5529549287336835*cse1157 - 0.3872755400043702*cse1158 - 1.3735579540080118*cse1159) - cse708*(0.0708789387907936*cse1157 + 3.7035619079275177*cse1158 - 4.6974781513675152*cse1159 + 3.6184459289309552*cse1160 - 1.6316107607322889*cse1161) - cse710*(-3.7541549062629067*cse1157 + 2.2094886994529301*cse1158 - 1.528565035159452*cse1159 + 5.0409049417480247*cse1160 + 2.0057552335975877*cse1161) - state.xn[0]*((cse700) ? (
      cse699*cse7*(116.29033200000001*cse1157 + 19.239805*cse1158 + 80.639324000000002*cse1159 + 37.383713*cse1160 + 145.968309*cse1161)
   )
   : ((cse702) ? (
      cse701*(8.2184944748967013*cse1155*cse721*cse8 - 52.189748993977005*cse1157 - 48.951834264235494*cse1158 - 196.44057098336626*cse1159 - 192.38155095558542*cse1161)
   )
   : (
      cse703*(-17.135767342440825*cse1157 + 17.889115159724135*cse1158 - 50.756388852554174*cse1159 + 10.695627721640689*cse1160 + 41.010708268606813*cse1161)
   ))) - state.xn[2]*((cse660) ? (
      cse716*(445.51869310442481*cse1157 + 11324.985706577943*cse1158 + 6766.9891340104432*cse1159 + 38.719649225812766*cse1160 + 2370.6849681533818*cse1161 + 1300.6686834345674*cse1162 + 5869.3152370465659*cse1163 + 11077.448644792414*cse1164)
   )
   : (
      cse717*(4.297112944704045*cse1157 - 101.40241318979159*cse1158 - 43.540996231705549*cse1159 + 3.8689780091986448*cse1160 + 27.91190909651122*cse1161 - 117.34186576967058*cse1162 + 103.76942484394611*cse1163 + 123.18901581604325*cse1164)
   ))))
)) - state.xn[7]*((cse837) ? (
   cse1186*cse825 + cse1190*cse831 + cse1191*cse835 + cse1191*cse836
)
: (
   0
)));
  jac(19, 20) = -9.9081323410184212e-48*std::pow(state.c_hat, 2)*cse1115*std::pow(cse182, 2)*cse949*state.rn[0] + cse1009*cse1125*cse1192 + cse1025*cse1030*cse1192*cse994 + cse1031*cse1125*cse369*cse64*(-cse1033*cse1193 - cse1036*cse1193) - cse1040*(cse1000*cse1194*cse995 - 4.5384267924580226e-10*cse1194*cse994*cse997) + cse1068*cse535 + 5.1822565226214897e-36*cse1192*cse2*cse989/std::pow(cse981, 5.0/6.0);
  jac(20, 1) = cse1197;
  jac(20, 2) = cse1197;
  jac(20, 4) = -cse1198 - cse1199*cse64;
  jac(20, 5) = -cse1198;
  jac(20, 8) = -cse452;
  jac(20, 10) = -cse555;
  jac(20, 11) = cse1197;
  jac(20, 13) = -cse462;
  jac(20, 15) = -cse1196 - cse231;
  jac(20, 16) = -cse1196 - cse237;
  jac(20, 17) = -4.1672335138426642e-21*cse1195;
  jac(20, 20) = cse1202;
  jac(21, 1) = cse1204;
  jac(21, 2) = cse1204;
  jac(21, 4) = -cse1199*cse1207 - cse1206;
  jac(21, 5) = -cse1206;
  jac(21, 8) = -cse1207*cse451;
  jac(21, 10) = -cse1207*cse554;
  jac(21, 11) = cse1204;
  jac(21, 13) = -cse495*state.rn[2*0+1];
  jac(21, 15) = -cse1203 - cse308*state.rn[2*0+1];
  jac(21, 16) = -cse1203 - cse310*state.rn[2*0+1];
  jac(21, 17) = -4.1672335138426642e-21*cse1205;
  jac(21, 21) = cse1202;
  // $JAFF END
  // $JAFF SUB nspec, nbands
  for (int i = 1; i <= INT_NEQS; i++) {
    jac(19, i) /= state.rho;
  }
  // $JAFF END

}

AMREX_INLINE
void actual_rhs_init() {}

#endif
