#ifndef actual_rhs_H
#define actual_rhs_H

#include <AMReX_Array.H>
#include <AMReX_REAL.H>
#include <cmath>

#include <actual_network.H>
#include <burn_type.H>
#include <extern_parameters.H>
#include <gow_interp.H>
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
    // RADIATION { ENABLED -> True }
// }

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
  const amrex::Real cse12 = state.c_hat*state.rn[2*0];
  const amrex::Real cse13 = 1.1464502398585298e-18*cse12;
  const amrex::Real cse14 = state.c_hat*state.rn[2*1];
  const amrex::Real cse15 = 2.7737849364247527e-18*cse14;
  const amrex::Real cse16 = state.c_hat*state.rn[2*2];
  const amrex::Real cse17 = 6.87721070339377e-18*cse16;
  const amrex::Real cse18 = state.c_hat*state.rn[2*3];
  const amrex::Real cse19 = 8.8938650996318353e-19*cse18;
  const amrex::Real cse20 = (cse13 + cse15 + cse17 + cse19)*state.xn[15];
  const amrex::Real cse21 = 3.5000000000000002e-11*state.xn[11]*state.xn[15];
  const amrex::Real cse22 = 1.0/state.T;
  const amrex::Real cse23 = 7.9513696852843427e-10*std::pow(state.T, -0.33900000000000002)*std::exp(0.11*cse22)*state.xn[15]*state.xn[7];
  const amrex::Real cse24 = M_LN10;
  const amrex::Real cse25 = 1.0/cse24;
  const amrex::Real cse26 = 2*state.xn[3];
  const amrex::Real cse27 = cse26 + state.xn[0] + state.xn[10] + state.xn[14] + state.xn[15] + 3*state.xn[16] + state.xn[1] + 2*state.xn[4];
  const amrex::Real cse28 = std::log((1.0/10000.0)*state.T);
  const amrex::Real cse29 = std::pow(cse24, -2);
  const amrex::Real cse30 = std::pow(cse28, 2)*cse29;
  const amrex::Real cse31 = 1.0/cse27;
  const amrex::Real cse32 = cse31*state.xn[3];
  const amrex::Real cse33 = cse31*state.xn[0];
  const amrex::Real cse34 = std::pow(10, 0.41599999999999998*cse25*cse28 + 0.32700000000000001*cse30 - 3)*cse33 + std::pow(10, 1.3*cse25*cse28 - 1.6200000000000001*cse30 - 4.8449999999999998)*cse32;
  const amrex::Real cse35 = cse27*((cse34 < 1.0e-50) ? (
   1.0e-50
)
: (
   cse34
));
  const amrex::Real cse36 = cse25/(cse35 + 1);
  const amrex::Real cse37 = cse35*cse36;
  const amrex::Real cse38 = state.T > 700.0;
  const amrex::Real cse39 = ((cse38) ? (
   std::pow(10, cse36*std::log(5.9959999999999997e-30*std::pow(state.T, 4.1881000000000004)*std::pow(6.7610000000000002e-6*state.T + 1, -5.6881000000000004)*std::exp(-54657.400000000001*cse22)) + cse37*std::log(1.3000000000000001e-9*std::exp(-53300*cse22)))
)
: (
   0.0
))*std::pow(state.xn[3], 2);
  const amrex::Real cse40 = std::max(1.0e-35, 2.0000000000000002e-15*ncol_H2);
  const amrex::Real cse41 = 0.035000000000000003*std::pow(cse40 + 1, -0.5)*std::exp(-0.00084999999999999995*std::sqrt(cse40 + 1.0)) + 0.96499999999999997*std::pow(cse40/std::max(1.0e-50, 1.4142135623730953e-5*vdisp) + 1.0, -2.0);
  const amrex::Real cse42 = cse14*cse41;
  const amrex::Real cse43 = cse16*cse41;
  const amrex::Real cse44 = cse18*cse41;
  const amrex::Real cse45 = 3.5021661223663468e-27*cse42 + 1.3838425473137645e-18*cse43 + 6.45697811687677e-18*cse44;
  const amrex::Real cse46 = (7.5e-10*std::pow(state.T, -0.45800000000000002) + 4.9899999999999997e-11*std::pow(state.T, 0.40500000000000003))*state.xn[0]*state.xn[17];
  const amrex::Real cse47 = ((cse38) ? (
   std::pow(10, cse36*std::log(6.6699999999999996e-12*cse2*std::exp(-63590*cse22 - 1)) + cse37*std::log(3.5199999999999998e-9*std::exp(-43900*cse22)))
)
: (
   0.0
))*state.xn[0];
  const amrex::Real cse48 = std::pow(state.T, -0.52000000000000002);
  const amrex::Real cse49 = cse48*state.xn[16]*state.xn[2];
  const amrex::Real cse50 = cse27*d2g;
  const amrex::Real cse51 = (4.0000000000000001e-10*std::pow(state.T, 0.0066899999999999998) + 1.1000000000000001e-11*std::pow(state.T, 0.51700000000000002))*std::exp(-227*cse22)*state.xn[11]*state.xn[1];
  const amrex::Real cse52 = std::pow(state.T, -1.3);
  const amrex::Real cse53 = std::exp(-23*cse22);
  const amrex::Real cse54 = cse52*cse53*state.xn[3]*state.xn[8];
  const amrex::Real cse55 = 5.4356586967624969e-18*cse12;
  const amrex::Real cse56 = 1.9473397080211686e-18*cse14;
  const amrex::Real cse57 = 1.0937796810239276e-18*cse16;
  const amrex::Real cse58 = 1.2582308138615626e-18*cse18;
  const amrex::Real cse59 = (cse55 + cse56 + cse57 + cse58)*state.xn[14];
  const amrex::Real cse60 = 7.0000000000000004e-11*state.xn[11]*state.xn[14];
  const amrex::Real cse61 = cse59 + cse60;
  const amrex::Real cse62 = (26.218500000000002*std::sqrt(3)*std::sqrt(cse22) + 0.62)*state.xn[15];
  const amrex::Real cse63 = 1.3500000000000001e-9*cse62*state.xn[6];
  const amrex::Real cse64 = state.xn[3]*state.xn[6];
  const amrex::Real cse65 = 1.2599999999999999e-13*cse64*std::exp(-22.5*cse22);
  const amrex::Real cse66 = cse63 + cse65;
  const amrex::Real cse67 = 2.8142984982406731e-11*std::pow(state.T, 0.26000000000000001)*state.xn[0]*state.xn[14];
  const amrex::Real cse68 = 1.7599999999999999e-9*std::pow(state.T, 0.042000000000000003)*std::exp(-2.145922746781116e-5*state.T)*state.xn[3]*state.xn[4];
  const amrex::Real cse69 = -cse67 + cse68;
  const amrex::Real cse70 = crate*state.xn[9];
  const amrex::Real cse71 = 6.5199999999999996*cse70;
  const amrex::Real cse72 = crate*(2.2999999999999998*cse32 + 1.5*cse33);
  const amrex::Real cse73 = cse72*state.xn[0];
  const amrex::Real cse74 = std::log(8.6173000000000003e-5*state.T);
  const amrex::Real cse75 = state.xn[0]*state.xn[2];
  const amrex::Real cse76 = cse75*((cse38) ? (
   5.9078447946382801e-70*std::pow(state.T, 13.536555999999999)*std::exp(-2.0391498499999999e-6*std::pow(cse74, 8) + 0.000111954395*std::pow(cse74, 7) - 0.00263197617*std::pow(cse74, 6) + 0.034825597700000002*std::pow(cse74, 5) - 0.28770560000000001*std::pow(cse74, 4) + 1.56315498*std::pow(cse74, 3) - 5.7393287500000003*std::pow(cse74, 2))
)
: (
   0.0
));
  const amrex::Real cse77 = 1.05849862170431e-5*std::pow(state.T, -0.64000000000000001)*state.xn[10]*state.xn[2];
  const amrex::Real cse78 = 4.8813573300130879e-6*std::pow(cse22, 1.5)*std::pow(114.81344434488582*std::pow(cse22, 0.40699999999999997) + 1, -2.242)*state.xn[1]*state.xn[2];
  const amrex::Real cse79 = 127.88871425900908*state.rn[2*0] + 175.16462312221924*state.rn[2*1] + 210.85430541663879*state.rn[2*2] + 232.36781247130679*state.rn[2*3];
  const amrex::Real cse80 = state.c_hat*cse79;
  const amrex::Real cse81 = cse3*cse80;
  const amrex::Real cse82 = 5.670589618368584e-11*cse81;
  const amrex::Real cse83 = std::log(state.T);
  const amrex::Real cse84 = 1.2250000000000001e-11*cse50*state.xn[1]/(6.1318063394789369e-20*std::pow(cse81, 1.3779999999999999)*(508.69999999999999*std::pow(cse82, -1.102e-5*cse83 - 0.4723)*std::pow(state.T, 0.015859999999999999) + 1) + 1);
  const amrex::Real cse85 = -cse71 - cse73 - cse76 + cse77 + cse78 + cse84;
  const amrex::Real cse86 = cse46 - cse51;
  const amrex::Real cse87 = 6.5847540874715911e-18*cse12;
  const amrex::Real cse88 = 4.4805860217826794e-17*cse14;
  const amrex::Real cse89 = 2.062878503133515e-17*cse16;
  const amrex::Real cse90 = 1.7217538808319277e-17*cse18;
  const amrex::Real cse91 = (cse87 + cse88 + cse89 + cse90)*state.xn[12];
  const amrex::Real cse92 = cse26*cse72;
  const amrex::Real cse93 = 1.4595000290776864e-10*std::pow(state.T, -0.62)*state.xn[13]*state.xn[2];
  const amrex::Real cse94 = 2.1659999999999999e-12*cse50*state.xn[13]/(3.5685736926834518e-27*std::pow(cse81, 1.8740000000000001)*(43750.0*std::pow(cse82, -7.538e-5*cse83 - 0.89639999999999997)*std::pow(state.T, 1.635e-6) + 1) + 1);
  const amrex::Real cse95 = 16800.0*crate*cse32*state.xn[12];
  const amrex::Real cse96 = (1.0537934368240927e-9*std::pow(cse22, 0.00231) + std::pow(state.T, -1.5)*(0.00015100000000000001*std::exp(-8110.0*cse22) + 1.31e-7*std::exp(-26.600000000000001*cse22) + 3.4e-8*std::exp(-7.6200000000000001*cse22) + 6.9699999999999997e-9*std::exp(-1.3799999999999999*cse22)))*state.xn[16]*state.xn[7];
  const amrex::Real cse97 = cse31*state.xn[2];
  const amrex::Real cse98 = ((cse97 < 1.0e-50) ? (
   10000000000.0
)
: (
   cse4
));
  const amrex::Real cse99 = cse10*cse98/(cse98 + 1);
  const amrex::Real cse100 = cse11 + 1.3007021411393064e-6*cse49 + cse96 + cse99;
  const amrex::Real cse101 = cse25*cse83;
  const amrex::Real cse102 = -1.1000000000000001*crate*state.xn[5] + 5.5720000000000001e-12*cse50*state.xn[6]/(1.0246966183388124e-22*std::pow(cse81, 1.512)*(5115.0*std::pow(cse82, -5.4939999999999998e-7*cse83 - 0.49559999999999998)*std::pow(state.T, 3.9029999999999999e-7) + 1) + 1) + 9.9999999999999994e-12*std::pow(state.T, -0.5)*(cse101*(cse101*(0.044330000000000001*cse101 - 0.28520000000000001) - 1.6759999999999999) + 11.19)*state.xn[2]*state.xn[6];
  const amrex::Real cse103 = interp_8_shielding_H2(ncol_H2)*interp_8_shielding_self(ncol_C);
  const amrex::Real cse104 = 1.3061417322445169e-17*cse103;
  const amrex::Real cse105 = cse104*cse16;
  const amrex::Real cse106 = 1.7033009950880063e-17*cse103;
  const amrex::Real cse107 = cse106*cse18;
  const amrex::Real cse108 = (cse105 + cse107)*state.xn[7];
  const amrex::Real cse109 = crate*state.xn[7];
  const amrex::Real cse110 = 3.8500000000000001*cse109;
  const amrex::Real cse111 = 0.15970000000000001*std::exp(-49550.0*cse22);
  const amrex::Real cse112 = ((1.6339999999999999e-6*std::exp(-15230.0*cse22) + 9.7930000000000004e-9*std::exp(-73.799999999999997*cse22) + 6.3460000000000002e-9*std::exp(-12.17*cse22))/std::pow(state.T, 3.0/2.0) + 2.4460185352936309e-10*std::pow(0.00071740367932898903*cse2 + 1, -cse111 - 1.7848999999999999)*std::pow(12.244387999457523*cse2 + 1, cse111 - 0.21509999999999996)/cse2)*state.xn[2]*state.xn[8];
  const amrex::Real cse113 = 4.5579999999999993e-11*cse50*state.xn[8]/(1.6851501542381549e-14*std::pow(cse81, 1.1279999999999999)*(433.10000000000002*std::pow(cse82, -0.00013329999999999999*cse83 - 0.81200000000000006)*std::pow(state.T, 0.04845) + 1) + 1);
  const amrex::Real cse114 = 1040.0*cse109*cse32;
  const amrex::Real cse115 = -cse108 - cse110 + cse112 + cse113 - cse114;
  const amrex::Real cse116 = 3.2999999999999996e-13*cse54;
  const amrex::Real cse117 = cse8 + cse9;
  const amrex::Real cse118 = cse116 + cse117;
  const amrex::Real cse119 = -cse96;
  const amrex::Real cse120 = -cse99;
  const amrex::Real cse121 = 1.6999999999999999e-9*state.xn[16]*state.xn[9];
  const amrex::Real cse122 = cse47*state.xn[3];
  const amrex::Real cse123 = 7.2000000000000002e-15*cse64;
  const amrex::Real cse124 = cse1 + cse123 + cse92;
  const amrex::Real cse125 = -cse68;
  const amrex::Real cse126 = 1.6000000000000001e-9*state.xn[6]*state.xn[9];
  const amrex::Real cse127 = cse102 + cse123 + cse126 + cse66;
  const amrex::Real cse128 = cse119 + cse67;
  const amrex::Real cse129 = interp_10_shielding_H2(ncol_H2)*interp_10_shielding_self(ncol_CO);
  const amrex::Real cse130 = 5.91270072314384e-18*cse129;
  const amrex::Real cse131 = cse130*cse16;
  const amrex::Real cse132 = 3.0014835315191357e-17*cse129;
  const amrex::Real cse133 = cse132*cse18;
  const amrex::Real cse134 = (cse131 + cse133)*state.xn[9];
  const amrex::Real cse135 = 184.0*cse32*cse70;
  const amrex::Real cse136 = cse134 + cse135 - cse23;
  const amrex::Real cse137 = 9.1500000000000005e-10*cse62*state.xn[8];
  const amrex::Real cse138 = cse121 + cse71 - cse77;
  const amrex::Real cse139 = cse126 - cse60;
  const amrex::Real cse140 = cse120 + cse20 + cse21;
  const amrex::Real cse141 = cse91 - cse93 - cse94 + cse95;
  const amrex::Real cse142 = std::pow(state.T, 0.14999999999999999)/(0.00088240812266906664*std::pow(state.T, 0.65000000000000002) + 1);
  const amrex::Real cse143 = cse142*state.xn[2];
  const amrex::Real cse144 = std::exp(-118400.0*cse22)/(4.4460389837719569e-9*cse143 + 626500000.0);
  const amrex::Real cse145 = cse82 + 50;
  const amrex::Real cse146 = std::log(5.670589618368584e-11*cse2*cse80/(state.xn[2] + 1.0e-50) + 50);
  const amrex::Real cse147 = std::sqrt(cse22)*state.xn[2];
  const amrex::Real cse148 = (1.0/100.0)*state.T;
  const amrex::Real cse149 = std::log(cse148);
  const amrex::Real cse150 = state.T < 500;
  const amrex::Real cse151 = std::pow(state.T, 0.070000000000000007);
  const amrex::Real cse152 = 4.5299999999999998e-6*cse147 + 7.5799999999999997e-10*std::pow(cse148, 0.0086999999999999994*cse149 + 0.12809999999999999)*state.xn[0] + (0.75*((cse150) ? (
   1.1000000000000001e-13*state.T + 5.3300000000000002e-10
)
: (
   3.74757785025e-10*cse151
)) + 0.25*((cse150) ? (
   3.3000000000000001e-13*state.T + 4.4299999999999997e-10
)
: (
   3.8899728635600003e-10*cse151
)))*state.xn[3];
  const amrex::Real cse153 = cse152*std::exp(-91.209999999999994*cse22)/(cse152 + 2.3e-6);
  const amrex::Real cse154 = 0.001*state.T;
  const amrex::Real cse155 = cse154 + 1;
  const amrex::Real cse156 = std::pow(cse155, -2);
  const amrex::Real cse157 = 1.0/cse155;
  const amrex::Real cse158 = cse31/(std::pow(10, -2.3900000000000001*cse156 + 0.055500000000000001*cse157 - 11.06)*cse33 + std::pow(10, -2.0230000000000001*cse156 - 3.6709999999999998*cse157 - 11.08)*cse32);
  const amrex::Real cse159 = cse25*std::log(cse27);
  const amrex::Real cse160 = std::pow(state.T, 1.0/4.0);
  const amrex::Real cse161 = 1.75*cse160*state.xn[0] + state.xn[3] + 680.10000000000002*state.xn[2]/cse160;
  const amrex::Real cse162 = l0_co_interp1d(state.T);
  const amrex::Real cse163 = 1.0/cse162;
  const amrex::Real cse164 = ((ncol_H <= 0) ? (
   100000.0*state.xn[9]/std::max(gradv, 5.7445531974402559e-18*M_SQRT2*cse2)
)
: (
   1756701454.4310031*ncol_CO/std::sqrt(std::max(1.0000000000000001e-30, cse31*ncol_H))
));
  const amrex::Real cse165 = 1.0/llte_co_interp2d(state.T, cse164);
  const amrex::Real cse166 = nhalf_co_interp2d(state.T, cse164);
  const amrex::Real cse167 = 0.042999999999999997*cse149;
  const amrex::Real cse168 = 4.34e-10*std::pow(cse148, 0.755 - 0.16*cse149)*state.xn[0] + 1.08e-14*std::pow(state.T, 0.92600000000000005)*state.xn[2] + (2.2500000000000003e-12*std::pow(cse148, 0.52500000000000002*cse149 + 1.198) + 5.2499999999999998e-13*std::pow(cse148, cse167 + 0.88900000000000001))*state.xn[3];
  const amrex::Real cse169 = 3.5700000000000001e-10*std::pow(cse148, 0.41899999999999998 - 0.0030000000000000001*cse149)*state.xn[0] + 5.1199999999999999e-10*std::pow(state.T, -0.074999999999999997)*state.xn[2] + (3.7250000000000001e-11*std::pow(cse148, 0.025000000000000001*cse149 + 0.26400000000000001) + 1.0274999999999999e-10*std::pow(cse148, cse167 + 0.29599999999999999))*state.xn[3];
  const amrex::Real cse170 = (1.0/5.0)*std::exp(-326.60000000000002*cse22);
  const amrex::Real cse171 = 3.1899999999999998e-10*std::pow(cse148, 0.36899999999999999 - 0.0060000000000000001*cse149)*state.xn[0] + 4.8599999999999998e-10*std::pow(state.T, -0.025999999999999999)*state.xn[2] + (4.7499999999999998e-11*std::pow(cse148, 0.041000000000000002*cse149 + 0.20300000000000001) + 1.6725000000000002e-10*std::pow(cse148, 0.058000000000000003*cse149 + 0.23699999999999999))*state.xn[3];
  const amrex::Real cse172 = cse169 + 8.9099999999999997e-5;
  const amrex::Real cse173 = cse170*cse171;
  const amrex::Real cse174 = cse168*cse169*cse170 + (1.0/15.0)*cse168*cse171*std::exp(-425.40000000000003*cse22) + cse172*cse173;
  const amrex::Real cse175 = cse168 + 1.7499999999999998e-5;
  const amrex::Real cse176 = cse171 + 1.34e-10;
  const amrex::Real cse177 = (3.0/5.0)*cse169*std::exp(-227.80000000000001*cse22);
  const amrex::Real cse178 = cse173*cse175 + cse175*cse177 + cse176*cse177;
  const amrex::Real cse179 = 1.0/((1.0/3.0)*cse168*cse176*std::exp(-98.800000000000011*cse22) + cse172*cse175 + cse172*cse176 + cse174 + cse178);
  const amrex::Real cse180 = state.T < 100.0;
  const amrex::Real cse181 = state.T < 6000.0;
  const amrex::Real cse182 = state.T < 10;
  const amrex::Real cse183 = std::log(cse154);
  const amrex::Real cse184 = cse183*cse25;
  const amrex::Real cse185 = std::pow(cse24, -8);
  const amrex::Real cse186 = std::pow(cse183, 8)*cse185;
  const amrex::Real cse187 = std::pow(cse24, -7);
  const amrex::Real cse188 = std::pow(cse183, 7)*cse187;
  const amrex::Real cse189 = std::pow(cse24, -6);
  const amrex::Real cse190 = std::pow(cse183, 6)*cse189;
  const amrex::Real cse191 = std::pow(cse24, -5);
  const amrex::Real cse192 = std::pow(cse183, 5)*cse191;
  const amrex::Real cse193 = std::pow(cse24, -4);
  const amrex::Real cse194 = std::pow(cse183, 4)*cse193;
  const amrex::Real cse195 = std::pow(cse24, -3);
  const amrex::Real cse196 = std::pow(cse183, 3);
  const amrex::Real cse197 = cse195*cse196;
  const amrex::Real cse198 = std::pow(cse183, 2)*cse29;
  const amrex::Real cse199 = std::pow(10.0, 16.815729999999999*cse184 + 70.609154000000004*cse186 + 364.14445999999998*cse188 + 801.81246999999996*cse190 + 983.67575999999997*cse192 + 734.71650999999997*cse194 + 343.1918*cse197 + 96.743155000000002*cse198 - 21.928795999999998)*state.xn[2];
  const amrex::Real cse200 = state.xn[5] + state.xn[6];
  const amrex::Real cse201 = std::pow(10.0, 1.5714710999999999*cse184 + 0.32168730000000001*cse192 - 0.51002221000000003*cse194 - 0.23619984999999999*cse197 + 0.015391166*cse198 - 22.089523)*state.xn[1] + std::pow(10.0, 2.1892372*cse184 + 0.19191374999999999*cse192 - 0.16596184*cse194 + 0.29036281000000003*cse197 - 0.81520437999999995*cse198 - 23.689236999999999)*cse200 + std::pow(10.0, 2.0943374000000001*cse183*cse25 - 0.033638326000000003*cse192 - 0.14913216000000001*cse194 + 0.43693353000000001*cse195*cse196 - 0.77151435999999995*cse198 - 23.962112000000001)*state.xn[3];
  const amrex::Real cse202 = std::pow(10, 37.383713*cse184 + 3.8479610000000002*cse192 + 20.159831000000001*cse194 + 48.656103000000002*cse197 + 58.145166000000003*cse198 - 16.818342000000001)*state.xn[0] + cse201;
  const amrex::Real cse203 = std::pow(10.0, 1.6802758*cse184 - 6.3701156000000001*cse186 + 6.4380698000000001*cse188 + 8.9167182999999994*cse190 - 8.8077017000000009*cse192 - 4.7274035999999997*cse194 + 4.0406627000000004*cse197 + 0.93310621999999999*cse198 - 22.921188999999998)*state.xn[2];
  const amrex::Real cse204 = state.T < 1000.0;
  const amrex::Real cse205 = std::pow(10.0, 3.5692468000000002*cse183*cse25 - 4.2519023000000002*cse192 - 21.328264000000001*cse194 - 27.850082*cse197 - 11.33286*cse198 - 24.311209000000002)*state.xn[0] + cse201;
  const amrex::Real cse206 = std::pow(10.0, 4.6450521*cse184 + 1.5538288*cse192 - 5.5108049000000001*cse194 + 5.9369081000000001*cse197 - 3.7209846*cse198 - 24.311209000000002)*state.xn[0] + cse201;
  const amrex::Real cse207 = std::pow(10.0, -0.62120210234156448*cse191 - 1.5370575959566981*cse193 + 2.5133588394664677*cse195 + 3.7525488682084647*cse25 - 2.4768712409536775*cse29 - 23.962112000000001)*state.xn[3] + std::pow(10.0, 3.5440891133599637*cse191 - 1.7105157385968939*cse193 + 1.6702447513374001*cse195 + 3.9225864834863131*cse25 - 2.6171337683481011*cse29 - 23.689236999999999)*cse200 + std::pow(10.0, 5.9406293599919788*cse191 - 5.2566362076906961*cse193 - 1.3586848802337363*cse195 + 2.8156982240432273*cse25 + 0.049411830040524525*cse29 - 22.089523)*state.xn[1] + std::pow(10.0, 28.69470131298657*cse191 - 56.79810800956944*cse193 + 34.150687525022562*cse195 + 8.3228160852326614*cse25 - 11.945856385319289*cse29 - 24.311209000000002)*state.xn[0];
  const amrex::Real cse208 = ((cse181) ? (
   cse22
)
: (
   0.00016666666666666666
));
  const amrex::Real cse209 = 1.6e-18*std::exp(-11700.0*cse208) + 6.7e-19*std::exp(-5860.0*cse208) + 3.0e-24*std::exp(-510.0*cse208) + 4.985670872372847e-33*((cse181) ? (
   std::pow(state.T, 3.7599999999999998)
)
: (
   160638152785302.53
))*std::exp(-2197000.0*((cse181) ? (
   std::pow(state.T, -3)
)
: (
   4.6296296296296294e-12
)))/(6.0142468035272636e-8*((cse181) ? (
   std::pow(state.T, 2.1000000000000001)
)
: (
   85924621.118090734
)) + 1.0);
  const amrex::Real cse210 = std::log(0.001*((cse181) ? (
   state.T
)
: (
   6000.0
)));
  const amrex::Real cse211 = cse191*std::pow(cse210, 5);
  const amrex::Real cse212 = cse193*std::pow(cse210, 4);
  const amrex::Real cse213 = std::pow(cse210, 3);
  const amrex::Real cse214 = std::pow(cse210, 2)*cse29;
  const amrex::Real cse215 = cse210*cse25;
  const amrex::Real cse216 = cse195*cse213;
  const amrex::Real cse217 = cse185*std::pow(cse210, 8);
  const amrex::Real cse218 = cse187*std::pow(cse210, 7);
  const amrex::Real cse219 = cse189*std::pow(cse210, 6);
  const amrex::Real cse220 = 0.045999999999999999*cse149;
  const amrex::Real cse221 = 1.7258000000000001e-6*cse147;
  const amrex::Real cse222 = 8.9000000000000003e-11*std::pow(cse148, cse220 + 0.22800000000000001)*state.xn[0] + cse221*std::exp(((cse204) ? (
   cse83*(cse83*(cse83*(0.0070527699999999999*cse83 - 0.11133800000000001) + 0.69763799999999998) - 1.3074300000000001) - 7.6973500000000001
)
: (
   cse83*(cse83*(cse83*(0.093813800000000003*cse83 - 3.0328300000000001) + 36.180300000000003) - 187.47399999999999) + 350.60899999999998
))) + (5.1749999999999996e-11*std::pow(cse148, 0.037999999999999999*cse149 + 0.16900000000000001) + 2.15e-11*std::pow(cse148, 0.048000000000000001*cse149 - 0.01))*state.xn[3];
  const amrex::Real cse223 = 2.6400000000000002e-10*std::pow(cse148, cse220 + 0.23100000000000001)*state.xn[0] + cse221*std::exp(((cse204) ? (
   cse83*(cse83*(cse83*(0.00235272*cse83 - 0.041816600000000002) + 0.35826400000000003) - 0.57443) - 7.4386999999999999
)
: (
   cse83*(cse83*(cse83*(0.097857299999999994*cse83 - 3.1926800000000002) + 38.504899999999999) - 202.19300000000001) + 386.18599999999998
))) + (1.11e-10*std::pow(cse148, 0.031*cse149 + 0.26300000000000001) + 4.3749999999999998e-11*std::pow(cse148, 0.064000000000000001*cse149 + 0.071999999999999995))*state.xn[3];
  const amrex::Real cse224 = 2.8763333333333333e-6*cse147*std::exp(((cse204) ? (
   cse83*(cse83*(cse83*(-0.00065632500000000003*cse83 - 0.015089200000000001) + 0.36118400000000001) - 0.77378199999999997) - 9.2514099999999999
)
: (
   cse83*(cse83*(cse83*(0.10508000000000001*cse83 - 3.4762) + 42.594999999999999) - 227.91300000000001) + 444.60000000000002
))) + 1.26e-10*std::pow(cse148, 0.057000000000000002*cse149 + 0.115)*state.xn[0] + (5.3249999999999997e-11*std::pow(cse148, 0.049000000000000002*cse149 - 0.0040000000000000001) + 1.675e-11*std::pow(cse148, 0.10199999999999999*cse149 - 0.085000000000000006))*state.xn[3];
  const amrex::Real cse225 = 5*std::exp(-62.460000000000001*cse22);
  const amrex::Real cse226 = cse224 + 7.8800000000000004e-8;
  const amrex::Real cse227 = cse222*cse225;
  const amrex::Real cse228 = (25.0/3.0)*cse222*cse223*std::exp(-101.30000000000001*cse22) + cse223*cse224*cse225 + cse226*cse227;
  const amrex::Real cse229 = cse223 + 2.65e-7;
  const amrex::Real cse230 = cse222 + 1.81e-14;
  const amrex::Real cse231 = 3*cse224*std::exp(-23.619999999999997*cse22);
  const amrex::Real cse232 = cse227*cse229 + cse229*cse231 + cse230*cse231;
  const amrex::Real cse233 = 1.0/((5.0/3.0)*cse223*cse230*std::exp(-38.840000000000003*cse22) + cse226*cse229 + cse226*cse230 + cse228 + cse232);
  const amrex::Real cse234 = Zd*cse27;
  const amrex::Real cse235 = 1.1627326161928671e-21*cse234;
  const amrex::Real cse236 = state.c_hat*state.rn[2*0+1];
  const amrex::Real cse237 = 5.3576915465847796e-20*state.xn[3];
  const amrex::Real cse238 = 1.4208709462150382e-21*cse234;
  const amrex::Real cse239 = state.c_hat*state.rn[2*1+1];
  const amrex::Real cse240 = 1.8804775722560676e-17*state.xn[3];
  const amrex::Real cse241 = 1.887632855452075e-21*cse234;
  const amrex::Real cse242 = state.c_hat*state.rn[2*2+1];
  const amrex::Real cse243 = 3.5957426289008978e-17*state.xn[3];
  const amrex::Real cse244 = 2.2658762392657621e-21*cse234;
  const amrex::Real cse245 = state.c_hat*state.rn[2*3+1];
  ydot(1) = cse1 + cse11 + cse20 + cse21 + cse23 + cse26*cse45 + cse26*cse47 + 2*cse39 - cse46 + 2.9935562710997168e-6*cse49 - 6.0000000000000005e-15*cse50*state.xn[0] + cse51 + 4.2899999999999994e-13*cse54 + cse61 + cse66 + cse69 + 3.2000000000000001e-9*cse7 + cse85 + cse9;
  ydot(2) = cse0 + cse65 + cse73 + cse76 - cse78 - cse84 + cse86;
  ydot(3) = -cse100 - cse102 - cse115 - cse118 - cse85 + cse91 + cse92 - cse93 - cse94 + cse95;
  ydot(4) = cse11 - cse118 - cse119 - cse120 + cse121 - cse122 - cse124 + 3.0000000000000002e-15*cse27*d2g*state.xn[0] - cse39 - cse45*state.xn[3] + 4.5427507615910105e-7*cse48*state.xn[16]*state.xn[2] - cse65 - cse69;
  ydot(5) = cse124 + cse125;
  ydot(6) = cse127;
  ydot(7) = -cse127;
  ydot(8) = cse115 + cse128 + cse136 + 9.8999999999999995e-14*cse54 + cse59;
  ydot(9) = cse108 + cse110 - cse112 - cse113 + cse114 - cse116 + cse126 - cse137;
  ydot(10) = -cse136 - cse138 - cse139;
  ydot(11) = cse137 + cse138;
  ydot(12) = cse134 + cse135 + cse139 + cse140 + cse8 + cse86;
  ydot(13) = -cse141;
  ydot(14) = cse141;
  ydot(15) = -cse128 + 2.3099999999999997e-13*cse52*cse53*state.xn[3]*state.xn[8] - cse61;
  ydot(16) = -cse137 - cse140 - cse23 - cse63 + cse9;
  ydot(17) = -cse100 - cse121 - cse125;
  ydot(18) = -cse117 + cse63 - cse86;
  ydot(19) = (5.6705896183685846e-35*state.c_hat*cse27*cse79*d2g*(2.25*std::pow(state.T, 0.14699999999999999) + 5.2199999999999998)/(0.049959999999999997*std::pow(cse145, 0.43099999999999999)*(0.0043*std::pow(cse145, 0.69199999999999995) + 1) + 1) + 1.602177e-12*crate*(26.399999999999999*std::sqrt(cse97/(cse97 + 0.070000000000000007)) + 6.5)*state.xn[0] + 3.2043540000000001e-12*crate*((cse159 < 2) ? (
   10.0
)
: ((cse159 < 4) ? (
   (3.0/2.0)*cse159 + 7
)
: ((cse159 < 7) ? (
   (4.0/3.0)*cse159 + 23.0/3.0
)
: ((cse159 < 10) ? (
   (1.0/3.0)*cse159 + 44.0/3.0
)
: (
   18.0
)))))*state.xn[3] - 7.1777529600000012e-12*cse122 - 1.3659996292638161e-10*cse142*cse144*cse75/(1.3338116951315871e-8*cse143*cse144 + 1) - 5.7927337833400002e-20*cse153*state.xn[8]/(2*cse153 + 1) - cse161*std::pow(1 - std::exp(-state.T), 1000)*state.xn[9]/(cse161*cse165 + cse163*std::pow(cse161/cse166, alpha_co_interp2d(state.T, cse164))*(-cse162*cse165*cse166 + 1) + cse163) + 3.0000000000000002e-15*cse27*d2g*(3.2043540000000002e-13 + 6.7291434000000005e-12/(cse158*(3.0000000000000002e-15*cse50/state.xn[0] + 1.9999999999999999e-7) + 1))*state.xn[0] - 7.1777529600000012e-12*cse39 + cse45*(6.4087080000000003e-13 + 2.5634832e-11/(cse158*(cse45 + 1.9999999999999999e-7) + 1))*state.xn[3] - 1.8998126555754159e-29*cse50*std::pow(state.T, 0.45350000000000001 + 2.234/cse146)*std::exp(cse146*(1.4419999999999999 - 0.050889999999999998*cse146))*state.xn[2] - 2.1789607199999999e-11*cse76 - (2.3872025442750958e-19*cse174*cse179 + 2.8023005140019997e-18*cse178*cse179)*state.xn[11] - ((((cse150 && cse180 && cse181) ? (
   cse182 || cse199 + cse202 < 1.0e-100
)
: (
   ((cse180 && cse181) ? (
      cse182 || cse202 + cse203 < 1.0e-100
   )
   : (
      ((cse150 && cse181 && cse204) ? (
         cse182 || cse199 + cse205 < 1.0e-100
      )
      : (
         ((cse181 && cse204) ? (
            cse182 || cse203 + cse205 < 1.0e-100
         )
         : (
            ((cse150 && cse181) ? (
               cse182 || cse199 + cse206 < 1.0e-100
            )
            : (
               ((cse181) ? (
                  cse182 || cse203 + cse206 < 1.0e-100
               )
               : (
                  ((cse150) ? (
                     cse182 || std::pow(10.0, 7500.646099484944*cse185 + 21588.957206746247*cse187 + 26530.843406310876*cse189 + 18165.631967965237*cse191 + 7572.4886742758581*cse193 + 1974.131269262874*cse195 + 30.129743459482278*cse25 + 310.58441786958315*cse29 - 21.928795999999998)*state.xn[2] + cse207 < 1.0e-100
                  )
                  : (
                     cse182 || std::pow(10.0, -676.68255490510751*cse185 + 381.69251128040054*cse187 + 295.04162851880631*cse189 - 162.65264843551881*cse191 - 48.723840736518795*cse193 + 23.242975457496804*cse195 + 3.0106500755647456*cse25 + 2.9956460707652877*cse29 - 22.921188999999998)*state.xn[2] + cse207 < 1.0e-100
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
   cse209/(cse209/(std::pow(10.0, 0.19191374999999999*cse211 - 0.16596184*cse212 - 0.81520437999999995*cse214 + 2.1892372*cse215 + 0.29036281000000003*cse216 - 23.689236999999999)*cse200 + std::pow(10.0, 0.32168730000000001*cse211 - 0.51002221000000003*cse212 + 0.015391166*cse214 + 1.5714710999999999*cse215 - 0.23619984999999999*cse216 - 22.089523)*state.xn[1] + std::pow(10.0, 0.43693353000000001*cse195*cse213 + 2.0943374000000001*cse210*cse25 - 0.033638326000000003*cse211 - 0.14913216000000001*cse212 - 0.77151435999999995*cse214 - 23.962112000000001)*state.xn[3] + ((cse150) ? (
      std::pow(10.0, 983.67575999999997*cse211 + 734.71650999999997*cse212 + 96.743155000000002*cse214 + 16.815729999999999*cse215 + 343.1918*cse216 + 70.609154000000004*cse217 + 364.14445999999998*cse218 + 801.81246999999996*cse219 - 21.928795999999998)
   )
   : (
      std::pow(10.0, -8.8077017000000009*cse211 - 4.7274035999999997*cse212 + 0.93310621999999999*cse214 + 1.6802758*cse215 + 4.0406627000000004*cse216 - 6.3701156000000001*cse217 + 6.4380698000000001*cse218 + 8.9167182999999994*cse219 - 22.921188999999998)
   ))*state.xn[2] + ((((cse181) ? (
      cse180
   )
   : (
      false
   ))) ? (
      std::pow(10, 3.8479610000000002*cse211 + 20.159831000000001*cse212 + 58.145166000000003*cse214 + 37.383713*cse215 + 48.656103000000002*cse216 - 16.818342000000001)
   )
   : ((((cse181) ? (
      cse204
   )
   : (
      false
   ))) ? (
      std::pow(10.0, 3.5692468000000002*cse210*cse25 - 4.2519023000000002*cse211 - 21.328264000000001*cse212 - 11.33286*cse214 - 27.850082*cse216 - 24.311209000000002)
   )
   : (
      std::pow(10.0, 1.5538288*cse211 - 5.5108049000000001*cse212 - 3.7209846*cse214 + 4.6450521*cse215 + 5.9369081000000001*cse216 - 24.311209000000002)
   )))*state.xn[0]) + 1)
))*state.xn[3] - ((state.T < 1000000.0) ? (
   1.4210469458259591e-21*cse228*cse233 + 2.5697412351439998e-22*cse232*cse233
)
: (
   0.0
))*state.xn[7] - std::min(100.0*cse50*psigd_coll_interp2d(cse27, state.T), 3.2000000000000002e-32*cse2*std::pow(cse27, 2)*d2g*(state.T - 2.1544346900318838*std::pow(2.8677558904086741e-6*cse80 + 4.1397668473788896, 1.0/6.0))))/(1.6737729999999998e-24*state.xn[0] + 4.8185222061629005e-23*state.xn[10] + 2.6567630000000005e-23*state.xn[11] + 4.6637070000000001e-23*state.xn[12] + 4.6636159061629004e-23*state.xn[13] + 2.1618503000000003e-23*state.xn[14] + 2.8241403000000005e-23*state.xn[15] + 5.0204080616289998e-24*state.xn[16] + 2.6566719061629008e-23*state.xn[17] + 1.6728620616289998e-24*state.xn[1] + 9.1093837099999994e-28*state.xn[2] + 3.3475459999999996e-24*state.xn[3] + 3.3466350616289996e-24*state.xn[4] + 6.6464729999999996e-24*state.xn[5] + 6.6455620616289996e-24*state.xn[6] + 1.9944730000000003e-23*state.xn[7] + 1.9943819061629003e-23*state.xn[8] + 4.6512360000000008e-23*state.xn[9]);
  ydot(20) = -cse12*cse235 - cse13*state.xn[15] - cse55*state.xn[14] - cse87*state.xn[12];
  ydot(21) = -cse235*cse236 - 6.5847540874715911e-18*cse236*state.xn[12] - 5.4356586967624969e-18*cse236*state.xn[14] - 1.1464502398585298e-18*cse236*state.xn[15];
  ydot(22) = -cse14*cse238 - cse15*state.xn[15] - cse237*cse42 - cse56*state.xn[14] - cse88*state.xn[12];
  ydot(23) = -cse237*cse239*cse41 - cse238*cse239 - 4.4805860217826794e-17*cse239*state.xn[12] - 1.9473397080211686e-18*cse239*state.xn[14] - 2.7737849364247527e-18*cse239*state.xn[15];
  ydot(24) = -cse105*state.xn[7] - cse131*state.xn[9] - cse16*cse241 - cse17*state.xn[15] - cse240*cse43 - cse57*state.xn[14] - cse89*state.xn[12];
  ydot(25) = -cse104*cse242*state.xn[7] - cse130*cse242*state.xn[9] - cse240*cse242*cse41 - cse241*cse242 - 2.062878503133515e-17*cse242*state.xn[12] - 1.0937796810239276e-18*cse242*state.xn[14] - 6.87721070339377e-18*cse242*state.xn[15];
  ydot(26) = -cse107*state.xn[7] - cse133*state.xn[9] - cse18*cse244 - cse19*state.xn[15] - cse243*cse44 - cse58*state.xn[14] - cse90*state.xn[12];
  ydot(27) = -cse106*cse245*state.xn[7] - cse132*cse245*state.xn[9] - cse243*cse245*cse41 - cse244*cse245 - 1.7217538808319277e-17*cse245*state.xn[12] - 1.2582308138615626e-18*cse245*state.xn[14] - 8.8938650996318353e-19*cse245*state.xn[15];
  // $$JAFF END
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


  // $$JAFF REPEAT idx, expr, cse IN jacobian { USE microphysics }
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
  const amrex::Real cse61 = 1.6737729999999998e-24*state.xn[0] + 4.8185222061629005e-23*state.xn[10] + 2.6567630000000005e-23*state.xn[11] + 4.6637070000000001e-23*state.xn[12] + 4.6636159061629004e-23*state.xn[13] + 2.1618503000000003e-23*state.xn[14] + 2.8241403000000005e-23*state.xn[15] + 5.0204080616289998e-24*state.xn[16] + 2.6566719061629008e-23*state.xn[17] + 1.6728620616289998e-24*state.xn[1] + 9.1093837099999994e-28*state.xn[2] + 3.3475459999999996e-24*state.xn[3] + 3.3466350616289996e-24*state.xn[4] + 6.6464729999999996e-24*state.xn[5] + 6.6455620616289996e-24*state.xn[6] + 1.9944730000000003e-23*state.xn[7] + 1.9943819061629003e-23*state.xn[8] + 4.6512360000000008e-23*state.xn[9];
  const amrex::Real cse62 = 2.0709734999998963e-16*state.T/cse61;
  const amrex::Real cse63 = 1.380649e-16*state.xn[0] + 1.380649e-16*state.xn[10] + 1.380649e-16*state.xn[11] + 1.380649e-16*state.xn[12] + 1.380649e-16*state.xn[13] + 1.380649e-16*state.xn[14] + 1.380649e-16*state.xn[15] + 1.380649e-16*state.xn[16] + 1.380649e-16*state.xn[17] + 1.380649e-16*state.xn[1] + 1.380649e-16*state.xn[2] + 1.380649e-16*state.xn[3] + 1.380649e-16*state.xn[4] + 1.380649e-16*state.xn[5] + 1.380649e-16*state.xn[6] + 1.380649e-16*state.xn[7] + 1.380649e-16*state.xn[8] + 1.380649e-16*state.xn[9];
  const amrex::Real cse64 = cse63*state.T/std::pow(cse61, 2);
  const amrex::Real cse65 = cse62 - 2.5106594999998743e-24*cse64;
  const amrex::Real cse66 = std::pow(state.T, -2);
  const amrex::Real cse67 = std::exp(-22.5*cse3);
  const amrex::Real cse68 = 2.8349999999999999e-12*cse66*cse67*state.xn[3]*state.xn[6];
  const amrex::Real cse69 = -cse68;
  const amrex::Real cse70 = 7.3171760954257497e-12*std::pow(state.T, -0.73999999999999999)*state.xn[0]*state.xn[14];
  const amrex::Real cse71 = std::sqrt(cse3);
  const amrex::Real cse72 = std::sqrt(3)*cse3*cse71*state.xn[15];
  const amrex::Real cse73 = 1.7697487500000004e-8*cse72*state.xn[6];
  const amrex::Real cse74 = cse26*cse66;
  const amrex::Real cse75 = cse13*cse3;
  const amrex::Real cse76 = cse12*cse75;
  const amrex::Real cse77 = cse3*cse8;
  const amrex::Real cse78 = ((cse20) ? (
   0
)
: (
   cse16*cse7*(1.3*cse3*cse8 - 3.2400000000000002*cse76) + cse18*cse7*(0.65400000000000003*cse76 + 0.41599999999999998*cse77)
));
  const amrex::Real cse79 = cse10*cse78;
  const amrex::Real cse80 = cse24*cse31;
  const amrex::Real cse81 = cse50*cse79;
  const amrex::Real cse82 = cse21*cse34*cse50*cse78;
  const amrex::Real cse83 = ((cse28) ? (
   cse54*(1.6677785190126752e+29*cse24*cse45*(-2.3058963562359999e-34*cse43*std::pow(cse44, -6.6881000000000004)*cse47 + 3.2772577039999999e-25*cse48*std::pow(state.T, 2.1881000000000004) + 2.51118476e-29*cse48*std::pow(state.T, 3.1881000000000004))*std::exp(cse46)/cse43 - cse31*cse82 - cse49*cse81 + 53300.0*cse74 + cse79*cse80)
)
: (
   0
));
  const amrex::Real cse84 = state.xn[0]*state.xn[17]*(-3.435e-10*std::pow(state.T, -1.458) + 2.0209499999999999e-11*std::pow(state.T, -0.59499999999999997));
  const amrex::Real cse85 = 1.0/cse2;
  const amrex::Real cse86 = std::pow(state.T, -3.0/2.0);
  const amrex::Real cse87 = cse24*cse25;
  const amrex::Real cse88 = ((cse28) ? (
   cse59*(149925037481.25937*cse24*cse85*(3.3349999999999998e-12*cse5*cse85 + 4.2414529999999999e-7*cse5*cse86)*std::exp(cse4) - cse25*cse82 - cse6*cse81 + 43900.0*cse74 + cse79*cse87)
)
: (
   0
));
  const amrex::Real cse89 = std::pow(state.T, -1.52)*state.xn[16]*state.xn[2];
  const amrex::Real cse90 = std::exp(-227*cse3);
  const amrex::Real cse91 = cse90*state.xn[1]*state.xn[11]*(2.6759999999999999e-12*std::pow(state.T, -0.99331000000000003) + 5.6870000000000003e-12*std::pow(state.T, -0.48299999999999998));
  const amrex::Real cse92 = std::pow(state.T, -2.2999999999999998);
  const amrex::Real cse93 = std::exp(-23*cse3);
  const amrex::Real cse94 = cse93*state.xn[3]*state.xn[8];
  const amrex::Real cse95 = cse92*cse94;
  const amrex::Real cse96 = std::pow(state.T, 0.042000000000000003);
  const amrex::Real cse97 = std::exp(-2.145922746781116e-5*state.T);
  const amrex::Real cse98 = cse97*state.xn[4];
  const amrex::Real cse99 = cse98*state.xn[3];
  const amrex::Real cse100 = 3.7768240343347639e-14*cse96*cse99;
  const amrex::Real cse101 = 7.3920000000000002e-11*cse99*std::pow(state.T, -0.95799999999999996);
  const amrex::Real cse102 = std::pow(state.T, -3.2999999999999998);
  const amrex::Real cse103 = cse90*(4.0000000000000001e-10*std::pow(state.T, 0.0066899999999999998) + 1.1000000000000001e-11*std::pow(state.T, 0.51700000000000002));
  const amrex::Real cse104 = cse103*state.xn[11];
  const amrex::Real cse105 = 227*cse104*cse66*state.xn[1];
  const amrex::Real cse106 = 1.0/state.xn[2];
  const amrex::Real cse107 = cse106*cse85;
  const amrex::Real cse108 = cse106*cse2;
  const amrex::Real cse109 = 0.00011320754716981131*cse108;
  const amrex::Real cse110 = cse109*state.xn[3];
  const amrex::Real cse111 = (((cse110 - 10000000000.0 > 0) ? (
   0
)
: ((cse110 - 10000000000.0 == 0) ? (
   1.0/2.0
)
: (
   1
))));
  const amrex::Real cse112 = cse111*state.xn[17];
  const amrex::Real cse113 = cse112*cse30;
  const amrex::Real cse114 = cse107*cse113;
  const amrex::Real cse115 = std::min(10000000000.0, cse110);
  const amrex::Real cse116 = cse115 + 1;
  const amrex::Real cse117 = std::pow(cse116, -2);
  const amrex::Real cse118 = 1.811320754716981e-13*cse117;
  const amrex::Real cse119 = 6.7743911789075847e-6*std::pow(state.T, -1.6400000000000001)*state.xn[10]*state.xn[2];
  const amrex::Real cse120 = std::exp(0.11*cse3);
  const amrex::Real cse121 = cse120*state.xn[15];
  const amrex::Real cse122 = cse121*state.xn[7];
  const amrex::Real cse123 = 8.7465066538127767e-11*cse122*std::pow(state.T, -2.339) + 2.6955143233113923e-10*cse122*std::pow(state.T, -1.339);
  const amrex::Real cse124 = cse119 + cse123;
  const amrex::Real cse125 = std::log(8.6173000000000003e-5*state.T);
  const amrex::Real cse126 = std::pow(cse125, 2);
  const amrex::Real cse127 = std::pow(cse125, 3);
  const amrex::Real cse128 = std::pow(cse125, 4);
  const amrex::Real cse129 = std::pow(cse125, 5);
  const amrex::Real cse130 = std::pow(cse125, 6);
  const amrex::Real cse131 = std::pow(cse125, 7);
  const amrex::Real cse132 = std::exp(-2.0391498499999999e-6*std::pow(cse125, 8) - 5.7393287500000003*cse126 + 1.56315498*cse127 - 0.28770560000000001*cse128 + 0.034825597700000002*cse129 - 0.00263197617*cse130 + 0.000111954395*cse131);
  const amrex::Real cse133 = 5.9078447946382801e-70*cse132*std::pow(state.T, 13.536555999999999);
  const amrex::Real cse134 = state.xn[0]*state.xn[2];
  const amrex::Real cse135 = cse134*((cse28) ? (
   7.9971871901929577e-69*cse132*std::pow(state.T, 12.536555999999999) + cse133*(-11.478657500000001*cse125*cse3 + 4.6894649399999997*cse126*cse3 - 1.1508224*cse127*cse3 + 0.1741279885*cse128*cse3 - 0.015791857020000001*cse129*cse3 + 0.00078368076500000001*cse130*cse3 - 1.6313198799999999e-5*cse131*cse3)
)
: (
   0
));
  const amrex::Real cse136 = std::pow(cse3, 1.5);
  const amrex::Real cse137 = 114.81344434488582*std::pow(cse3, 0.40699999999999997) + 1;
  const amrex::Real cse138 = std::pow(cse137, -2.242);
  const amrex::Real cse139 = cse3*state.xn[1]*state.xn[2];
  const amrex::Real cse140 = 7.3220359950196314e-6*cse136*cse138*cse139;
  const amrex::Real cse141 = 0.0005114031087522854*std::pow(cse137, -3.242)*cse139*std::pow(cse3, 1.907);
  const amrex::Real cse142 = 127.88871425900908*state.rn[2*0] + 175.16462312221924*state.rn[2*1] + 210.85430541663879*state.rn[2*2] + 232.36781247130679*state.rn[2*3];
  const amrex::Real cse143 = state.c_hat*cse142;
  const amrex::Real cse144 = cse108*cse143;
  const amrex::Real cse145 = std::pow(cse144, 1.3779999999999999);
  const amrex::Real cse146 = 5.670589618368584e-11*cse144;
  const amrex::Real cse147 = std::log(state.T);
  const amrex::Real cse148 = 1.102e-5*cse147 + 0.4723;
  const amrex::Real cse149 = std::pow(cse146, -cse148);
  const amrex::Real cse150 = cse149*std::pow(state.T, 0.015859999999999999);
  const amrex::Real cse151 = 508.69999999999999*cse150;
  const amrex::Real cse152 = cse151 + 1;
  const amrex::Real cse153 = cse145*cse152;
  const amrex::Real cse154 = -cse148;
  const amrex::Real cse155 = 0.5*cse3;
  const amrex::Real cse156 = cse3*std::log(cse146);
  const amrex::Real cse157 = 6.1318063394789369e-20*cse145;
  const amrex::Real cse158 = cse152*cse157 + 1;
  const amrex::Real cse159 = d2g*state.xn[1];
  const amrex::Real cse160 = 1.2250000000000001e-11*cse10*cse159/std::pow(cse158, 2);
  const amrex::Real cse161 = cse160*(-4.2248145679009874e-20*cse153*cse3 - cse157*(8.0679819999999989*cse149*std::pow(state.T, -0.98414000000000001) + cse151*(cse154*cse155 - 1.102e-5*cse156)));
  const amrex::Real cse162 = 1.0/cse116;
  const amrex::Real cse163 = 9.0566037735849048e-14*cse114;
  const amrex::Real cse164 = cse162*cse163;
  const amrex::Real cse165 = cse117*cse163;
  const amrex::Real cse166 = cse115*cse165;
  const amrex::Real cse167 = -cse164 + cse166;
  const amrex::Real cse168 = cse162*state.xn[16];
  const amrex::Real cse169 = 3.7810000000000002e-10*std::pow(state.T, -1.1899999999999999)*state.xn[11];
  const amrex::Real cse170 = cse106*cse111*cse117*state.xn[11]*state.xn[16];
  const amrex::Real cse171 = cse168*cse169 + 1.1264150943396225e-13*cse170*std::pow(state.T, -0.68999999999999995)*state.xn[3];
  const amrex::Real cse172 = cse135 + cse140 - cse141 - cse161 + cse167 + cse171;
  const amrex::Real cse173 = 0.66666666666670005*cse61/cse63;
  const amrex::Real cse174 = cse173*(-cse100 + cse101 + 9.8669999999999991e-12*cse102*cse93*state.xn[3]*state.xn[8] + cse105 - cse114*cse118 - cse124 - cse172 + 2*cse30*cse83 - cse69 - cse70 - cse73 - cse84 + 2*cse88*state.xn[0]*state.xn[3] - 1.5566492609718528e-6*cse89 + cse91 - 5.5769999999999994e-13*cse95);
  const amrex::Real cse175 = 7.5e-10*std::pow(state.T, -0.45800000000000002) + 4.9899999999999997e-11*std::pow(state.T, 0.40500000000000003);
  const amrex::Real cse176 = cse175*state.xn[17];
  const amrex::Real cse177 = 6.3999999999999996e-10*state.xn[4];
  const amrex::Real cse178 = crate*(1.5*cse11*state.xn[0] + 2.2999999999999998*cse11*state.xn[3]);
  const amrex::Real cse179 = ((cse28) ? (
   cse133
)
: (
   0.0
));
  const amrex::Real cse180 = cse179*state.xn[2];
  const amrex::Real cse181 = 2.2999999999999998*cse36 + 1.5*cse38;
  const amrex::Real cse182 = 1.5*cse11 - cse181;
  const amrex::Real cse183 = crate*state.xn[0];
  const amrex::Real cse184 = cse182*cse183;
  const amrex::Real cse185 = 1.0/cse158;
  const amrex::Real cse186 = 1.2250000000000001e-11*cse185*d2g;
  const amrex::Real cse187 = cse186*state.xn[1];
  const amrex::Real cse188 = cse176 + cse177 + cse178 + cse180 + cse184 - cse187;
  const amrex::Real cse189 = cse62 - 2.5092930924433742e-24*cse64;
  const amrex::Real cse190 = 4.8813573300130879e-6*cse136*cse138;
  const amrex::Real cse191 = -cse181;
  const amrex::Real cse192 = -cse183*cse191 + cse187;
  const amrex::Real cse193 = cse10*cse186 + cse190*state.xn[2] + cse192;
  const amrex::Real cse194 = cse104 + cse193;
  const amrex::Real cse195 = cse10*((cse20) ? (
   0
)
: (
   -cse40
));
  const amrex::Real cse196 = cse50*(-cse195 - cse21);
  const amrex::Real cse197 = cse196*cse22;
  const amrex::Real cse198 = cse30*((cse28) ? (
   cse54*(cse195*cse80 + cse196*cse49 + cse197*cse31 + cse33)
)
: (
   0
));
  const amrex::Real cse199 = -cse57;
  const amrex::Real cse200 = ((cse28) ? (
   cse59*(cse195*cse87 + cse196*cse6 + cse197*cse25 + cse58)
)
: (
   0
));
  const amrex::Real cse201 = cse9*state.xn[0];
  const amrex::Real cse202 = 2*cse198 + cse199 + cse200*cse201;
  const amrex::Real cse203 = std::pow(state.xn[2], 2);
  const amrex::Real cse204 = 1.0/cse203;
  const amrex::Real cse205 = 2.2528301886792451e-13*std::pow(state.T, 0.31);
  const amrex::Real cse206 = cse111*cse117*cse204*cse205*state.xn[11]*state.xn[16]*state.xn[3];
  const amrex::Real cse207 = std::pow(state.T, -0.52000000000000002);
  const amrex::Real cse208 = 2.9935562710997168e-6*cse207;
  const amrex::Real cse209 = cse62 - 1.3664075564999315e-27*cse64;
  const amrex::Real cse210 = cse2*cse204;
  const amrex::Real cse211 = cse113*cse210;
  const amrex::Real cse212 = 3.6226415094339619e-13*cse117;
  const amrex::Real cse213 = 1.811320754716981e-13*cse162;
  const amrex::Real cse214 = cse211*cse213;
  const amrex::Real cse215 = cse118*cse211;
  const amrex::Real cse216 = cse115*cse215;
  const amrex::Real cse217 = -cse214 + cse216;
  const amrex::Real cse218 = 1.05849862170431e-5*std::pow(state.T, -0.64000000000000001);
  const amrex::Real cse219 = cse218*state.xn[10];
  const amrex::Real cse220 = cse179*state.xn[0];
  const amrex::Real cse221 = cse145*cse150*cse154;
  const amrex::Real cse222 = cse160*(8.4496291358019747e-20*cse106*cse153 + 3.1192498848929353e-17*cse106*cse221) + cse190*state.xn[1] - cse220;
  const amrex::Real cse223 = cse219 + cse222;
  const amrex::Real cse224 = 1.2599999999999999e-13*cse67;
  const amrex::Real cse225 = cse224*state.xn[6];
  const amrex::Real cse226 = 1.7599999999999999e-9*cse96;
  const amrex::Real cse227 = cse226*cse98;
  const amrex::Real cse228 = cse159*cse185;
  const amrex::Real cse229 = 2.4500000000000003e-11*cse228;
  const amrex::Real cse230 = cse29*state.xn[0];
  const amrex::Real cse231 = 2*cse21;
  const amrex::Real cse232 = cse231*cse80;
  const amrex::Real cse233 = cse35*cse9;
  const amrex::Real cse234 = 2*state.xn[0];
  const amrex::Real cse235 = cse234*cse35;
  const amrex::Real cse236 = cse15*cse233 + cse17*cse235;
  const amrex::Real cse237 = cse10*((cse20) ? (
   0
)
: (
   cse11*cse15 - cse236
));
  const amrex::Real cse238 = cse50*(-cse231 - cse237);
  const amrex::Real cse239 = cse22*cse238;
  const amrex::Real cse240 = cse30*((cse28) ? (
   cse54*(cse232 + cse237*cse80 + cse238*cse49 + cse239*cse31)
)
: (
   0
));
  const amrex::Real cse241 = ((cse28) ? (
   cse53
)
: (
   0.0
));
  const amrex::Real cse242 = 4*state.xn[3];
  const amrex::Real cse243 = d2g*state.xn[0];
  const amrex::Real cse244 = 1.2000000000000001e-14*cse243;
  const amrex::Real cse245 = cse231*cse87;
  const amrex::Real cse246 = ((cse28) ? (
   cse59*(cse237*cse87 + cse238*cse6 + cse239*cse25 + cse245)
)
: (
   0
));
  const amrex::Real cse247 = state.c_hat*state.rn[2*1];
  const amrex::Real cse248 = std::max(1.0e-35, 2.0000000000000002e-15*ncol_H2);
  const amrex::Real cse249 = 0.035000000000000003*std::pow(cse248 + 1, -0.5)*std::exp(-0.00084999999999999995*std::sqrt(cse248 + 1.0)) + 0.96499999999999997*std::pow(cse248/std::max(1.0e-50, 1.4142135623730953e-5*vdisp) + 1.0, -2.0);
  const amrex::Real cse250 = 7.0043322447326937e-27*cse249;
  const amrex::Real cse251 = state.c_hat*state.rn[2*2];
  const amrex::Real cse252 = 2.767685094627529e-18*cse249;
  const amrex::Real cse253 = state.c_hat*state.rn[2*3];
  const amrex::Real cse254 = 1.291395623375354e-17*cse249;
  const amrex::Real cse255 = cse93*std::pow(state.T, -1.3);
  const amrex::Real cse256 = 4.2899999999999994e-13*cse255;
  const amrex::Real cse257 = cse62 - 5.0213189999997486e-24*cse64;
  const amrex::Real cse258 = cse108*cse112*state.xn[3];
  const amrex::Real cse259 = 1.6000000000000001e-9*cse162;
  const amrex::Real cse260 = cse259*state.xn[17];
  const amrex::Real cse261 = cse118*cse258;
  const amrex::Real cse262 = cse115*cse260 - cse115*cse261 + cse213*cse258;
  const amrex::Real cse263 = 4.5999999999999996*cse36 + 3.0*cse38;
  const amrex::Real cse264 = 2.2999999999999998*cse11 - cse263;
  const amrex::Real cse265 = cse183*cse264;
  const amrex::Real cse266 = cse170*cse205;
  const amrex::Real cse267 = -cse265 - cse266;
  const amrex::Real cse268 = cse10*((cse20) ? (
   0
)
: (
   -cse236
));
  const amrex::Real cse269 = cse50*(-cse231 - cse268);
  const amrex::Real cse270 = cse22*cse269;
  const amrex::Real cse271 = ((cse28) ? (
   cse54*(cse232 + cse268*cse80 + cse269*cse49 + cse270*cse31)
)
: (
   0
));
  const amrex::Real cse272 = ((cse28) ? (
   cse59*(cse245 + cse25*cse270 + cse268*cse87 + cse269*cse6)
)
: (
   0
));
  const amrex::Real cse273 = cse226*cse97*state.xn[3];
  const amrex::Real cse274 = cse62 - 5.0199525924432485e-24*cse64;
  const amrex::Real cse275 = -cse229;
  const amrex::Real cse276 = 6.3999999999999996e-10*state.xn[0];
  const amrex::Real cse277 = -cse263;
  const amrex::Real cse278 = cse183*cse277;
  const amrex::Real cse279 = cse275 + cse276 + cse278;
  const amrex::Real cse280 = cse62 - 9.9697094999995006e-24*cse64;
  const amrex::Real cse281 = cse62 - 9.9683430924430005e-24*cse64;
  const amrex::Real cse282 = cse224*state.xn[3];
  const amrex::Real cse283 = 26.218500000000002*std::sqrt(3)*std::sqrt(cse3) + 0.62;
  const amrex::Real cse284 = 1.3500000000000001e-9*cse283;
  const amrex::Real cse285 = cse284*state.xn[15];
  const amrex::Real cse286 = cse282 + cse285;
  const amrex::Real cse287 = 7.9513696852843427e-10*std::pow(state.T, -0.33900000000000002);
  const amrex::Real cse288 = cse121*cse287;
  const amrex::Real cse289 = cse62 - 2.9917094999998509e-23*cse64;
  const amrex::Real cse290 = cse62 - 2.991572859244201e-23*cse64;
  const amrex::Real cse291 = 6.5199999999999996*crate;
  const amrex::Real cse292 = cse62 - 6.976853999999652e-23*cse64;
  const amrex::Real cse293 = cse62 - 7.2277833092439893e-23*cse64;
  const amrex::Real cse294 = cse218*state.xn[2];
  const amrex::Real cse295 = cse192 + cse294;
  const amrex::Real cse296 = 1.99e-9*std::pow(state.T, -0.19);
  const amrex::Real cse297 = cse168*cse296;
  const amrex::Real cse298 = 3.5000000000000002e-11*state.xn[15];
  const amrex::Real cse299 = cse62 - 3.9851444999998012e-23*cse64;
  const amrex::Real cse300 = cse103*state.xn[1];
  const amrex::Real cse301 = 7.0000000000000004e-11*state.xn[14];
  const amrex::Real cse302 = cse300 + cse301;
  const amrex::Real cse303 = cse62 - 6.9955604999996505e-23*cse64;
  const amrex::Real cse304 = cse62 - 6.9954238592440006e-23*cse64;
  const amrex::Real cse305 = state.c_hat*state.rn[2*0];
  const amrex::Real cse306 = 5.4356586967624969e-18*cse305;
  const amrex::Real cse307 = 1.9473397080211686e-18*cse247;
  const amrex::Real cse308 = 1.0937796810239276e-18*cse251;
  const amrex::Real cse309 = 1.2582308138615626e-18*cse253;
  const amrex::Real cse310 = 7.0000000000000004e-11*state.xn[11];
  const amrex::Real cse311 = cse0*state.xn[0];
  const amrex::Real cse312 = -cse311;
  const amrex::Real cse313 = cse62 - 3.242775449999838e-23*cse64;
  const amrex::Real cse314 = cse192 + cse202;
  const amrex::Real cse315 = cse62 - 4.2362104499997889e-23*cse64;
  const amrex::Real cse316 = cse284*state.xn[6];
  const amrex::Real cse317 = cse120*cse287*state.xn[7];
  const amrex::Real cse318 = 1.1464502398585298e-18*cse305;
  const amrex::Real cse319 = 2.7737849364247527e-18*cse247;
  const amrex::Real cse320 = 6.87721070339377e-18*cse251;
  const amrex::Real cse321 = 8.8938650996318353e-19*cse253;
  const amrex::Real cse322 = cse318 + cse319 + cse320 + cse321 + 3.5000000000000002e-11*state.xn[11];
  const amrex::Real cse323 = cse316 + cse317 + cse322;
  const amrex::Real cse324 = 3*cse21;
  const amrex::Real cse325 = cse10*((cse20) ? (
   0
)
: (
   -3*cse37 - 3*cse39
));
  const amrex::Real cse326 = cse50*(-cse324 - cse325);
  const amrex::Real cse327 = cse22*cse326;
  const amrex::Real cse328 = cse30*((cse28) ? (
   cse54*(cse31*cse327 + cse324*cse80 + cse325*cse80 + cse326*cse49)
)
: (
   0
));
  const amrex::Real cse329 = ((cse28) ? (
   cse59*(cse25*cse327 + cse324*cse87 + cse325*cse87 + cse326*cse6)
)
: (
   0
));
  const amrex::Real cse330 = cse62 - 7.5306120924431232e-24*cse64;
  const amrex::Real cse331 = cse296*state.xn[11];
  const amrex::Real cse332 = cse162*cse331;
  const amrex::Real cse333 = -6.8999999999999995*cse36 - 4.5*cse38;
  const amrex::Real cse334 = -cse183*cse333 + 3.6750000000000002e-11*cse228;
  const amrex::Real cse335 = cse332 + cse334;
  const amrex::Real cse336 = cse175*state.xn[0];
  const amrex::Real cse337 = cse259*state.xn[3];
  const amrex::Real cse338 = cse115*cse337;
  const amrex::Real cse339 = cse62 - 3.9850078592441519e-23*cse64;
  const amrex::Real cse340 = 1.0/cse142;
  const amrex::Real cse341 = cse153*cse340;
  const amrex::Real cse342 = cse221*cse340;
  const amrex::Real cse343 = cse160*(-1.0806122061431765e-17*cse341 - 3.9891685723151951e-15*cse342);
  const amrex::Real cse344 = state.c_hat*state.xn[14];
  const amrex::Real cse345 = 5.4356586967624969e-18*cse344;
  const amrex::Real cse346 = state.c_hat*state.xn[15];
  const amrex::Real cse347 = 1.1464502398585298e-18*cse346;
  const amrex::Real cse348 = cse345 + cse347;
  const amrex::Real cse349 = state.c_hat*state.xn[3];
  const amrex::Real cse350 = cse160*(-1.4800761030952758e-17*cse341 - 5.4638223051129673e-15*cse342);
  const amrex::Real cse351 = 1.9473397080211686e-18*cse344;
  const amrex::Real cse352 = 2.7737849364247527e-18*cse346;
  const amrex::Real cse353 = cse351 + cse352;
  const amrex::Real cse354 = 1.0937796810239276e-18*cse344;
  const amrex::Real cse355 = 6.87721070339377e-18*cse346;
  const amrex::Real cse356 = cse160*(-1.7816406824577193e-17*cse341 - 6.5770726790003031e-15*cse342);
  const amrex::Real cse357 = 1.2582308138615626e-18*cse344;
  const amrex::Real cse358 = 8.8938650996318353e-19*cse346;
  const amrex::Real cse359 = cse160*(-1.9634218384801235e-17*cse341 - 7.2481327230394677e-15*cse342);
  const amrex::Real cse360 = cse105 - cse84 + cse91;
  const amrex::Real cse361 = cse173*(cse135 + cse140 - cse141 - cse161 - cse360 - cse69);
  const amrex::Real cse362 = crate*cse9;
  const amrex::Real cse363 = cse182*cse362;
  const amrex::Real cse364 = -cse363;
  const amrex::Real cse365 = std::pow(cse3, 0.00231);
  const amrex::Real cse366 = std::exp(-8110.0*cse3);
  const amrex::Real cse367 = std::exp(-26.600000000000001*cse3);
  const amrex::Real cse368 = std::exp(-7.6200000000000001*cse3);
  const amrex::Real cse369 = std::exp(-1.3799999999999999*cse3);
  const amrex::Real cse370 = 0.00015100000000000001*cse366 + 1.31e-7*cse367 + 3.4e-8*cse368 + 6.9699999999999997e-9*cse369;
  const amrex::Real cse371 = std::pow(state.T, -1.5);
  const amrex::Real cse372 = state.xn[16]*state.xn[7]*(-2.4342628390636543e-12*cse3*cse365 - 1.5*cse370*std::pow(state.T, -2.5) + cse371*(1.2246100000000002*cse366*cse66 + 3.4846000000000001e-6*cse367*cse66 + 2.5908000000000001e-7*cse368*cse66 + 9.6185999999999982e-9*cse369*cse66));
  const amrex::Real cse373 = -cse372;
  const amrex::Real cse374 = 6.7636511339243933e-7*cse89;
  const amrex::Real cse375 = cse11*state.xn[2];
  const amrex::Real cse376 = cse375 < 1.0e-50;
  const amrex::Real cse377 = ((cse376) ? (
   10000000000.0
)
: (
   cse110
));
  const amrex::Real cse378 = cse377 + 1;
  const amrex::Real cse379 = 1.0/cse378;
  const amrex::Real cse380 = cse377*cse379;
  const amrex::Real cse381 = cse380*state.xn[16];
  const amrex::Real cse382 = cse169*cse381;
  const amrex::Real cse383 = ((cse376) ? (
   0
)
: (
   5.6603773584905654e-5*cse107*state.xn[3]
));
  const amrex::Real cse384 = cse331*state.xn[16];
  const amrex::Real cse385 = cse379*cse384;
  const amrex::Real cse386 = cse383*cse385;
  const amrex::Real cse387 = -cse386;
  const amrex::Real cse388 = cse377*cse384/std::pow(cse378, 2);
  const amrex::Real cse389 = cse383*cse388;
  const amrex::Real cse390 = std::exp(-15230.0*cse3);
  const amrex::Real cse391 = std::exp(-73.799999999999997*cse3);
  const amrex::Real cse392 = std::exp(-12.17*cse3);
  const amrex::Real cse393 = 1.6339999999999999e-6*cse390 + 9.7930000000000004e-9*cse391 + 6.3460000000000002e-9*cse392;
  const amrex::Real cse394 = 0.00071740367932898903*cse2 + 1;
  const amrex::Real cse395 = std::exp(-49550.0*cse3);
  const amrex::Real cse396 = 0.15970000000000001*cse395;
  const amrex::Real cse397 = cse396 + 1.7848999999999999;
  const amrex::Real cse398 = 12.244387999457523*cse2 + 1;
  const amrex::Real cse399 = 0.21509999999999996 - cse396;
  const amrex::Real cse400 = std::pow(cse394, -cse397)*std::pow(cse398, -cse399);
  const amrex::Real cse401 = 7913.1350000000002*cse395*cse66;
  const amrex::Real cse402 = 2.4460185352936309e-10*cse400*cse85;
  const amrex::Real cse403 = state.xn[2]*state.xn[8]*(-3.0/2.0*cse393/std::pow(state.T, 5.0/2.0) - 1.2230092676468155e-10*cse400*cse86 + cse402*(-cse401*std::log(cse394) - 0.00035870183966449452*cse397*cse85/cse394) + cse402*(cse401*std::log(cse398) - 6.1221939997287613*cse399*cse85/cse398) + cse86*(0.024885819999999999*cse390*cse66 + 7.2272340000000005e-7*cse391*cse66 + 7.7230820000000007e-8*cse392*cse66));
  const amrex::Real cse404 = 9.0489001802816564e-11*std::pow(state.T, -1.6200000000000001)*state.xn[13]*state.xn[2];
  const amrex::Real cse405 = cse147*cse8;
  const amrex::Real cse406 = 0.044330000000000001*cse405 - 0.28520000000000001;
  const amrex::Real cse407 = cse405*cse406 - 1.6759999999999999;
  const amrex::Real cse408 = cse405*cse407 + 11.19;
  const amrex::Real cse409 = cse408*state.xn[6];
  const amrex::Real cse410 = 4.9999999999999997e-12*cse371*cse409*state.xn[2];
  const amrex::Real cse411 = cse102*cse94;
  const amrex::Real cse412 = 7.5899999999999998e-12*cse411;
  const amrex::Real cse413 = 9.9999999999999994e-12*std::pow(state.T, -0.5);
  const amrex::Real cse414 = cse413*state.xn[2];
  const amrex::Real cse415 = cse414*state.xn[6]*(cse405*(0.044330000000000001*cse147*cse75 + cse406*cse77) + cse407*cse77);
  const amrex::Real cse416 = std::pow(cse144, 1.8740000000000001);
  const amrex::Real cse417 = 7.538e-5*cse147 + 0.89639999999999997;
  const amrex::Real cse418 = std::pow(cse146, -cse417);
  const amrex::Real cse419 = cse418*std::pow(state.T, 1.635e-6);
  const amrex::Real cse420 = 43750.0*cse419;
  const amrex::Real cse421 = cse420 + 1;
  const amrex::Real cse422 = cse416*cse421;
  const amrex::Real cse423 = -cse417;
  const amrex::Real cse424 = 3.5685736926834518e-27*cse416;
  const amrex::Real cse425 = cse421*cse424 + 1;
  const amrex::Real cse426 = cse10*d2g;
  const amrex::Real cse427 = 2.1659999999999999e-12*cse426;
  const amrex::Real cse428 = cse427*state.xn[13]/std::pow(cse425, 2);
  const amrex::Real cse429 = cse428*(-3.3437535500443945e-27*cse3*cse422 - cse424*(0.071531250000000005*cse418*std::pow(state.T, -0.99999836499999994) + cse420*(cse155*cse423 - 7.538e-5*cse156)));
  const amrex::Real cse430 = std::pow(cse144, 1.512);
  const amrex::Real cse431 = 5.4939999999999998e-7*cse147 + 0.49559999999999998;
  const amrex::Real cse432 = std::pow(cse146, -cse431);
  const amrex::Real cse433 = cse432*std::pow(state.T, 3.9029999999999999e-7);
  const amrex::Real cse434 = 5115.0*cse433;
  const amrex::Real cse435 = cse434 + 1;
  const amrex::Real cse436 = cse430*cse435;
  const amrex::Real cse437 = -cse431;
  const amrex::Real cse438 = 1.0246966183388124e-22*cse430;
  const amrex::Real cse439 = cse435*cse438 + 1;
  const amrex::Real cse440 = 5.5720000000000001e-12*cse426;
  const amrex::Real cse441 = cse440*state.xn[6]/std::pow(cse439, 2);
  const amrex::Real cse442 = cse441*(-7.7467064346414215e-23*cse3*cse436 - cse438*(0.0019963845000000001*cse432*std::pow(state.T, -0.99999960970000001) + cse434*(cse155*cse437 - 5.4939999999999998e-7*cse156)));
  const amrex::Real cse443 = std::pow(cse144, 1.1279999999999999);
  const amrex::Real cse444 = 0.00013329999999999999*cse147 + 0.81200000000000006;
  const amrex::Real cse445 = std::pow(cse146, -cse444);
  const amrex::Real cse446 = cse445*std::pow(state.T, 0.04845);
  const amrex::Real cse447 = 433.10000000000002*cse446;
  const amrex::Real cse448 = cse447 + 1;
  const amrex::Real cse449 = cse443*cse448;
  const amrex::Real cse450 = -cse444;
  const amrex::Real cse451 = 1.6851501542381549e-14*cse443;
  const amrex::Real cse452 = cse448*cse451 + 1;
  const amrex::Real cse453 = 4.5579999999999993e-11*cse426;
  const amrex::Real cse454 = cse453*state.xn[8]/std::pow(cse452, 2);
  const amrex::Real cse455 = cse454*(-9.5042468699031921e-15*cse3*cse449 - cse451*(20.983695000000001*cse445*std::pow(state.T, -0.95155000000000001) + cse447*(cse155*cse450 - 0.00013329999999999999*cse156)));
  const amrex::Real cse456 = cse173*(cse119 + cse165 + cse172 + cse373 + cse374 + cse382 + cse387 + cse389 - cse403 + cse404 + cse410 - cse412 - cse415 - cse429 - cse442 - cse455 + 4.2899999999999994e-13*cse95);
  const amrex::Real cse457 = 1.0/cse439;
  const amrex::Real cse458 = cse457*d2g*state.xn[6];
  const amrex::Real cse459 = 5.5720000000000001e-12*cse458;
  const amrex::Real cse460 = 1.0/cse452;
  const amrex::Real cse461 = cse460*d2g*state.xn[8];
  const amrex::Real cse462 = 4.5579999999999993e-11*cse461;
  const amrex::Real cse463 = crate*cse36;
  const amrex::Real cse464 = cse463*state.xn[7];
  const amrex::Real cse465 = 1040.0*cse464;
  const amrex::Real cse466 = cse462 + cse465;
  const amrex::Real cse467 = 1.0/cse425;
  const amrex::Real cse468 = cse467*d2g*state.xn[13];
  const amrex::Real cse469 = cse463*state.xn[12];
  const amrex::Real cse470 = 2.1659999999999999e-12*cse468 + 16800.0*cse469;
  const amrex::Real cse471 = cse459 + cse466 + cse470;
  const amrex::Real cse472 = cse191*cse362;
  const amrex::Real cse473 = cse471 - cse472;
  const amrex::Real cse474 = cse430*cse433*cse437;
  const amrex::Real cse475 = cse409*cse413 + cse441*(1.5493412869282843e-22*cse106*cse436 + 5.2413232028030254e-19*cse106*cse474);
  const amrex::Real cse476 = cse393*cse86 + cse402;
  const amrex::Real cse477 = cse443*cse446*cse450;
  const amrex::Real cse478 = cse454*(1.9008493739806384e-14*cse106*cse449 + 7.2983853180054489e-12*cse106*cse477) + cse476*state.xn[8];
  const amrex::Real cse479 = 1.4595000290776864e-10*std::pow(state.T, -0.62);
  const amrex::Real cse480 = cse416*cse419*cse423;
  const amrex::Real cse481 = cse428*(6.687507100088789e-27*cse106*cse422 + 1.5612509905490101e-22*cse106*cse480) + cse479*state.xn[13];
  const amrex::Real cse482 = cse215 + cse217;
  const amrex::Real cse483 = 1.3007021411393064e-6*cse207;
  const amrex::Real cse484 = ((cse376) ? (
   0
)
: (
   -0.00011320754716981131*cse210*state.xn[3]
));
  const amrex::Real cse485 = cse385*cse484;
  const amrex::Real cse486 = cse388*cse484;
  const amrex::Real cse487 = cse485 - cse486;
  const amrex::Real cse488 = cse206 + cse483*state.xn[16] + cse487;
  const amrex::Real cse489 = 2*cse178;
  const amrex::Real cse490 = cse264*cse362;
  const amrex::Real cse491 = crate*cse11;
  const amrex::Real cse492 = 16800.0*cse491;
  const amrex::Real cse493 = 4.3319999999999997e-12*cse468 + 33600.0*cse469;
  const amrex::Real cse494 = -cse492*state.xn[12] + cse493;
  const amrex::Real cse495 = 3.2999999999999996e-13*cse255;
  const amrex::Real cse496 = cse495*state.xn[8];
  const amrex::Real cse497 = 1040.0*cse491;
  const amrex::Real cse498 = 9.1159999999999986e-11*cse461 + 2080.0*cse464;
  const amrex::Real cse499 = -cse497*state.xn[7] + cse498;
  const amrex::Real cse500 = cse496 + cse499;
  const amrex::Real cse501 = ((cse376) ? (
   0
)
: (
   cse109
));
  const amrex::Real cse502 = cse385*cse501;
  const amrex::Real cse503 = cse388*cse501;
  const amrex::Real cse504 = cse502 - cse503;
  const amrex::Real cse505 = cse262 + cse504;
  const amrex::Real cse506 = cse260 - cse261;
  const amrex::Real cse507 = 1.1144e-11*cse458;
  const amrex::Real cse508 = cse229 + cse507;
  const amrex::Real cse509 = cse277*cse362;
  const amrex::Real cse510 = -cse509;
  const amrex::Real cse511 = 1.1000000000000001*crate;
  const amrex::Real cse512 = cse408*cse414 + cse440*cse457;
  const amrex::Real cse513 = 1.0537934368240927e-9*cse365 + cse370*cse371;
  const amrex::Real cse514 = cse513*state.xn[16];
  const amrex::Real cse515 = interp_8_shielding_H2(ncol_H2)*interp_8_shielding_self(ncol_C);
  const amrex::Real cse516 = 1.7033009950880063e-17*cse515;
  const amrex::Real cse517 = cse253*cse516;
  const amrex::Real cse518 = 1.3061417322445169e-17*cse515;
  const amrex::Real cse519 = cse251*cse518;
  const amrex::Real cse520 = 3.8500000000000001*crate + cse497*state.xn[3] + cse517 + cse519;
  const amrex::Real cse521 = cse495*state.xn[3];
  const amrex::Real cse522 = cse453*cse460 + cse476*state.xn[2];
  const amrex::Real cse523 = cse521 + cse522;
  const amrex::Real cse524 = cse296*cse381;
  const amrex::Real cse525 = cse297 + cse524;
  const amrex::Real cse526 = 1.7217538808319277e-17*cse253;
  const amrex::Real cse527 = 4.4805860217826794e-17*cse247;
  const amrex::Real cse528 = 2.062878503133515e-17*cse251;
  const amrex::Real cse529 = 6.5847540874715911e-18*cse305;
  const amrex::Real cse530 = cse492*state.xn[3] + cse526 + cse527 + cse528 + cse529;
  const amrex::Real cse531 = cse427*cse467 + cse479*state.xn[2];
  const amrex::Real cse532 = cse192 + cse473;
  const amrex::Real cse533 = 1.6715999999999999e-11*cse458;
  const amrex::Real cse534 = cse513*state.xn[7];
  const amrex::Real cse535 = cse331*cse380;
  const amrex::Real cse536 = cse483*state.xn[2];
  const amrex::Real cse537 = cse333*cse362;
  const amrex::Real cse538 = -cse537;
  const amrex::Real cse539 = 1.3673999999999998e-10*cse461;
  const amrex::Real cse540 = 3120.0*cse464;
  const amrex::Real cse541 = cse539 + cse540;
  const amrex::Real cse542 = 6.4979999999999992e-12*cse468 + 50400.0*cse469;
  const amrex::Real cse543 = cse337 + cse338;
  const amrex::Real cse544 = state.c_hat*state.xn[12];
  const amrex::Real cse545 = 6.5847540874715911e-18*cse544;
  const amrex::Real cse546 = cse340*cse422;
  const amrex::Real cse547 = cse340*cse480;
  const amrex::Real cse548 = cse428*(-8.552566846283495e-25*cse546 - 1.9966638181691725e-20*cse547);
  const amrex::Real cse549 = cse340*cse436;
  const amrex::Real cse550 = cse340*cse474;
  const amrex::Real cse551 = cse441*(-1.9814326513365678e-20*cse549 - 6.7030608542239037e-17*cse550);
  const amrex::Real cse552 = cse340*cse449;
  const amrex::Real cse553 = cse340*cse477;
  const amrex::Real cse554 = cse454*(-2.4309718243842617e-12*cse552 - 9.3338111448654608e-10*cse553);
  const amrex::Real cse555 = 4.4805860217826794e-17*cse544;
  const amrex::Real cse556 = cse428*(-1.1714146608142181e-24*cse546 - 2.734759413587088e-20*cse547);
  const amrex::Real cse557 = cse441*(-2.7138978261248704e-20*cse549 - 9.1809440348073507e-17*cse550);
  const amrex::Real cse558 = cse454*(-3.3296156420542493e-12*cse552 - 1.2784189136291626e-9*cse553);
  const amrex::Real cse559 = 2.062878503133515e-17*cse544;
  const amrex::Real cse560 = cse428*(-1.410089664558062e-24*cse546 - 3.2919649319325084e-20*cse547);
  const amrex::Real cse561 = cse441*(-3.2668528090858467e-20*cse549 - 1.1051555633911445e-16*cse550);
  const amrex::Real cse562 = state.c_hat*state.xn[7];
  const amrex::Real cse563 = cse518*cse562;
  const amrex::Real cse564 = cse454*(-4.008022744523402e-12*cse552 - 1.5388959668910334e-9*cse553);
  const amrex::Real cse565 = -cse563 + cse564;
  const amrex::Real cse566 = 1.7217538808319277e-17*cse544;
  const amrex::Real cse567 = cse428*(-1.5539613957339645e-24*cse546 - 3.6278447739253435e-20*cse547);
  const amrex::Real cse568 = cse441*(-3.6001704561500472e-20*cse549 - 1.2179148070904424e-16*cse550);
  const amrex::Real cse569 = cse516*cse562;
  const amrex::Real cse570 = cse454*(-4.4169621086933389e-12*cse552 - 1.6959098309176289e-9*cse553);
  const amrex::Real cse571 = -cse569 + cse570;
  const amrex::Real cse572 = -3.0000000000000002e-15*d2g*state.xn[0];
  const amrex::Real cse573 = cse29*state.xn[3];
  const amrex::Real cse574 = state.xn[0]*state.xn[3];
  const amrex::Real cse575 = cse574*cse60;
  const amrex::Real cse576 = cse30*cse55;
  const amrex::Real cse577 = -cse70;
  const amrex::Real cse578 = cse30*cse83;
  const amrex::Real cse579 = cse574*cse88;
  const amrex::Real cse580 = cse412 - 4.2899999999999994e-13*cse92*cse93*state.xn[3]*state.xn[8];
  const amrex::Real cse581 = cse164 - cse166;
  const amrex::Real cse582 = -cse100 + cse101;
  const amrex::Real cse583 = cse171 + cse373 + cse382 + cse387 + cse389 + cse582;
  const amrex::Real cse584 = cse173*(cse165 - cse577 - cse578 - cse579 - cse580 - cse581 - cse583 - cse68 - 2.3622303960273254e-7*cse89);
  const amrex::Real cse585 = cse200*cse574;
  const amrex::Real cse586 = cse198 + cse472 + cse572 + cse585;
  const amrex::Real cse587 = 4.5427507615910105e-7*cse207;
  const amrex::Real cse588 = -cse215 + cse487;
  const amrex::Real cse589 = cse214 - cse216;
  const amrex::Real cse590 = cse246*cse574;
  const amrex::Real cse591 = 3.5021661223663468e-27*cse249;
  const amrex::Real cse592 = 1.3838425473137645e-18*cse249;
  const amrex::Real cse593 = 6.45697811687677e-18*cse249;
  const amrex::Real cse594 = cse247*cse591 + cse251*cse592 + cse253*cse593;
  const amrex::Real cse595 = 7.2000000000000002e-15*state.xn[6];
  const amrex::Real cse596 = cse225 + cse595;
  const amrex::Real cse597 = cse489 + cse490;
  const amrex::Real cse598 = cse227 + cse266 - cse502 + cse503;
  const amrex::Real cse599 = cse262 + cse506;
  const amrex::Real cse600 = cse272*cse574;
  const amrex::Real cse601 = cse271*cse30;
  const amrex::Real cse602 = 7.2000000000000002e-15*state.xn[3];
  const amrex::Real cse603 = 1.6999999999999999e-9*state.xn[16];
  const amrex::Real cse604 = 9.0000000000000011e-15*cse243;
  const amrex::Real cse605 = cse329*cse574;
  const amrex::Real cse606 = 1.6999999999999999e-9*state.xn[9];
  const amrex::Real cse607 = cse332 + cse534 + cse535 + cse606;
  const amrex::Real cse608 = cse349*cse591;
  const amrex::Real cse609 = cse349*cse592;
  const amrex::Real cse610 = cse349*cse593;
  const amrex::Real cse611 = -cse173*cse582;
  const amrex::Real cse612 = -cse73;
  const amrex::Real cse613 = -cse410 + cse415 + cse442 + cse612 + cse68;
  const amrex::Real cse614 = cse173*cse613;
  const amrex::Real cse615 = cse507 + cse596;
  const amrex::Real cse616 = 1.6000000000000001e-9*state.xn[9];
  const amrex::Real cse617 = cse286 + cse512 + cse602 + cse616;
  const amrex::Real cse618 = 1.6000000000000001e-9*state.xn[6];
  const amrex::Real cse619 = cse316 + cse459;
  const amrex::Real cse620 = -cse173*cse613;
  const amrex::Real cse621 = cse403 + cse455;
  const amrex::Real cse622 = cse173*(cse123 + cse373 + 2.2769999999999997e-12*cse411 + cse621 + cse70 - 1.287e-13*cse95);
  const amrex::Real cse623 = cse463*state.xn[9];
  const amrex::Real cse624 = 184.0*cse623;
  const amrex::Real cse625 = -cse624;
  const amrex::Real cse626 = cse466 + cse625;
  const amrex::Real cse627 = 9.8999999999999995e-14*cse255;
  const amrex::Real cse628 = 184.0*cse491;
  const amrex::Real cse629 = cse628*state.xn[9];
  const amrex::Real cse630 = 368.0*cse623;
  const amrex::Real cse631 = -cse630;
  const amrex::Real cse632 = cse629 + cse631;
  const amrex::Real cse633 = interp_10_shielding_H2(ncol_H2)*interp_10_shielding_self(ncol_CO);
  const amrex::Real cse634 = 3.0014835315191357e-17*cse633;
  const amrex::Real cse635 = cse253*cse634;
  const amrex::Real cse636 = 5.91270072314384e-18*cse633;
  const amrex::Real cse637 = cse251*cse636;
  const amrex::Real cse638 = cse628*state.xn[3] + cse635 + cse637;
  const amrex::Real cse639 = cse306 + cse307 + cse308 + cse309 + cse311;
  const amrex::Real cse640 = cse317 + cse624;
  const amrex::Real cse641 = 552.0*cse623;
  const amrex::Real cse642 = state.c_hat*state.xn[9];
  const amrex::Real cse643 = cse636*cse642;
  const amrex::Real cse644 = cse634*cse642;
  const amrex::Real cse645 = 1.1994963750000002e-8*cse72*state.xn[8];
  const amrex::Real cse646 = -cse645;
  const amrex::Real cse647 = cse173*(-cse580 - cse621 - cse646);
  const amrex::Real cse648 = 9.1500000000000005e-10*cse283;
  const amrex::Real cse649 = cse648*state.xn[15];
  const amrex::Real cse650 = cse648*state.xn[8];
  const amrex::Real cse651 = -cse124*cse173;
  const amrex::Real cse652 = cse291 + cse603;
  const amrex::Real cse653 = cse618 + cse638;
  const amrex::Real cse654 = cse310 + cse624;
  const amrex::Real cse655 = cse173*(cse119 + cse646);
  const amrex::Real cse656 = -cse382 + cse386 - cse389;
  const amrex::Real cse657 = cse165 + cse360;
  const amrex::Real cse658 = cse173*(-cse656 - cse657);
  const amrex::Real cse659 = cse355 + cse643;
  const amrex::Real cse660 = cse358 + cse644;
  const amrex::Real cse661 = -cse404 + cse429;
  const amrex::Real cse662 = cse173*cse661;
  const amrex::Real cse663 = cse545 - cse548;
  const amrex::Real cse664 = cse555 - cse556;
  const amrex::Real cse665 = cse559 - cse560;
  const amrex::Real cse666 = cse566 - cse567;
  const amrex::Real cse667 = -cse173*cse661;
  const amrex::Real cse668 = cse173*(cse372 + 5.3129999999999997e-12*cse411 + cse577 - 3.0029999999999999e-13*cse95);
  const amrex::Real cse669 = 2.3099999999999997e-13*cse255;
  const amrex::Real cse670 = cse173*(cse123 + cse581 + cse645 + cse656 + cse73);
  const amrex::Real cse671 = cse173*(cse374 + cse583);
  const amrex::Real cse672 = cse173*(cse167 + cse612 + cse657);
  const amrex::Real cse673 = 1.6737729999999998e-24*state.xn[0] + 1.6728620616289998e-24*state.xn[1] + 4.8185222061629005e-23*state.xn[10] + 2.6567630000000005e-23*state.xn[11] + 4.6637070000000001e-23*state.xn[12] + 4.6636159061629004e-23*state.xn[13] + 2.1618503000000003e-23*state.xn[14] + 2.8241403000000005e-23*state.xn[15] + 5.0204080616289998e-24*state.xn[16] + 2.6566719061629008e-23*state.xn[17] + 9.1093837099999994e-28*state.xn[2] + 3.3475459999999996e-24*state.xn[3] + 3.3466350616289996e-24*state.xn[4] + 6.6464729999999996e-24*state.xn[5] + 6.6455620616289996e-24*state.xn[6] + 1.9944730000000003e-23*state.xn[7] + 1.9943819061629003e-23*state.xn[8] + 4.6512360000000008e-23*state.xn[9];
  const amrex::Real cse674 = cse375 + 0.070000000000000007;
  const amrex::Real cse675 = 1.0/cse674;
  const amrex::Real cse676 = std::sqrt(cse375*cse675);
  const amrex::Real cse677 = 26.399999999999999*cse676 + 6.5;
  const amrex::Real cse678 = 2.1789607199999999e-11*cse180;
  const amrex::Real cse679 = std::pow(state.T, 0.14999999999999999);
  const amrex::Real cse680 = 0.00088240812266906664*std::pow(state.T, 0.65000000000000002) + 1;
  const amrex::Real cse681 = 1.0/cse680;
  const amrex::Real cse682 = cse679*cse681;
  const amrex::Real cse683 = cse681*state.xn[2];
  const amrex::Real cse684 = cse679*cse683;
  const amrex::Real cse685 = 4.4460389837719569e-9*cse684 + 626500000.0;
  const amrex::Real cse686 = 1.0/cse685;
  const amrex::Real cse687 = std::exp(-118400.0*cse3);
  const amrex::Real cse688 = cse686*cse687;
  const amrex::Real cse689 = 1.3338116951315871e-8*cse688;
  const amrex::Real cse690 = cse684*cse689 + 1;
  const amrex::Real cse691 = 1.0/cse690;
  const amrex::Real cse692 = cse688*cse691;
  const amrex::Real cse693 = 1.3659996292638161e-10*cse692;
  const amrex::Real cse694 = cse682*cse693;
  const amrex::Real cse695 = psigd_coll_interp2d(cse10, state.T);
  const amrex::Real cse696 = cse695*d2g;
  const amrex::Real cse697 = 100.0*cse696;
  const amrex::Real cse698 = cse10*cse697;
  const amrex::Real cse699 = 2.8677558904086741e-6*cse143 + 4.1397668473788896;
  const amrex::Real cse700 = -2.1544346900318838*std::pow(cse699, 1.0/6.0) + state.T;
  const amrex::Real cse701 = cse34*d2g;
  const amrex::Real cse702 = 3.2000000000000002e-32*cse2;
  const amrex::Real cse703 = cse701*cse702;
  const amrex::Real cse704 = cse700*cse703;
  const amrex::Real cse705 = 2.25*std::pow(state.T, 0.14699999999999999) + 5.2199999999999998;
  const amrex::Real cse706 = cse146 + 50;
  const amrex::Real cse707 = 0.0043*std::pow(cse706, 0.69199999999999995) + 1;
  const amrex::Real cse708 = 0.049959999999999997*std::pow(cse706, 0.43099999999999999)*cse707 + 1;
  const amrex::Real cse709 = 1.0/cse708;
  const amrex::Real cse710 = state.xn[2] + 1.0e-50;
  const amrex::Real cse711 = 1.0/cse710;
  const amrex::Real cse712 = cse2*cse711;
  const amrex::Real cse713 = 5.670589618368584e-11*cse143;
  const amrex::Real cse714 = cse712*cse713 + 50;
  const amrex::Real cse715 = std::log(cse714);
  const amrex::Real cse716 = 0.45350000000000001 + 2.234/cse715;
  const amrex::Real cse717 = std::pow(state.T, cse716);
  const amrex::Real cse718 = 1.4419999999999999 - 0.050889999999999998*cse715;
  const amrex::Real cse719 = std::exp(cse715*cse718);
  const amrex::Real cse720 = 1.8998126555754159e-29*cse717*cse719;
  const amrex::Real cse721 = cse426*cse720;
  const amrex::Real cse722 = cse721*state.xn[2];
  const amrex::Real cse723 = 1.0/state.xn[0];
  const amrex::Real cse724 = 3.0000000000000002e-15*d2g;
  const amrex::Real cse725 = cse10*cse724;
  const amrex::Real cse726 = cse723*cse725 + 1.9999999999999999e-7;
  const amrex::Real cse727 = 0.001*state.T;
  const amrex::Real cse728 = cse727 + 1;
  const amrex::Real cse729 = std::pow(cse728, -2);
  const amrex::Real cse730 = 1.0/cse728;
  const amrex::Real cse731 = std::pow(10, -2.0230000000000001*cse729 - 3.6709999999999998*cse730 - 11.08);
  const amrex::Real cse732 = cse11*cse731;
  const amrex::Real cse733 = cse732*state.xn[3];
  const amrex::Real cse734 = std::pow(10, -2.3900000000000001*cse729 + 0.055500000000000001*cse730 - 11.06);
  const amrex::Real cse735 = cse11*cse734;
  const amrex::Real cse736 = cse735*state.xn[0];
  const amrex::Real cse737 = cse733 + cse736;
  const amrex::Real cse738 = 1.0/cse737;
  const amrex::Real cse739 = cse11*cse738;
  const amrex::Real cse740 = cse726*cse739 + 1;
  const amrex::Real cse741 = 3.2043540000000002e-13 + 6.7291434000000005e-12/cse740;
  const amrex::Real cse742 = cse8*std::log(cse10);
  const amrex::Real cse743 = cse742 < 2;
  const amrex::Real cse744 = cse742 < 4;
  const amrex::Real cse745 = cse742 < 7;
  const amrex::Real cse746 = cse742 < 10;
  const amrex::Real cse747 = ((cse743) ? (
   10.0
)
: ((cse744) ? (
   (3.0/2.0)*cse742 + 7
)
: ((cse745) ? (
   (4.0/3.0)*cse742 + 23.0/3.0
)
: ((cse746) ? (
   (1.0/3.0)*cse742 + 44.0/3.0
)
: (
   18.0
)))));
  const amrex::Real cse748 = cse71*state.xn[2];
  const amrex::Real cse749 = (1.0/100.0)*state.T;
  const amrex::Real cse750 = std::log(cse749);
  const amrex::Real cse751 = 0.0086999999999999994*cse750;
  const amrex::Real cse752 = std::pow(cse749, cse751 + 0.12809999999999999);
  const amrex::Real cse753 = 7.5799999999999997e-10*cse752*state.xn[0];
  const amrex::Real cse754 = state.T < 500;
  const amrex::Real cse755 = std::pow(state.T, 0.070000000000000007);
  const amrex::Real cse756 = 0.75*((cse754) ? (
   1.1000000000000001e-13*state.T + 5.3300000000000002e-10
)
: (
   3.74757785025e-10*cse755
)) + 0.25*((cse754) ? (
   3.3000000000000001e-13*state.T + 4.4299999999999997e-10
)
: (
   3.8899728635600003e-10*cse755
));
  const amrex::Real cse757 = 4.5299999999999998e-6*cse748 + cse753 + cse756*state.xn[3];
  const amrex::Real cse758 = cse757 + 2.3e-6;
  const amrex::Real cse759 = std::exp(-91.209999999999994*cse3);
  const amrex::Real cse760 = cse759/cse758;
  const amrex::Real cse761 = 2*cse760;
  const amrex::Real cse762 = cse757*cse761 + 1;
  const amrex::Real cse763 = 1.0/cse762;
  const amrex::Real cse764 = cse763*state.xn[8];
  const amrex::Real cse765 = cse757*cse760;
  const amrex::Real cse766 = 5.7927337833400002e-20*cse765;
  const amrex::Real cse767 = std::pow(state.T, 1.0/4.0);
  const amrex::Real cse768 = 680.10000000000002/cse767;
  const amrex::Real cse769 = 1.75*cse767;
  const amrex::Real cse770 = cse768*state.xn[2] + cse769*state.xn[0] + state.xn[3];
  const amrex::Real cse771 = std::exp(-state.T);
  const amrex::Real cse772 = 1 - cse771;
  const amrex::Real cse773 = std::pow(cse772, 1000);
  const amrex::Real cse774 = l0_co_interp1d(state.T);
  const amrex::Real cse775 = 1.0/cse774;
  const amrex::Real cse776 = M_SQRT2;
  const amrex::Real cse777 = std::max(gradv, 5.7445531974402559e-18*cse2*cse776);
  const amrex::Real cse778 = 100000.0/cse777;
  const amrex::Real cse779 = ncol_H <= 0;
  const amrex::Real cse780 = cse11*ncol_H;
  const amrex::Real cse781 = std::max(1.0000000000000001e-30, cse780);
  const amrex::Real cse782 = 1756701454.4310031*ncol_CO;
  const amrex::Real cse783 = ((cse779) ? (
   cse778*state.xn[9]
)
: (
   cse782/std::sqrt(cse781)
));
  const amrex::Real cse784 = llte_co_interp2d(state.T, cse783);
  const amrex::Real cse785 = 1.0/cse784;
  const amrex::Real cse786 = nhalf_co_interp2d(state.T, cse783);
  const amrex::Real cse787 = cse774*cse785;
  const amrex::Real cse788 = -cse786*cse787 + 1;
  const amrex::Real cse789 = 1.0/cse786;
  const amrex::Real cse790 = cse770*cse789;
  const amrex::Real cse791 = alpha_co_interp2d(state.T, cse783);
  const amrex::Real cse792 = std::pow(cse790, cse791);
  const amrex::Real cse793 = cse775*cse792;
  const amrex::Real cse794 = cse788*cse793;
  const amrex::Real cse795 = cse770*cse785 + cse775 + cse794;
  const amrex::Real cse796 = 1.0/cse795;
  const amrex::Real cse797 = cse773*cse796;
  const amrex::Real cse798 = cse797*state.xn[9];
  const amrex::Real cse799 = cse594 + 1.9999999999999999e-7;
  const amrex::Real cse800 = cse739*cse799 + 1;
  const amrex::Real cse801 = 6.4087080000000003e-13 + 2.5634832e-11/cse800;
  const amrex::Real cse802 = 7.1777529600000012e-12*cse573;
  const amrex::Real cse803 = std::pow(state.T, 0.92600000000000005);
  const amrex::Real cse804 = 1.08e-14*cse803;
  const amrex::Real cse805 = 0.16*cse750;
  const amrex::Real cse806 = std::pow(cse749, 0.755 - cse805);
  const amrex::Real cse807 = 4.34e-10*cse806;
  const amrex::Real cse808 = cse807*state.xn[0];
  const amrex::Real cse809 = 0.042999999999999997*cse750;
  const amrex::Real cse810 = 5.2499999999999998e-13*std::pow(cse749, cse809 + 0.88900000000000001);
  const amrex::Real cse811 = 0.52500000000000002*cse750;
  const amrex::Real cse812 = 2.2500000000000003e-12*std::pow(cse749, cse811 + 1.198);
  const amrex::Real cse813 = cse810 + cse812;
  const amrex::Real cse814 = cse804*state.xn[2] + cse808 + cse813*state.xn[3];
  const amrex::Real cse815 = std::pow(state.T, -0.074999999999999997);
  const amrex::Real cse816 = 5.1199999999999999e-10*cse815;
  const amrex::Real cse817 = 0.0030000000000000001*cse750;
  const amrex::Real cse818 = std::pow(cse749, 0.41899999999999998 - cse817);
  const amrex::Real cse819 = 3.5700000000000001e-10*cse818;
  const amrex::Real cse820 = cse819*state.xn[0];
  const amrex::Real cse821 = 0.025000000000000001*cse750;
  const amrex::Real cse822 = 3.7250000000000001e-11*std::pow(cse749, cse821 + 0.26400000000000001);
  const amrex::Real cse823 = 1.0274999999999999e-10*std::pow(cse749, cse809 + 0.29599999999999999);
  const amrex::Real cse824 = cse822 + cse823;
  const amrex::Real cse825 = cse816*state.xn[2] + cse820 + cse824*state.xn[3];
  const amrex::Real cse826 = std::exp(-326.60000000000002*cse3);
  const amrex::Real cse827 = (1.0/5.0)*cse826;
  const amrex::Real cse828 = cse825*cse827;
  const amrex::Real cse829 = std::exp(-425.40000000000003*cse3);
  const amrex::Real cse830 = cse814*cse829;
  const amrex::Real cse831 = std::pow(state.T, -0.025999999999999999);
  const amrex::Real cse832 = 4.8599999999999998e-10*cse831;
  const amrex::Real cse833 = 0.0060000000000000001*cse750;
  const amrex::Real cse834 = std::pow(cse749, 0.36899999999999999 - cse833);
  const amrex::Real cse835 = 3.1899999999999998e-10*cse834;
  const amrex::Real cse836 = cse835*state.xn[0];
  const amrex::Real cse837 = 0.041000000000000002*cse750;
  const amrex::Real cse838 = 4.7499999999999998e-11*std::pow(cse749, cse837 + 0.20300000000000001);
  const amrex::Real cse839 = 0.058000000000000003*cse750;
  const amrex::Real cse840 = 1.6725000000000002e-10*std::pow(cse749, cse839 + 0.23699999999999999);
  const amrex::Real cse841 = cse838 + cse840;
  const amrex::Real cse842 = cse832*state.xn[2] + cse836 + cse841*state.xn[3];
  const amrex::Real cse843 = (1.0/15.0)*cse842;
  const amrex::Real cse844 = cse825 + 8.9099999999999997e-5;
  const amrex::Real cse845 = cse827*cse842;
  const amrex::Real cse846 = cse814*cse828 + cse830*cse843 + cse844*cse845;
  const amrex::Real cse847 = cse814 + 1.7499999999999998e-5;
  const amrex::Real cse848 = cse842 + 1.34e-10;
  const amrex::Real cse849 = std::exp(-98.800000000000011*cse3);
  const amrex::Real cse850 = cse814*cse849;
  const amrex::Real cse851 = (1.0/3.0)*cse850;
  const amrex::Real cse852 = std::exp(-227.80000000000001*cse3);
  const amrex::Real cse853 = cse825*cse852;
  const amrex::Real cse854 = (3.0/5.0)*cse853;
  const amrex::Real cse855 = cse845*cse847 + cse847*cse854 + cse848*cse854;
  const amrex::Real cse856 = cse844*cse847 + cse844*cse848 + cse846 + cse848*cse851 + cse855;
  const amrex::Real cse857 = 1.0/cse856;
  const amrex::Real cse858 = 2.3872025442750958e-19*cse857;
  const amrex::Real cse859 = 2.8023005140019997e-18*cse857;
  const amrex::Real cse860 = cse846*cse858 + cse855*cse859;
  const amrex::Real cse861 = state.T < 100.0;
  const amrex::Real cse862 = state.T < 6000.0;
  const amrex::Real cse863 = state.T < 10;
  const amrex::Real cse864 = std::log(cse727);
  const amrex::Real cse865 = cse8*cse864;
  const amrex::Real cse866 = std::pow(cse7, -8);
  const amrex::Real cse867 = std::pow(cse864, 8)*cse866;
  const amrex::Real cse868 = std::pow(cse7, -7);
  const amrex::Real cse869 = std::pow(cse864, 7)*cse868;
  const amrex::Real cse870 = std::pow(cse7, -6);
  const amrex::Real cse871 = std::pow(cse864, 6)*cse870;
  const amrex::Real cse872 = std::pow(cse7, -5);
  const amrex::Real cse873 = std::pow(cse864, 5)*cse872;
  const amrex::Real cse874 = std::pow(cse7, -4);
  const amrex::Real cse875 = std::pow(cse864, 4)*cse874;
  const amrex::Real cse876 = std::pow(cse7, -3);
  const amrex::Real cse877 = std::pow(cse864, 3);
  const amrex::Real cse878 = cse876*cse877;
  const amrex::Real cse879 = cse13*std::pow(cse864, 2);
  const amrex::Real cse880 = std::pow(10.0, 16.815729999999999*cse865 + 70.609154000000004*cse867 + 364.14445999999998*cse869 + 801.81246999999996*cse871 + 983.67575999999997*cse873 + 734.71650999999997*cse875 + 343.1918*cse878 + 96.743155000000002*cse879 - 21.928795999999998)*state.xn[2];
  const amrex::Real cse881 = state.xn[5] + state.xn[6];
  const amrex::Real cse882 = std::pow(10.0, 1.5714710999999999*cse865 + 0.32168730000000001*cse873 - 0.51002221000000003*cse875 - 0.23619984999999999*cse878 + 0.015391166*cse879 - 22.089523)*state.xn[1] + std::pow(10.0, 2.1892372*cse865 + 0.19191374999999999*cse873 - 0.16596184*cse875 + 0.29036281000000003*cse878 - 0.81520437999999995*cse879 - 23.689236999999999)*cse881 + std::pow(10.0, 2.0943374000000001*cse8*cse864 - 0.033638326000000003*cse873 - 0.14913216000000001*cse875 + 0.43693353000000001*cse876*cse877 - 0.77151435999999995*cse879 - 23.962112000000001)*state.xn[3];
  const amrex::Real cse883 = std::pow(10, 37.383713*cse865 + 3.8479610000000002*cse873 + 20.159831000000001*cse875 + 48.656103000000002*cse878 + 58.145166000000003*cse879 - 16.818342000000001)*state.xn[0] + cse882;
  const amrex::Real cse884 = std::pow(10.0, 1.6802758*cse865 - 6.3701156000000001*cse867 + 6.4380698000000001*cse869 + 8.9167182999999994*cse871 - 8.8077017000000009*cse873 - 4.7274035999999997*cse875 + 4.0406627000000004*cse878 + 0.93310621999999999*cse879 - 22.921188999999998)*state.xn[2];
  const amrex::Real cse885 = state.T < 1000.0;
  const amrex::Real cse886 = std::pow(10.0, 3.5692468000000002*cse8*cse864 - 4.2519023000000002*cse873 - 21.328264000000001*cse875 - 27.850082*cse878 - 11.33286*cse879 - 24.311209000000002)*state.xn[0] + cse882;
  const amrex::Real cse887 = std::pow(10.0, 4.6450521*cse865 + 1.5538288*cse873 - 5.5108049000000001*cse875 + 5.9369081000000001*cse878 - 3.7209846*cse879 - 24.311209000000002)*state.xn[0] + cse882;
  const amrex::Real cse888 = std::pow(10.0, -11.945856385319289*cse13 + 8.3228160852326614*cse8 + 28.69470131298657*cse872 - 56.79810800956944*cse874 + 34.150687525022562*cse876 - 24.311209000000002)*state.xn[0] + std::pow(10.0, -2.6171337683481011*cse13 + 3.9225864834863131*cse8 + 3.5440891133599637*cse872 - 1.7105157385968939*cse874 + 1.6702447513374001*cse876 - 23.689236999999999)*cse881 + std::pow(10.0, -2.4768712409536775*cse13 + 3.7525488682084647*cse8 - 0.62120210234156448*cse872 - 1.5370575959566981*cse874 + 2.5133588394664677*cse876 - 23.962112000000001)*state.xn[3] + std::pow(10.0, 0.049411830040524525*cse13 + 2.8156982240432273*cse8 + 5.9406293599919788*cse872 - 5.2566362076906961*cse874 - 1.3586848802337363*cse876 - 22.089523)*state.xn[1];
  const amrex::Real cse889 = ((cse754 && cse861 && cse862) ? (
   cse863 || cse880 + cse883 < 1.0e-100
)
: (
   ((cse861 && cse862) ? (
      cse863 || cse883 + cse884 < 1.0e-100
   )
   : (
      ((cse754 && cse862 && cse885) ? (
         cse863 || cse880 + cse886 < 1.0e-100
      )
      : (
         ((cse862 && cse885) ? (
            cse863 || cse884 + cse886 < 1.0e-100
         )
         : (
            ((cse754 && cse862) ? (
               cse863 || cse880 + cse887 < 1.0e-100
            )
            : (
               ((cse862) ? (
                  cse863 || cse884 + cse887 < 1.0e-100
               )
               : (
                  ((cse754) ? (
                     cse863 || std::pow(10.0, 310.58441786958315*cse13 + 30.129743459482278*cse8 + 7500.646099484944*cse866 + 21588.957206746247*cse868 + 26530.843406310876*cse870 + 18165.631967965237*cse872 + 7572.4886742758581*cse874 + 1974.131269262874*cse876 - 21.928795999999998)*state.xn[2] + cse888 < 1.0e-100
                  )
                  : (
                     cse863 || std::pow(10.0, 2.9956460707652877*cse13 + 3.0106500755647456*cse8 - 676.68255490510751*cse866 + 381.69251128040054*cse868 + 295.04162851880631*cse870 - 162.65264843551881*cse872 - 48.723840736518795*cse874 + 23.242975457496804*cse876 - 22.921188999999998)*state.xn[2] + cse888 < 1.0e-100
                  ))
               ))
            ))
         ))
      ))
   ))
));
  const amrex::Real cse890 = ((cse862) ? (
   cse3
)
: (
   0.00016666666666666666
));
  const amrex::Real cse891 = std::exp(-11700.0*cse890);
  const amrex::Real cse892 = std::exp(-5860.0*cse890);
  const amrex::Real cse893 = std::exp(-510.0*cse890);
  const amrex::Real cse894 = std::exp(-2197000.0*((cse862) ? (
   std::pow(state.T, -3)
)
: (
   4.6296296296296294e-12
)));
  const amrex::Real cse895 = cse894*((cse862) ? (
   std::pow(state.T, 3.7599999999999998)
)
: (
   160638152785302.53
));
  const amrex::Real cse896 = 6.0142468035272636e-8*((cse862) ? (
   std::pow(state.T, 2.1000000000000001)
)
: (
   85924621.118090734
)) + 1.0;
  const amrex::Real cse897 = 1.0/cse896;
  const amrex::Real cse898 = 1.6e-18*cse891 + 6.7e-19*cse892 + 3.0e-24*cse893 + 4.985670872372847e-33*cse895*cse897;
  const amrex::Real cse899 = std::log(0.001*((cse862) ? (
   state.T
)
: (
   6000.0
)));
  const amrex::Real cse900 = std::pow(cse899, 5);
  const amrex::Real cse901 = cse872*cse900;
  const amrex::Real cse902 = std::pow(cse899, 4);
  const amrex::Real cse903 = cse874*cse902;
  const amrex::Real cse904 = std::pow(cse899, 3);
  const amrex::Real cse905 = std::pow(cse899, 2);
  const amrex::Real cse906 = cse13*cse905;
  const amrex::Real cse907 = std::pow(10.0, 2.0943374000000001*cse8*cse899 + 0.43693353000000001*cse876*cse904 - 0.033638326000000003*cse901 - 0.14913216000000001*cse903 - 0.77151435999999995*cse906 - 23.962112000000001);
  const amrex::Real cse908 = cse907*state.xn[3];
  const amrex::Real cse909 = cse8*cse899;
  const amrex::Real cse910 = cse876*cse904;
  const amrex::Real cse911 = std::pow(10.0, 0.32168730000000001*cse901 - 0.51002221000000003*cse903 + 0.015391166*cse906 + 1.5714710999999999*cse909 - 0.23619984999999999*cse910 - 22.089523);
  const amrex::Real cse912 = cse911*state.xn[1];
  const amrex::Real cse913 = std::pow(10.0, 0.19191374999999999*cse901 - 0.16596184*cse903 - 0.81520437999999995*cse906 + 2.1892372*cse909 + 0.29036281000000003*cse910 - 23.689236999999999);
  const amrex::Real cse914 = cse881*cse913;
  const amrex::Real cse915 = std::pow(10, 3.8479610000000002*cse901 + 20.159831000000001*cse903 + 58.145166000000003*cse906 + 37.383713*cse909 + 48.656103000000002*cse910 - 16.818342000000001);
  const amrex::Real cse916 = ((cse862) ? (
   cse861
)
: (
   false
));
  const amrex::Real cse917 = std::pow(10.0, 3.5692468000000002*cse8*cse899 - 4.2519023000000002*cse901 - 21.328264000000001*cse903 - 11.33286*cse906 - 27.850082*cse910 - 24.311209000000002);
  const amrex::Real cse918 = ((cse862) ? (
   cse885
)
: (
   false
));
  const amrex::Real cse919 = std::pow(10.0, 1.5538288*cse901 - 5.5108049000000001*cse903 - 3.7209846*cse906 + 4.6450521*cse909 + 5.9369081000000001*cse910 - 24.311209000000002);
  const amrex::Real cse920 = ((cse916) ? (
   cse915
)
: ((cse918) ? (
   cse917
)
: (
   cse919
)));
  const amrex::Real cse921 = cse866*std::pow(cse899, 8);
  const amrex::Real cse922 = std::pow(cse899, 7);
  const amrex::Real cse923 = cse868*cse922;
  const amrex::Real cse924 = std::pow(cse899, 6);
  const amrex::Real cse925 = cse870*cse924;
  const amrex::Real cse926 = std::pow(10.0, 983.67575999999997*cse901 + 734.71650999999997*cse903 + 96.743155000000002*cse906 + 16.815729999999999*cse909 + 343.1918*cse910 + 70.609154000000004*cse921 + 364.14445999999998*cse923 + 801.81246999999996*cse925 - 21.928795999999998);
  const amrex::Real cse927 = std::pow(10.0, -8.8077017000000009*cse901 - 4.7274035999999997*cse903 + 0.93310621999999999*cse906 + 1.6802758*cse909 + 4.0406627000000004*cse910 - 6.3701156000000001*cse921 + 6.4380698000000001*cse923 + 8.9167182999999994*cse925 - 22.921188999999998);
  const amrex::Real cse928 = ((cse754) ? (
   cse926
)
: (
   cse927
));
  const amrex::Real cse929 = cse908 + cse912 + cse914 + cse920*state.xn[0] + cse928*state.xn[2];
  const amrex::Real cse930 = 1.0/cse929;
  const amrex::Real cse931 = cse898*cse930 + 1;
  const amrex::Real cse932 = 1.0/cse931;
  const amrex::Real cse933 = ((cse889) ? (
   0.0
)
: (
   cse898*cse932
));
  const amrex::Real cse934 = 0.045999999999999999*cse750;
  const amrex::Real cse935 = std::pow(cse749, cse934 + 0.22800000000000001);
  const amrex::Real cse936 = 8.9000000000000003e-11*cse935;
  const amrex::Real cse937 = cse936*state.xn[0];
  const amrex::Real cse938 = 0.037999999999999999*cse750;
  const amrex::Real cse939 = 5.1749999999999996e-11*std::pow(cse749, cse938 + 0.16900000000000001);
  const amrex::Real cse940 = 0.048000000000000001*cse750;
  const amrex::Real cse941 = 2.15e-11*std::pow(cse749, cse940 - 0.01);
  const amrex::Real cse942 = cse939 + cse941;
  const amrex::Real cse943 = 0.0070527699999999999*cse147;
  const amrex::Real cse944 = cse943 - 0.11133800000000001;
  const amrex::Real cse945 = cse147*cse944 + 0.69763799999999998;
  const amrex::Real cse946 = cse147*cse945 - 1.3074300000000001;
  const amrex::Real cse947 = 0.093813800000000003*cse147;
  const amrex::Real cse948 = cse947 - 3.0328300000000001;
  const amrex::Real cse949 = cse147*cse948 + 36.180300000000003;
  const amrex::Real cse950 = cse147*cse949 - 187.47399999999999;
  const amrex::Real cse951 = std::exp(((cse885) ? (
   cse147*cse946 - 7.6973500000000001
)
: (
   cse147*cse950 + 350.60899999999998
)));
  const amrex::Real cse952 = 1.7258000000000001e-6*cse748;
  const amrex::Real cse953 = cse951*cse952;
  const amrex::Real cse954 = cse937 + cse942*state.xn[3] + cse953;
  const amrex::Real cse955 = std::exp(-101.30000000000001*cse3);
  const amrex::Real cse956 = cse954*cse955;
  const amrex::Real cse957 = std::pow(cse749, cse934 + 0.23100000000000001);
  const amrex::Real cse958 = 2.6400000000000002e-10*cse957;
  const amrex::Real cse959 = cse958*state.xn[0];
  const amrex::Real cse960 = 0.031*cse750;
  const amrex::Real cse961 = 1.11e-10*std::pow(cse749, cse960 + 0.26300000000000001);
  const amrex::Real cse962 = 0.064000000000000001*cse750;
  const amrex::Real cse963 = 4.3749999999999998e-11*std::pow(cse749, cse962 + 0.071999999999999995);
  const amrex::Real cse964 = cse961 + cse963;
  const amrex::Real cse965 = 0.00235272*cse147;
  const amrex::Real cse966 = cse965 - 0.041816600000000002;
  const amrex::Real cse967 = cse147*cse966 + 0.35826400000000003;
  const amrex::Real cse968 = cse147*cse967 - 0.57443;
  const amrex::Real cse969 = 0.097857299999999994*cse147;
  const amrex::Real cse970 = cse969 - 3.1926800000000002;
  const amrex::Real cse971 = cse147*cse970 + 38.504899999999999;
  const amrex::Real cse972 = cse147*cse971 - 202.19300000000001;
  const amrex::Real cse973 = std::exp(((cse885) ? (
   cse147*cse968 - 7.4386999999999999
)
: (
   cse147*cse972 + 386.18599999999998
)));
  const amrex::Real cse974 = cse952*cse973;
  const amrex::Real cse975 = cse959 + cse964*state.xn[3] + cse974;
  const amrex::Real cse976 = (25.0/3.0)*cse975;
  const amrex::Real cse977 = 0.057000000000000002*cse750;
  const amrex::Real cse978 = std::pow(cse749, cse977 + 0.115);
  const amrex::Real cse979 = 1.26e-10*cse978;
  const amrex::Real cse980 = cse979*state.xn[0];
  const amrex::Real cse981 = 0.049000000000000002*cse750;
  const amrex::Real cse982 = 5.3249999999999997e-11*std::pow(cse749, cse981 - 0.0040000000000000001);
  const amrex::Real cse983 = 0.10199999999999999*cse750;
  const amrex::Real cse984 = 1.675e-11*std::pow(cse749, cse983 - 0.085000000000000006);
  const amrex::Real cse985 = cse982 + cse984;
  const amrex::Real cse986 = 0.00065632500000000003*cse147;
  const amrex::Real cse987 = -cse986 - 0.015089200000000001;
  const amrex::Real cse988 = cse147*cse987 + 0.36118400000000001;
  const amrex::Real cse989 = cse147*cse988 - 0.77378199999999997;
  const amrex::Real cse990 = 0.10508000000000001*cse147;
  const amrex::Real cse991 = cse990 - 3.4762;
  const amrex::Real cse992 = cse147*cse991 + 42.594999999999999;
  const amrex::Real cse993 = cse147*cse992 - 227.91300000000001;
  const amrex::Real cse994 = std::exp(((cse885) ? (
   cse147*cse989 - 9.2514099999999999
)
: (
   cse147*cse993 + 444.60000000000002
)));
  const amrex::Real cse995 = 2.8763333333333333e-6*cse994;
  const amrex::Real cse996 = cse748*cse995;
  const amrex::Real cse997 = cse980 + cse985*state.xn[3] + cse996;
  const amrex::Real cse998 = std::exp(-62.460000000000001*cse3);
  const amrex::Real cse999 = 5*cse998;
  const amrex::Real cse1000 = cse997*cse999;
  const amrex::Real cse1001 = cse997 + 7.8800000000000004e-8;
  const amrex::Real cse1002 = cse954*cse999;
  const amrex::Real cse1003 = cse1000*cse975 + cse1001*cse1002 + cse956*cse976;
  const amrex::Real cse1004 = cse975 + 2.65e-7;
  const amrex::Real cse1005 = cse954 + 1.81e-14;
  const amrex::Real cse1006 = std::exp(-38.840000000000003*cse3);
  const amrex::Real cse1007 = cse1006*cse975;
  const amrex::Real cse1008 = (5.0/3.0)*cse1007;
  const amrex::Real cse1009 = std::exp(-23.619999999999997*cse3);
  const amrex::Real cse1010 = cse1009*cse997;
  const amrex::Real cse1011 = 3*cse1010;
  const amrex::Real cse1012 = cse1002*cse1004 + cse1004*cse1011 + cse1005*cse1011;
  const amrex::Real cse1013 = cse1001*cse1004 + cse1001*cse1005 + cse1003 + cse1005*cse1008 + cse1012;
  const amrex::Real cse1014 = 1.0/cse1013;
  const amrex::Real cse1015 = 1.4210469458259591e-21*cse1014;
  const amrex::Real cse1016 = 2.5697412351439998e-22*cse1014;
  const amrex::Real cse1017 = state.T < 1000000.0;
  const amrex::Real cse1018 = ((cse1017) ? (
   cse1003*cse1015 + cse1012*cse1016
)
: (
   0.0
));
  const amrex::Real cse1019 = (5.6705896183685846e-35*state.c_hat*cse10*cse142*cse705*cse709*d2g + 1.602177e-12*crate*cse677*state.xn[0] + 3.2043540000000001e-12*crate*cse747*state.xn[3] + 3.0000000000000002e-15*cse10*cse741*d2g*state.xn[0] - cse1018*state.xn[7] - cse134*cse694 - 7.1777529600000012e-12*cse241*cse30 + cse594*cse801*state.xn[3] - cse678*state.xn[0] - cse722 - cse764*cse766 - cse770*cse798 - cse802*state.xn[0] - cse860*state.xn[11] - cse933*state.xn[3] - std::min(cse698, cse704))/std::pow(cse673, 2);
  const amrex::Real cse1020 = 1.0/cse673;
  const amrex::Real cse1021 = cse826*cse842;
  const amrex::Real cse1022 = 7.1400000000000007e-11*cse818;
  const amrex::Real cse1023 = cse814*cse826;
  const amrex::Real cse1024 = 8.68e-11*cse806;
  const amrex::Real cse1025 = cse825*cse826;
  const amrex::Real cse1026 = cse829*cse842;
  const amrex::Real cse1027 = 6.3800000000000002e-11*cse834;
  const amrex::Real cse1028 = cse826*cse844;
  const amrex::Real cse1029 = cse1021*cse1022 + cse1022*cse1023 + cse1024*cse1025 + 2.8933333333333332e-11*cse1026*cse806 + cse1027*cse1028 + 2.1266666666666666e-11*cse830*cse834;
  const amrex::Real cse1030 = cse826*cse847;
  const amrex::Real cse1031 = 2.1419999999999999e-10*cse818;
  const amrex::Real cse1032 = cse848*cse852;
  const amrex::Real cse1033 = cse847*cse852;
  const amrex::Real cse1034 = cse1021*cse1024 + cse1027*cse1030 + cse1031*cse1032 + cse1031*cse1033 + 2.604e-10*cse806*cse853 + 1.9139999999999998e-10*cse834*cse853;
  const amrex::Real cse1035 = std::pow(cse856, -2);
  const amrex::Real cse1036 = cse848*cse849;
  const amrex::Real cse1037 = cse1035*(-cse1029 - cse1034 - 1.4466666666666666e-10*cse1036*cse806 - cse807*cse844 - cse819*cse847 - cse819*cse848 - 1.0633333333333333e-10*cse834*cse850 - cse835*cse844);
  const amrex::Real cse1038 = 2.3872025442750958e-19*cse846;
  const amrex::Real cse1039 = 2.8023005140019997e-18*cse855;
  const amrex::Real cse1040 = std::pow(cse929, -2);
  const amrex::Real cse1041 = std::pow(cse931, -2);
  const amrex::Real cse1042 = cse1040*cse1041*std::pow(cse898, 2);
  const amrex::Real cse1043 = cse955*cse975;
  const amrex::Real cse1044 = cse997*cse998;
  const amrex::Real cse1045 = 1.32e-9*cse957;
  const amrex::Real cse1046 = 6.3e-10*cse978;
  const amrex::Real cse1047 = cse954*cse998;
  const amrex::Real cse1048 = cse975*cse998;
  const amrex::Real cse1049 = cse1001*cse998;
  const amrex::Real cse1050 = 4.4500000000000001e-10*cse935;
  const amrex::Real cse1051 = 7.4166666666666679e-10*cse1043*cse935 + cse1044*cse1045 + cse1046*cse1047 + cse1046*cse1048 + cse1049*cse1050 + 2.2000000000000003e-9*cse956*cse957;
  const amrex::Real cse1052 = cse1004*cse998;
  const amrex::Real cse1053 = 3.7800000000000004e-10*cse978;
  const amrex::Real cse1054 = cse1005*cse1009;
  const amrex::Real cse1055 = cse1004*cse1009;
  const amrex::Real cse1056 = 2.6700000000000001e-10*cse1010*cse935 + 7.9200000000000006e-10*cse1010*cse957 + cse1045*cse1047 + cse1050*cse1052 + cse1053*cse1054 + cse1053*cse1055;
  const amrex::Real cse1057 = std::pow(cse1013, -2);
  const amrex::Real cse1058 = cse1005*cse1006;
  const amrex::Real cse1059 = cse1057*(-cse1001*cse936 - cse1001*cse958 - cse1004*cse979 - cse1005*cse979 - 1.4833333333333334e-10*cse1007*cse935 - cse1051 - cse1056 - 4.4000000000000003e-10*cse1058*cse957);
  const amrex::Real cse1060 = 1.4210469458259591e-21*cse1003;
  const amrex::Real cse1061 = 2.5697412351439998e-22*cse1012;
  const amrex::Real cse1062 = std::pow(cse800, -2);
  const amrex::Real cse1063 = cse35*cse738*cse799;
  const amrex::Real cse1064 = cse36*cse731;
  const amrex::Real cse1065 = cse38*cse734;
  const amrex::Real cse1066 = cse1064 + cse1065;
  const amrex::Real cse1067 = std::pow(cse737, -2);
  const amrex::Real cse1068 = cse1067*cse11;
  const amrex::Real cse1069 = cse1068*(cse1066 - cse735);
  const amrex::Real cse1070 = cse35*ncol_H*(((cse780 - 1.0000000000000001e-30 < 0) ? (
   0
)
: ((cse780 - 1.0000000000000001e-30 == 0) ? (
   1.0/2.0
)
: (
   1
))))/std::pow(cse781, 3.0/2.0);
  const amrex::Real cse1071 = cse1070*ncol_CO;
  const amrex::Real cse1072 = ((cse779) ? (
   0
)
: (
   878350727.21550155*cse1071
));
  const amrex::Real cse1073 = alpha_co_interp2d_partial_1(state.T, cse783);
  const amrex::Real cse1074 = std::log(cse790);
  const amrex::Real cse1075 = cse1073*cse1074;
  const amrex::Real cse1076 = cse770/std::pow(cse786, 2);
  const amrex::Real cse1077 = nhalf_co_interp2d_partial_1(state.T, cse783);
  const amrex::Real cse1078 = cse1072*cse1077;
  const amrex::Real cse1079 = cse791/cse770;
  const amrex::Real cse1080 = cse1079*cse786;
  const amrex::Real cse1081 = std::pow(cse784, -2);
  const amrex::Real cse1082 = llte_co_interp2d_partial_1(state.T, cse783);
  const amrex::Real cse1083 = -cse1072*cse1081*cse1082*cse770 + cse793*(cse1072*cse1081*cse1082*cse774*cse786 - cse1078*cse787);
  const amrex::Real cse1084 = cse770*state.xn[9];
  const amrex::Real cse1085 = cse1084*cse773/std::pow(cse795, 2);
  const amrex::Real cse1086 = cse760*cse764;
  const amrex::Real cse1087 = std::pow(cse740, -2);
  const amrex::Real cse1088 = -cse35*cse726*cse738;
  const amrex::Real cse1089 = cse723*cse724;
  const amrex::Real cse1090 = 1.5159999999999999e-9*cse752;
  const amrex::Real cse1091 = std::pow(cse758, -2);
  const amrex::Real cse1092 = cse1091*cse757*cse759;
  const amrex::Real cse1093 = cse766*state.xn[8]/std::pow(cse762, 2);
  const amrex::Real cse1094 = cse426*psigd_coll_interp2d_partial_0(cse10, state.T);
  const amrex::Real cse1095 = -cse698 + cse704;
  const amrex::Real cse1096 = (((cse1095 < 0) ? (
   0
)
: ((cse1095 == 0) ? (
   1.0/2.0
)
: (
   1
))));
  const amrex::Real cse1097 = cse1096*(100.0*cse1094 + cse697);
  const amrex::Real cse1098 = cse11*cse8;
  const amrex::Real cse1099 = ((cse743) ? (
   0
)
: ((cse744) ? (
   (3.0/2.0)*cse1098
)
: ((cse745) ? (
   (4.0/3.0)*cse1098
)
: ((cse746) ? (
   (1.0/3.0)*cse1098
)
: (
   0
)))));
  const amrex::Real cse1100 = d2g*state.xn[2];
  const amrex::Real cse1101 = cse1100*cse720;
  const amrex::Real cse1102 = (((cse1095 > 0) ? (
   0
)
: ((cse1095 == 0) ? (
   1.0/2.0
)
: (
   1
))));
  const amrex::Real cse1103 = cse1102*cse700*cse702*d2g;
  const amrex::Real cse1104 = cse1103*(cse234 + cse242 + 2*state.xn[1] + 2*state.xn[10] + 2*state.xn[14] + 2*state.xn[15] + 6*state.xn[16] + 4*state.xn[4]);
  const amrex::Real cse1105 = cse35*state.xn[2];
  const amrex::Real cse1106 = std::pow(cse674, -2);
  const amrex::Real cse1107 = 0.5*cse1106;
  const amrex::Real cse1108 = cse203/std::pow(cse10, 3);
  const amrex::Real cse1109 = -0.5*cse1105*cse675 + cse1107*cse1108;
  const amrex::Real cse1110 = -5.6705896183685846e-35*state.c_hat*cse142*cse705*cse709*d2g - 4.22974728e-11*crate*cse10*cse106*cse1109*cse674*cse676*state.xn[0] - 3.2043540000000001e-12*crate*cse1099*state.xn[3] + cse1097 + cse1101 + cse1104 - 3.0000000000000002e-15*cse741*d2g*state.xn[0];
  const amrex::Real cse1111 = cse688*std::pow(state.T, -1.8500000000000001);
  const amrex::Real cse1112 = cse134*cse691;
  const amrex::Real cse1113 = std::pow(state.T, -0.84999999999999998);
  const amrex::Real cse1114 = std::pow(state.T, -0.19999999999999998);
  const amrex::Real cse1115 = std::pow(cse680, -2);
  const amrex::Real cse1116 = cse1113*cse683;
  const amrex::Real cse1117 = std::pow(cse685, -2);
  const amrex::Real cse1118 = cse1117*cse687*(2.5500935934394033e-12*cse1114*cse1115*state.xn[2] - 6.6690584756579352e-10*cse1116);
  const amrex::Real cse1119 = 1.3659996292638161e-10*cse682;
  const amrex::Real cse1120 = std::pow(cse708, -2);
  const amrex::Real cse1121 = std::pow(cse706, 0.12299999999999994);
  const amrex::Real cse1122 = cse1121*cse143;
  const amrex::Real cse1123 = std::pow(cse706, -0.56899999999999995)*cse707;
  const amrex::Real cse1124 = cse1123*cse143;
  const amrex::Real cse1125 = cse143*cse711*cse85;
  const amrex::Real cse1126 = 1.0/cse714;
  const amrex::Real cse1127 = cse1126*cse147/std::pow(cse715, 2);
  const amrex::Real cse1128 = cse1119*cse134*cse688/std::pow(cse690, 2);
  const amrex::Real cse1129 = cse1126*cse715;
  const amrex::Real cse1130 = cse1126*cse718;
  const amrex::Real cse1131 = cse66*cse765;
  const amrex::Real cse1132 = cse3*cse748;
  const amrex::Real cse1133 = std::pow(state.T, -0.92999999999999994);
  const amrex::Real cse1134 = 100*cse3;
  const amrex::Real cse1135 = -2.2649999999999999e-6*cse1132 + cse753*(cse1134*(8.7000000000000001e-5*cse750 + 0.001281) + cse3*cse751) + state.xn[3]*(0.75*((cse754) ? (
   1.1000000000000001e-13
)
: (
   2.6233044951750002e-11*cse1133
)) + 0.25*((cse754) ? (
   3.3000000000000001e-13
)
: (
   2.7229810044920004e-11*cse1133
)));
  const amrex::Real cse1136 = 5.7927337833400002e-20*cse1086;
  const amrex::Real cse1137 = 0.4375*state.xn[0]/std::pow(state.T, 3.0/4.0) - 170.02500000000001*state.xn[2]/std::pow(state.T, 5.0/4.0);
  const amrex::Real cse1138 = -cse1135;
  const amrex::Real cse1139 = 5.7927337833400002e-20*cse1092*cse764;
  const amrex::Real cse1140 = std::pow(cse728, -3);
  const amrex::Real cse1141 = -cse7*cse733*(0.0040460000000000001*cse1140 + 0.0036709999999999998*cse729) - cse7*cse736*(0.0047800000000000004*cse1140 - 5.5500000000000001e-5*cse729);
  const amrex::Real cse1142 = 2.0187430200000004e-26*cse1087*cse243;
  const amrex::Real cse1143 = cse1062*cse594;
  const amrex::Real cse1144 = 2.5634832e-11*cse1143*state.xn[3];
  const amrex::Real cse1145 = 2*cse1092;
  const amrex::Real cse1146 = l0_co_interp1d_partial_0(state.T);
  const amrex::Real cse1147 = std::pow(cse774, -2);
  const amrex::Real cse1148 = ((cse779) ? (
   -2.872276598720128e-13*cse776*cse85*state.xn[9]*(((5.7445531974402559e-18*cse2*cse776 - gradv < 0) ? (
      0
   )
   : ((5.7445531974402559e-18*cse2*cse776 - gradv == 0) ? (
      1.0/2.0
   )
   : (
      1
   ))))/std::pow(cse777, 2)
)
: (
   0
));
  const amrex::Real cse1149 = -cse1082*cse1148 - llte_co_interp2d_partial_0(state.T, cse783);
  const amrex::Real cse1150 = cse1077*cse1148 + nhalf_co_interp2d_partial_0(state.T, cse783);
  const amrex::Real cse1151 = ((cse862) ? (
   -cse66
)
: (
   0
));
  const amrex::Real cse1152 = -1.8719999999999998e-14*cse1151*cse891 - 3.9261999999999998e-15*cse1151*cse892 - 1.53e-21*cse1151*cse893 + 4.985670872372847e-33*cse894*cse897*((cse862) ? (
   3.7599999999999998*std::pow(state.T, 2.7599999999999998)
)
: (
   0
)) - 1.0953518906603145e-26*cse895*cse897*((cse862) ? (
   -3/std::pow(state.T, 4)
)
: (
   0
)) - 2.9985055107607381e-40*cse895*((cse862) ? (
   2.1000000000000001*std::pow(state.T, 1.1000000000000001)
)
: (
   0
))/std::pow(cse896, 2);
  const amrex::Real cse1153 = ((cse862) ? (
   1
)
: (
   0
));
  const amrex::Real cse1154 = cse1153*cse890;
  const amrex::Real cse1155 = cse1154*cse13*cse899;
  const amrex::Real cse1156 = cse1154*cse872*cse902;
  const amrex::Real cse1157 = cse1154*cse874*cse904;
  const amrex::Real cse1158 = cse1154*cse8;
  const amrex::Real cse1159 = cse1154*cse876*cse905;
  const amrex::Real cse1160 = cse1154*cse866*cse922;
  const amrex::Real cse1161 = cse1154*cse868*cse924;
  const amrex::Real cse1162 = cse1154*cse870*cse900;
  const amrex::Real cse1163 = 65.320000000000007*cse1021;
  const amrex::Real cse1164 = cse3*cse809;
  const amrex::Real cse1165 = 0.00042999999999999999*cse750;
  const amrex::Real cse1166 = cse820*(cse1134*(0.0041900000000000001 - 3.0000000000000001e-5*cse750) - cse3*cse817) - 3.8399999999999998e-11*std::pow(state.T, -1.075)*state.xn[2] + state.xn[3]*(cse822*(cse1134*(0.00025000000000000001*cse750 + 0.00264) + cse3*cse821) + cse823*(cse1134*(cse1165 + 0.00296) + cse1164));
  const amrex::Real cse1167 = cse814*cse827;
  const amrex::Real cse1168 = cse836*(cse1134*(0.0036900000000000001 - 6.0000000000000002e-5*cse750) - cse3*cse833) - 1.2635999999999999e-11*std::pow(state.T, -1.026)*state.xn[2] + state.xn[3]*(cse838*(cse1134*(0.00041000000000000005*cse750 + 0.0020300000000000001) + cse3*cse837) + cse840*(cse1134*(0.00058*cse750 + 0.0023700000000000001) + cse3*cse839));
  const amrex::Real cse1169 = (1.0/15.0)*cse830;
  const amrex::Real cse1170 = cse808*(cse1134*(0.0075500000000000003 - 0.0016000000000000001*cse750) - cse3*cse805) + 1.0000800000000001e-14*std::pow(state.T, -0.073999999999999955)*state.xn[2] + state.xn[3]*(cse810*(cse1134*(cse1165 + 0.0088900000000000003) + cse1164) + cse812*(cse1134*(0.0052500000000000003*cse750 + 0.011979999999999999) + cse3*cse811));
  const amrex::Real cse1171 = cse829*cse843;
  const amrex::Real cse1172 = cse1168*cse844;
  const amrex::Real cse1173 = 65.320000000000007*cse1025*cse66*cse814 + cse1163*cse66*cse844 + cse1166*cse1167 + cse1166*cse845 + cse1168*cse1169 + cse1170*cse1171 + cse1170*cse828 + cse1172*cse827 + 28.360000000000003*cse66*cse830*cse842;
  const amrex::Real cse1174 = 136.68000000000001*cse853;
  const amrex::Real cse1175 = cse66*cse848;
  const amrex::Real cse1176 = cse66*cse847;
  const amrex::Real cse1177 = cse1166*cse848;
  const amrex::Real cse1178 = (3.0/5.0)*cse852;
  const amrex::Real cse1179 = cse1166*cse847;
  const amrex::Real cse1180 = cse827*cse847;
  const amrex::Real cse1181 = cse1163*cse1176 + cse1168*cse1180 + cse1168*cse854 + cse1170*cse845 + cse1170*cse854 + cse1174*cse1175 + cse1174*cse1176 + cse1177*cse1178 + cse1178*cse1179;
  const amrex::Real cse1182 = (1.0/3.0)*cse1036;
  const amrex::Real cse1183 = cse1035*(-cse1168*cse851 - cse1170*cse1182 - cse1170*cse844 - cse1172 - cse1173 - 32.933333333333337*cse1175*cse850 - cse1177 - cse1179 - cse1181);
  const amrex::Real cse1184 = 312.30000000000001*cse66;
  const amrex::Real cse1185 = cse3*cse934;
  const amrex::Real cse1186 = 0.00046000000000000001*cse750;
  const amrex::Real cse1187 = 8.6290000000000003e-7*cse1132;
  const amrex::Real cse1188 = -cse1187*cse973 + cse959*(cse1134*(cse1186 + 0.00231) + cse1185) + cse974*((cse885) ? (
   cse147*(cse147*(cse3*cse965 + cse3*cse966) + cse3*cse967) + cse3*cse968
)
: (
   cse147*(cse147*(cse3*cse969 + cse3*cse970) + cse3*cse971) + cse3*cse972
)) + state.xn[3]*(cse961*(cse1134*(0.00031*cse750 + 0.00263) + cse3*cse960) + cse963*(cse1134*(0.00064000000000000005*cse750 + 0.00071999999999999994) + cse3*cse962));
  const amrex::Real cse1189 = (25.0/3.0)*cse956;
  const amrex::Real cse1190 = -1.4381666666666666e-6*cse1132*cse994 + cse980*(cse1134*(0.00057000000000000009*cse750 + 0.00115) + cse3*cse977) + cse996*((cse885) ? (
   cse147*(cse147*(-cse3*cse986 + cse3*cse987) + cse3*cse988) + cse3*cse989
)
: (
   cse147*(cse147*(cse3*cse990 + cse3*cse991) + cse3*cse992) + cse3*cse993
)) + state.xn[3]*(cse982*(cse1134*(0.00048999999999999998*cse750 - 4.0000000000000003e-5) + cse3*cse981) + cse984*(cse1134*(0.0010199999999999999*cse750 - 0.00085000000000000006) + cse3*cse983));
  const amrex::Real cse1191 = -cse1187*cse951 + cse937*(cse1134*(cse1186 + 0.0022800000000000003) + cse1185) + cse953*((cse885) ? (
   cse147*(cse147*(cse3*cse943 + cse3*cse944) + cse3*cse945) + cse3*cse946
)
: (
   cse147*(cse147*(cse3*cse947 + cse3*cse948) + cse3*cse949) + cse3*cse950
)) + state.xn[3]*(cse939*(cse1134*(0.00038000000000000002*cse750 + 0.0016900000000000001) + cse3*cse938) + cse941*(cse1134*(0.00048000000000000001*cse750 - 0.0001) + cse3*cse940));
  const amrex::Real cse1192 = cse955*cse976;
  const amrex::Real cse1193 = cse975*cse999;
  const amrex::Real cse1194 = cse1001*cse1191;
  const amrex::Real cse1195 = cse1000*cse1188 + cse1002*cse1190 + cse1044*cse1184*cse975 + cse1049*cse1184*cse954 + cse1188*cse1189 + cse1190*cse1193 + cse1191*cse1192 + cse1194*cse999 + 844.16666666666686*cse66*cse956*cse975;
  const amrex::Real cse1196 = 70.859999999999985*cse1010*cse66;
  const amrex::Real cse1197 = cse1005*cse1190;
  const amrex::Real cse1198 = 3*cse1009;
  const amrex::Real cse1199 = cse1004*cse999;
  const amrex::Real cse1200 = cse1004*cse1190;
  const amrex::Real cse1201 = cse1002*cse1188 + cse1004*cse1047*cse1184 + cse1004*cse1196 + cse1005*cse1196 + cse1011*cse1188 + cse1011*cse1191 + cse1191*cse1199 + cse1197*cse1198 + cse1198*cse1200;
  const amrex::Real cse1202 = (5.0/3.0)*cse1058;
  const amrex::Real cse1203 = cse1057*(-cse1001*cse1188 - 64.733333333333348*cse1005*cse1007*cse66 - cse1008*cse1191 - cse1188*cse1202 - cse1194 - cse1195 - cse1197 - cse1200 - cse1201);
  const amrex::Real cse1204 = cse1020*cse173*(5.6705896183685846e-35*state.c_hat*cse10*cse1120*cse142*cse705*d2g*(-4.2149769358107055e-15*cse107*cse1122 - 6.1051722655411154e-13*cse107*cse1124) + 1.8755475162754094e-35*state.c_hat*cse10*cse142*cse709*d2g*std::pow(state.T, -0.85299999999999998) - cse1067*cse1141*cse1142*cse726 - cse1068*cse1141*cse1144*cse799 - 1000*cse1084*cse771*std::pow(cse772, 999)*cse796 - cse1085*(-cse1081*cse1149*cse770 - cse1137*cse785 + cse1146*cse1147*cse788*cse792 + cse1146*cse1147 - cse793*(-cse1081*cse1149*cse774*cse786 - cse1146*cse785*cse786 - cse1150*cse787) - cse794*(cse1074*(cse1073*cse1148 + alpha_co_interp2d_partial_0(state.T, cse783)) + cse1080*(-cse1076*cse1150 + cse1137*cse789))) - cse1093*(-182.41999999999999*cse1131 - cse1135*cse761 - cse1138*cse1145) - 100.0*cse1096*cse426*psigd_coll_interp2d_partial_1(cse10, state.T) - cse1102*(1.6000000000000001e-32*cse700*cse701*cse85 + cse703) - 1.6173435610483583e-5*cse1111*cse1112*cse681 - cse1112*cse1118*cse1119 - 2.0489994438957239e-11*cse1113*cse134*cse681*cse692 + 7.8348995947646122e-14*cse1114*cse1115*cse686*cse687*cse691*state.xn[0]*state.xn[2] - cse1128*(-0.0015792330470357991*cse1111*cse683 + 7.6502807803182096e-12*cse1114*cse1115*cse686*cse687*state.xn[2] - 2.0007175426973804e-9*cse1116*cse688 - 1.3338116951315871e-8*cse1118*cse684) - 5.2835524837844135e-18*cse1131*cse764 - cse1135*cse1136 - cse1137*cse798 - cse1138*cse1139 - 2.1789607199999999e-11*cse135 - 7.1777529600000012e-12*cse578 - 7.1777529600000012e-12*cse579 - cse722*(-6.3340486037177084e-11*cse1125*cse1127 + cse3*cse716) - cse722*(-1.4428815283938862e-12*cse1125*cse1129 + 2.835294809184292e-11*cse1125*cse1130) - state.xn[11]*(cse1038*cse1183 + cse1039*cse1183 + cse1173*cse858 + cse1181*cse859) - state.xn[3]*((cse889) ? (
   0
)
: (
   cse1041*cse898*(-cse1040*cse898*(-cse908*(4.8223900769399082*cse1153*cse8*cse890 + 3.0182298984217999*cse1153*cse876*cse890*cse905 - 3.5529549287336835*cse1155 - 0.3872755400043702*cse1156 - 1.3735579540080118*cse1157) - cse912*(0.0708789387907936*cse1155 + 3.7035619079275177*cse1156 - 4.6974781513675152*cse1157 + 3.6184459289309552*cse1158 - 1.6316107607322889*cse1159) - cse914*(-3.7541549062629067*cse1155 + 2.2094886994529301*cse1156 - 1.528565035159452*cse1157 + 5.0409049417480247*cse1158 + 2.0057552335975877*cse1159) - state.xn[0]*((cse916) ? (
      cse7*cse915*(116.29033200000001*cse1155 + 19.239805*cse1156 + 80.639324000000002*cse1157 + 37.383713*cse1158 + 145.968309*cse1159)
   )
   : ((cse918) ? (
      cse917*(8.2184944748967013*cse1153*cse8*cse890 - 52.189748993977005*cse1155 - 48.951834264235494*cse1156 - 196.44057098336626*cse1157 - 192.38155095558542*cse1159)
   )
   : (
      cse919*(-17.135767342440825*cse1155 + 17.889115159724135*cse1156 - 50.756388852554174*cse1157 + 10.695627721640689*cse1158 + 41.010708268606813*cse1159)
   ))) - state.xn[2]*((cse754) ? (
      cse926*(445.51869310442481*cse1155 + 11324.985706577943*cse1156 + 6766.9891340104432*cse1157 + 38.719649225812766*cse1158 + 2370.6849681533818*cse1159 + 1300.6686834345674*cse1160 + 5869.3152370465659*cse1161 + 11077.448644792414*cse1162)
   )
   : (
      cse927*(4.297112944704045*cse1155 - 101.40241318979159*cse1156 - 43.540996231705549*cse1157 + 3.8689780091986448*cse1158 + 27.91190909651122*cse1159 - 117.34186576967058*cse1160 + 103.76942484394611*cse1161 + 123.18901581604325*cse1162)
   ))) - cse1152*cse930) + cse1152*cse932
)) - state.xn[7]*((cse1017) ? (
   cse1015*cse1195 + cse1016*cse1201 + cse1060*cse1203 + cse1061*cse1203
)
: (
   0
)));
  const amrex::Real cse1205 = 7.1777529600000012e-12*cse198;
  const amrex::Real cse1206 = 7.1777529600000012e-12*cse585;
  const amrex::Real cse1207 = cse1066*cse1068;
  const amrex::Real cse1208 = -cse1207*cse799 + cse35*cse738*cse799;
  const amrex::Real cse1209 = cse789*cse791;
  const amrex::Real cse1210 = cse1085*(-cse1083 - cse794*(cse1072*cse1073*cse1074 - cse1078*cse1209));
  const amrex::Real cse1211 = -cse1088 - cse1089*cse739 - cse1207*cse726;
  const amrex::Real cse1212 = std::pow(state.T, 0.29999999999999999);
  const amrex::Real cse1213 = std::pow(cse710, -2);
  const amrex::Real cse1214 = cse1213*cse2;
  const amrex::Real cse1215 = cse717*cse719;
  const amrex::Real cse1216 = cse1127*cse1215*state.xn[2];
  const amrex::Real cse1217 = 9.0599999999999997e-6*cse71;
  const amrex::Real cse1218 = 1.024e-10*cse815;
  const amrex::Real cse1219 = 2.1600000000000004e-15*cse803;
  const amrex::Real cse1220 = 9.7199999999999998e-11*cse831;
  const amrex::Real cse1221 = cse1021*cse1218 + cse1023*cse1218 + cse1025*cse1219 + 7.1999999999999997e-16*cse1026*cse803 + cse1028*cse1220 + 3.2399999999999999e-11*cse830*cse831;
  const amrex::Real cse1222 = 3.0719999999999998e-10*cse815;
  const amrex::Real cse1223 = cse1021*cse1219 + cse1030*cse1220 + cse1032*cse1222 + cse1033*cse1222 + 6.4799999999999996e-15*cse803*cse853 + 2.9159999999999995e-10*cse831*cse853;
  const amrex::Real cse1224 = cse1035*(-3.6000000000000001e-15*cse1036*cse803 - cse1221 - cse1223 - cse804*cse844 - cse816*cse847 - cse816*cse848 - 1.6199999999999998e-10*cse831*cse850 - cse832*cse844);
  const amrex::Real cse1225 = cse71*cse973;
  const amrex::Real cse1226 = cse71*cse994;
  const amrex::Real cse1227 = 1.4381666666666666e-5*cse1226;
  const amrex::Real cse1228 = cse71*cse951;
  const amrex::Real cse1229 = 8.6289999999999999e-6*cse1225;
  const amrex::Real cse1230 = 8.6289999999999999e-6*cse1228;
  const amrex::Real cse1231 = 1.4381666666666668e-5*cse1043*cse1228 + cse1044*cse1229 + cse1047*cse1227 + cse1048*cse1227 + cse1049*cse1230 + 1.4381666666666668e-5*cse1225*cse956;
  const amrex::Real cse1232 = 5.1774000000000006e-6*cse1010;
  const amrex::Real cse1233 = 8.6289999999999999e-6*cse1226;
  const amrex::Real cse1234 = cse1047*cse1229 + cse1052*cse1230 + cse1054*cse1233 + cse1055*cse1233 + cse1225*cse1232 + cse1228*cse1232;
  const amrex::Real cse1235 = cse71*cse995;
  const amrex::Real cse1236 = 1.7258000000000001e-6*cse1001;
  const amrex::Real cse1237 = cse1057*(-cse1004*cse1235 - cse1005*cse1235 - 2.8763333333333337e-6*cse1007*cse1228 - 2.8763333333333337e-6*cse1058*cse1225 - cse1225*cse1236 - cse1228*cse1236 - cse1231 - cse1234);
  const amrex::Real cse1238 = cse1100*cse1215;
  const amrex::Real cse1239 = 3.7996253111508318e-29*cse1238;
  const amrex::Real cse1240 = cse1103*(4*state.xn[0] + 4*state.xn[1] + 4*state.xn[10] + 4*state.xn[14] + 4*state.xn[15] + 12*state.xn[16] + 8*state.xn[3] + 8*state.xn[4]);
  const amrex::Real cse1241 = cse1096*(200.0*cse1094 + 200.0*cse696);
  const amrex::Real cse1242 = cse1105*cse675;
  const amrex::Real cse1243 = cse1106*cse1108;
  const amrex::Real cse1244 = -1.0*cse1242 + 1.0*cse1243;
  const amrex::Real cse1245 = ((cse743) ? (
   0
)
: ((cse744) ? (
   3*cse1098
)
: ((cse745) ? (
   (8.0/3.0)*cse1098
)
: ((cse746) ? (
   (2.0/3.0)*cse1098
)
: (
   0
)))));
  const amrex::Real cse1246 = -cse756;
  const amrex::Real cse1247 = cse233*cse731 + cse235*cse734;
  const amrex::Real cse1248 = cse1068*(cse1247 - cse732);
  const amrex::Real cse1249 = cse723*cse739;
  const amrex::Real cse1250 = cse1249*cse56 - 2*cse35*cse726*cse738;
  const amrex::Real cse1251 = ((cse779) ? (
   0
)
: (
   cse1070*cse782
));
  const amrex::Real cse1252 = cse1077*cse1251;
  const amrex::Real cse1253 = -cse1081*cse1082*cse1251*cse770 + cse793*(cse1081*cse1082*cse1251*cse774*cse786 - cse1252*cse787);
  const amrex::Real cse1254 = cse841*cse844;
  const amrex::Real cse1255 = cse1167*cse824 + cse1169*cse841 + cse1171*cse813 + cse1254*cse827 + cse813*cse828 + cse824*cse845;
  const amrex::Real cse1256 = cse824*cse848;
  const amrex::Real cse1257 = cse824*cse847;
  const amrex::Real cse1258 = cse1178*cse1256 + cse1178*cse1257 + cse1180*cse841 + cse813*cse845 + cse813*cse854 + cse841*cse854;
  const amrex::Real cse1259 = cse1035*(-cse1182*cse813 - cse1254 - cse1255 - cse1256 - cse1257 - cse1258 - cse813*cse844 - cse841*cse851);
  const amrex::Real cse1260 = cse1001*cse942;
  const amrex::Real cse1261 = cse1000*cse964 + cse1002*cse985 + cse1189*cse964 + cse1192*cse942 + cse1193*cse985 + cse1260*cse999;
  const amrex::Real cse1262 = cse1005*cse985;
  const amrex::Real cse1263 = cse1004*cse985;
  const amrex::Real cse1264 = cse1002*cse964 + cse1011*cse942 + cse1011*cse964 + cse1198*cse1262 + cse1198*cse1263 + cse1199*cse942;
  const amrex::Real cse1265 = cse1057*(-cse1001*cse964 - cse1008*cse942 - cse1202*cse964 - cse1260 - cse1261 - cse1262 - cse1263 - cse1264);
  const amrex::Real cse1266 = cse705*cse709;
  const amrex::Real cse1267 = cse1266*d2g;
  const amrex::Real cse1268 = cse1267*cse143;
  const amrex::Real cse1269 = 4.22974728e-11*cse10*cse106*cse183*cse674*cse676;
  const amrex::Real cse1270 = 3.2043540000000001e-12*crate*state.xn[3];
  const amrex::Real cse1271 = cse1068*cse1247;
  const amrex::Real cse1272 = cse10*cse1142;
  const amrex::Real cse1273 = cse1020*state.xn[3]*((cse889) ? (
   0
)
: (
   cse1042*cse913
));
  const amrex::Real cse1274 = ((cse779) ? (
   cse778
)
: (
   0
));
  const amrex::Real cse1275 = cse1077*cse1274;
  const amrex::Real cse1276 = 5.6705896183685846e-35*cse143;
  const amrex::Real cse1277 = -cse1020*(-cse1097 + cse1099*cse1270 - cse1101 - cse1104 + cse1109*cse1269 + cse1144*cse1208 - cse1205 - cse1206 - cse1210 + cse1211*cse1272 + cse1267*cse1276 + 3.0000000000000002e-15*cse243*cse741);
  const amrex::Real cse1278 = cse1068*(3*cse1064 + 3*cse1065);
  const amrex::Real cse1279 = ((cse779) ? (
   0
)
: (
   2635052181.6465044*cse1071
));
  const amrex::Real cse1280 = cse1077*cse1279;
  const amrex::Real cse1281 = state.c_hat*cse1266*cse426;
  const amrex::Real cse1282 = state.c_hat*cse1102*cse2*cse701/std::pow(cse699, 5.0/6.0);
  const amrex::Real cse1283 = state.c_hat*cse712;
  const amrex::Real cse1284 = cse1216*cse1283*cse426;
  const amrex::Real cse1285 = state.c_hat*cse108;
  const amrex::Real cse1286 = cse1121*cse1285;
  const amrex::Real cse1287 = cse1123*cse1285;
  const amrex::Real cse1288 = cse1120*cse1276*cse426*cse705;
  const amrex::Real cse1289 = cse1129*cse1283;
  const amrex::Real cse1290 = cse1130*cse1283;
  const amrex::Real cse1291 = cse1143*cse249*cse349*cse739;
  const amrex::Real cse1292 = Zd*cse305;
  const amrex::Real cse1293 = 1.1627326161928671e-21*cse1292;
  const amrex::Real cse1294 = -cse1293;
  const amrex::Real cse1295 = -2.3254652323857341e-21*cse1292;
  const amrex::Real cse1296 = 1.1627326161928671e-21*Zd;
  const amrex::Real cse1297 = state.c_hat*cse10;
  const amrex::Real cse1298 = -cse1296*cse1297 - cse348 - cse545;
  const amrex::Real cse1299 = state.c_hat*state.rn[2*0+1];
  const amrex::Real cse1300 = cse1296*cse1299;
  const amrex::Real cse1301 = -cse1300;
  const amrex::Real cse1302 = Zd*cse1299;
  const amrex::Real cse1303 = -2.3254652323857341e-21*cse1302;
  const amrex::Real cse1304 = Zd*cse247;
  const amrex::Real cse1305 = 1.4208709462150382e-21*cse1304;
  const amrex::Real cse1306 = -cse1305;
  const amrex::Real cse1307 = 2.8417418924300764e-21*cse1304;
  const amrex::Real cse1308 = 5.3576915465847796e-20*cse249;
  const amrex::Real cse1309 = 1.4208709462150382e-21*Zd;
  const amrex::Real cse1310 = -cse1297*cse1309 - cse1308*cse349 - cse353 - cse555;
  const amrex::Real cse1311 = state.c_hat*state.rn[2*1+1];
  const amrex::Real cse1312 = cse1309*cse1311;
  const amrex::Real cse1313 = -cse1312;
  const amrex::Real cse1314 = Zd*cse1311;
  const amrex::Real cse1315 = 2.8417418924300764e-21*cse1314;
  const amrex::Real cse1316 = Zd*cse251;
  const amrex::Real cse1317 = 1.887632855452075e-21*cse1316;
  const amrex::Real cse1318 = -cse1317;
  const amrex::Real cse1319 = 3.7752657109041501e-21*cse1316;
  const amrex::Real cse1320 = 1.8804775722560676e-17*cse249;
  const amrex::Real cse1321 = 1.887632855452075e-21*Zd;
  const amrex::Real cse1322 = -cse1297*cse1321 - cse1320*cse349 - cse354 - cse559 - cse563 - cse659;
  const amrex::Real cse1323 = state.c_hat*state.rn[2*2+1];
  const amrex::Real cse1324 = cse1321*cse1323;
  const amrex::Real cse1325 = -cse1324;
  const amrex::Real cse1326 = Zd*cse1323;
  const amrex::Real cse1327 = 3.7752657109041501e-21*cse1326;
  const amrex::Real cse1328 = Zd*cse253;
  const amrex::Real cse1329 = 2.2658762392657621e-21*cse1328;
  const amrex::Real cse1330 = -cse1329;
  const amrex::Real cse1331 = 4.5317524785315243e-21*cse1328;
  const amrex::Real cse1332 = 3.5957426289008978e-17*cse249;
  const amrex::Real cse1333 = 2.2658762392657621e-21*Zd;
  const amrex::Real cse1334 = -cse1297*cse1333 - cse1332*cse349 - cse357 - cse566 - cse569 - cse660;
  const amrex::Real cse1335 = state.c_hat*state.rn[2*3+1];
  const amrex::Real cse1336 = cse1333*cse1335;
  const amrex::Real cse1337 = -cse1336;
  const amrex::Real cse1338 = Zd*cse1335;
  const amrex::Real cse1339 = 4.5317524785315243e-21*cse1338;
  jac(1, 1) = -cse1 - cse10*cse56 - cse174*cse65 - cse188 + 2*cse29*state.xn[3] + 2*cse30*cse55 - cse57 + 2*cse60*state.xn[0]*state.xn[3];
  jac(1, 2) = -cse174*cse189 + cse194 + cse202;
  jac(1, 3) = -cse174*cse209 + cse206 + cse208*state.xn[16] + cse211*cse212 + cse217 + cse223;
  jac(1, 4) = 3.2000000000000001e-9*cse162*state.xn[17] - cse174*cse257 + cse201*cse246 - cse212*cse258 + cse225 + cse227 + cse229 + 2*cse230 + 2*cse240 + cse241*cse242 - cse244 + cse247*cse250 + cse251*cse252 + cse253*cse254 + cse256*state.xn[8] + cse262 + cse267;
  jac(1, 5) = -cse174*cse274 - cse244 + 2*cse271*cse30 + 2*cse272*state.xn[0]*state.xn[3] + cse273 - cse279;
  jac(1, 6) = -cse174*cse280;
  jac(1, 7) = -cse174*cse281 + cse286;
  jac(1, 8) = -cse174*cse289 + cse288;
  jac(1, 9) = -cse174*cse290 + cse256*state.xn[3];
  jac(1, 10) = -cse174*cse292 - cse291;
  jac(1, 11) = -cse174*cse293 + cse202 + cse295;
  jac(1, 12) = -cse174*cse299 + cse297 + cse298 + cse302;
  jac(1, 13) = -cse174*cse303;
  jac(1, 14) = -cse174*cse304;
  jac(1, 15) = -cse174*cse313 + cse306 + cse307 + cse308 + cse309 + cse310 + cse312 + cse314;
  jac(1, 16) = -cse174*cse315 + cse314 + cse323;
  jac(1, 17) = -cse174*cse330 + cse201*cse329 + cse208*state.xn[2] - 1.8000000000000002e-14*cse243 + 2*cse328 + cse335;
  jac(1, 18) = 3.2000000000000001e-9*cse162*state.xn[3] - cse174*cse339 - cse336 + cse338;
  jac(1, 19) = cse174;
  jac(1, 20) = cse343 + cse348;
  jac(1, 22) = cse250*cse349 + cse350 + cse353;
  jac(1, 24) = cse252*cse349 + cse354 + cse355 + cse356;
  jac(1, 26) = cse254*cse349 + cse357 + cse358 + cse359;
  jac(2, 1) = cse188 - cse361*cse65;
  jac(2, 2) = -cse189*cse361 - cse194;
  jac(2, 3) = -cse209*cse361 - cse222;
  jac(2, 4) = cse225 - cse257*cse361 + cse265 + cse275;
  jac(2, 5) = -cse274*cse361 + cse279;
  jac(2, 6) = -cse280*cse361;
  jac(2, 7) = -cse281*cse361 + cse282;
  jac(2, 8) = -cse289*cse361;
  jac(2, 9) = -cse290*cse361;
  jac(2, 10) = -cse292*cse361;
  jac(2, 11) = -cse192 - cse293*cse361;
  jac(2, 12) = -cse299*cse361 - cse300;
  jac(2, 13) = -cse303*cse361;
  jac(2, 14) = -cse304*cse361;
  jac(2, 15) = -cse192 - cse313*cse361;
  jac(2, 16) = -cse192 - cse315*cse361;
  jac(2, 17) = -cse330*cse361 - cse334;
  jac(2, 18) = cse336 - cse339*cse361;
  jac(2, 19) = cse361;
  jac(2, 20) = -cse343;
  jac(2, 22) = -cse350;
  jac(2, 24) = -cse356;
  jac(2, 26) = -cse359;
  jac(3, 1) = cse178 + cse180 + cse184 - cse187 - cse364 - cse456*cse65 - cse471;
  jac(3, 2) = -cse189*cse456 - cse193 - cse473;
  jac(3, 3) = -cse209*cse456 - cse223 - cse475 - cse478 - cse481 - cse482 - cse488;
  jac(3, 4) = -cse257*cse456 - cse267 + cse489 + cse490 - cse494 - cse500 - cse505 - cse506 - cse508;
  jac(3, 5) = -cse274*cse456 + cse278 - cse493 - cse498 - cse508 - cse510;
  jac(3, 6) = -cse280*cse456 + cse511;
  jac(3, 7) = -cse281*cse456 - cse512;
  jac(3, 8) = -cse289*cse456 - cse514 + cse520;
  jac(3, 9) = -cse290*cse456 - cse523;
  jac(3, 10) = cse291 - cse292*cse456;
  jac(3, 11) = -cse293*cse456 - cse295 - cse473;
  jac(3, 12) = -cse299*cse456 - cse525;
  jac(3, 13) = -cse303*cse456 + cse530;
  jac(3, 14) = -cse304*cse456 - cse531;
  jac(3, 15) = -cse313*cse456 - cse532;
  jac(3, 16) = -cse315*cse456 - cse532;
  jac(3, 17) = -cse330*cse456 - cse335 - cse533 - cse534 - cse535 - cse536 - cse538 - cse541 - cse542;
  jac(3, 18) = -cse339*cse456 - cse543;
  jac(3, 19) = cse456;
  jac(3, 20) = -cse343 + cse545 - cse548 - cse551 - cse554;
  jac(3, 22) = -cse350 + cse555 - cse556 - cse557 - cse558;
  jac(3, 24) = -cse356 + cse559 - cse560 - cse561 - cse565;
  jac(3, 26) = -cse359 + cse566 - cse567 - cse568 - cse571;
  jac(4, 1) = cse1 + 3.0000000000000002e-15*cse10*d2g + cse177 - cse363 - cse572 - cse573 - cse575 - cse576 - cse584*cse65;
  jac(4, 2) = -cse189*cse584 - cse586;
  jac(4, 3) = cse206 - cse209*cse584 + cse587*state.xn[16] + cse588 + cse589;
  jac(4, 4) = -cse199 - cse230 - cse240 - cse241*cse9 - cse257*cse584 - cse496 - cse590 - cse594 - cse596 - cse597 - cse598 - cse599;
  jac(4, 5) = -cse199 - cse273 - cse274*cse584 + cse276 - cse509 - cse600 - cse601;
  jac(4, 6) = -cse280*cse584;
  jac(4, 7) = -cse281*cse584 - cse282 - cse602;
  jac(4, 8) = -cse289*cse584 + cse514;
  jac(4, 9) = -cse290*cse584 - cse521;
  jac(4, 10) = -cse292*cse584 + cse603;
  jac(4, 11) = -cse293*cse584 - cse586;
  jac(4, 12) = -cse299*cse584 + cse525;
  jac(4, 13) = -cse303*cse584;
  jac(4, 14) = -cse304*cse584;
  jac(4, 15) = -cse312 - cse313*cse584 - cse586;
  jac(4, 16) = -cse315*cse584 - cse586;
  jac(4, 17) = -cse328 - cse330*cse584 + cse538 + cse587*state.xn[2] + cse604 - cse605 + cse607;
  jac(4, 18) = -cse339*cse584 - cse543;
  jac(4, 19) = cse584;
  jac(4, 22) = -cse608;
  jac(4, 24) = -cse609;
  jac(4, 26) = -cse610;
  jac(5, 1) = -cse177 - cse364 - cse611*cse65;
  jac(5, 2) = -cse189*cse611 + cse472;
  jac(5, 3) = -cse209*cse611;
  jac(5, 4) = -cse227 - cse257*cse611 + cse595 + cse597;
  jac(5, 5) = -cse273 - cse274*cse611 - cse276 - cse510;
  jac(5, 6) = -cse280*cse611;
  jac(5, 7) = -cse281*cse611 + cse602;
  jac(5, 8) = -cse289*cse611;
  jac(5, 9) = -cse290*cse611;
  jac(5, 10) = -cse292*cse611;
  jac(5, 11) = -cse293*cse611 + cse472;
  jac(5, 12) = -cse299*cse611;
  jac(5, 13) = -cse303*cse611;
  jac(5, 14) = -cse304*cse611;
  jac(5, 15) = -cse313*cse611 + cse472;
  jac(5, 16) = -cse315*cse611 + cse472;
  jac(5, 17) = -cse330*cse611 + cse537;
  jac(5, 18) = -cse339*cse611;
  jac(5, 19) = cse611;
  jac(6, 1) = cse459 - cse614*cse65;
  jac(6, 2) = -cse189*cse614 + cse459;
  jac(6, 3) = -cse209*cse614 + cse475;
  jac(6, 4) = -cse257*cse614 + cse615;
  jac(6, 5) = -cse274*cse614 + cse507;
  jac(6, 6) = -cse280*cse614 - cse511;
  jac(6, 7) = -cse281*cse614 + cse617;
  jac(6, 8) = -cse289*cse614;
  jac(6, 9) = -cse290*cse614;
  jac(6, 10) = -cse292*cse614 + cse618;
  jac(6, 11) = -cse293*cse614 + cse459;
  jac(6, 12) = -cse299*cse614;
  jac(6, 13) = -cse303*cse614;
  jac(6, 14) = -cse304*cse614;
  jac(6, 15) = -cse313*cse614 + cse459;
  jac(6, 16) = -cse315*cse614 + cse619;
  jac(6, 17) = -cse330*cse614 + cse533;
  jac(6, 18) = -cse339*cse614;
  jac(6, 19) = cse614;
  jac(6, 20) = cse551;
  jac(6, 22) = cse557;
  jac(6, 24) = cse561;
  jac(6, 26) = cse568;
  jac(7, 1) = -cse459 - cse620*cse65;
  jac(7, 2) = -cse189*cse620 - cse459;
  jac(7, 3) = -cse209*cse620 - cse475;
  jac(7, 4) = -cse257*cse620 - cse615;
  jac(7, 5) = -cse274*cse620 - cse507;
  jac(7, 6) = -cse280*cse620 + cse511;
  jac(7, 7) = -cse281*cse620 - cse617;
  jac(7, 8) = -cse289*cse620;
  jac(7, 9) = -cse290*cse620;
  jac(7, 10) = -cse292*cse620 - cse618;
  jac(7, 11) = -cse293*cse620 - cse459;
  jac(7, 12) = -cse299*cse620;
  jac(7, 13) = -cse303*cse620;
  jac(7, 14) = -cse304*cse620;
  jac(7, 15) = -cse313*cse620 - cse459;
  jac(7, 16) = -cse315*cse620 - cse619;
  jac(7, 17) = -cse330*cse620 - cse533;
  jac(7, 18) = -cse339*cse620;
  jac(7, 19) = cse620;
  jac(7, 20) = -cse551;
  jac(7, 22) = -cse557;
  jac(7, 24) = -cse561;
  jac(7, 26) = -cse568;
  jac(8, 1) = cse1 - cse622*cse65 + cse626;
  jac(8, 2) = -cse189*cse622 + cse626;
  jac(8, 3) = -cse209*cse622 + cse478;
  jac(8, 4) = -cse257*cse622 + cse499 + cse627*state.xn[8] + cse632;
  jac(8, 5) = -cse274*cse622 + cse498 + cse631;
  jac(8, 6) = -cse280*cse622;
  jac(8, 7) = -cse281*cse622;
  jac(8, 8) = -cse288 - cse289*cse622 - cse514 - cse520;
  jac(8, 9) = -cse290*cse622 + cse522 + cse627*state.xn[3];
  jac(8, 10) = -cse292*cse622 + cse638;
  jac(8, 11) = -cse293*cse622 + cse626;
  jac(8, 12) = -cse299*cse622;
  jac(8, 13) = -cse303*cse622;
  jac(8, 14) = -cse304*cse622;
  jac(8, 15) = -cse313*cse622 + cse626 + cse639;
  jac(8, 16) = -cse315*cse622 + cse462 + cse465 - cse640;
  jac(8, 17) = -cse330*cse622 - cse534 + cse539 + cse540 - cse641;
  jac(8, 18) = -cse339*cse622;
  jac(8, 19) = cse622;
  jac(8, 20) = cse345 + cse554;
  jac(8, 22) = cse351 + cse558;
  jac(8, 24) = cse354 + cse565 + cse643;
  jac(8, 26) = cse357 + cse571 + cse644;
  jac(9, 1) = -cse466 - cse647*cse65;
  jac(9, 2) = -cse189*cse647 - cse466;
  jac(9, 3) = -cse209*cse647 - cse478;
  jac(9, 4) = -cse257*cse647 - cse500;
  jac(9, 5) = -cse274*cse647 - cse498;
  jac(9, 6) = -cse280*cse647;
  jac(9, 7) = -cse281*cse647 + cse616;
  jac(9, 8) = -cse289*cse647 + cse520;
  jac(9, 9) = -cse290*cse647 - cse523 - cse649;
  jac(9, 10) = -cse292*cse647 + cse618;
  jac(9, 11) = -cse293*cse647 - cse466;
  jac(9, 12) = -cse299*cse647;
  jac(9, 13) = -cse303*cse647;
  jac(9, 14) = -cse304*cse647;
  jac(9, 15) = -cse313*cse647 - cse466;
  jac(9, 16) = -cse315*cse647 - cse466 - cse650;
  jac(9, 17) = -cse330*cse647 - cse541;
  jac(9, 18) = -cse339*cse647;
  jac(9, 19) = cse647;
  jac(9, 20) = -cse554;
  jac(9, 22) = -cse558;
  jac(9, 24) = cse563 - cse564;
  jac(9, 26) = cse569 - cse570;
  jac(10, 1) = cse624 - cse65*cse651;
  jac(10, 2) = -cse189*cse651 + cse624;
  jac(10, 3) = -cse209*cse651 + cse219;
  jac(10, 4) = -cse257*cse651 - cse632;
  jac(10, 5) = -cse274*cse651 + cse630;
  jac(10, 6) = -cse280*cse651;
  jac(10, 7) = -cse281*cse651 - cse616;
  jac(10, 8) = cse288 - cse289*cse651;
  jac(10, 9) = -cse290*cse651;
  jac(10, 10) = -cse292*cse651 - cse652 - cse653;
  jac(10, 11) = -cse293*cse651 + cse294 + cse624;
  jac(10, 12) = -cse299*cse651 + cse301;
  jac(10, 13) = -cse303*cse651;
  jac(10, 14) = -cse304*cse651;
  jac(10, 15) = -cse313*cse651 + cse654;
  jac(10, 16) = -cse315*cse651 + cse640;
  jac(10, 17) = -cse330*cse651 - cse606 + cse641;
  jac(10, 18) = -cse339*cse651;
  jac(10, 19) = cse651;
  jac(10, 24) = -cse643;
  jac(10, 26) = -cse644;
  jac(11, 1) = -cse65*cse655;
  jac(11, 2) = -cse189*cse655;
  jac(11, 3) = -cse209*cse655 - cse219;
  jac(11, 4) = -cse257*cse655;
  jac(11, 5) = -cse274*cse655;
  jac(11, 6) = -cse280*cse655;
  jac(11, 7) = -cse281*cse655;
  jac(11, 8) = -cse289*cse655;
  jac(11, 9) = -cse290*cse655 + cse649;
  jac(11, 10) = -cse292*cse655 + cse652;
  jac(11, 11) = -cse293*cse655 - cse294;
  jac(11, 12) = -cse299*cse655;
  jac(11, 13) = -cse303*cse655;
  jac(11, 14) = -cse304*cse655;
  jac(11, 15) = -cse313*cse655;
  jac(11, 16) = -cse315*cse655 + cse650;
  jac(11, 17) = -cse330*cse655 + cse606;
  jac(11, 18) = -cse339*cse655;
  jac(11, 19) = cse655;
  jac(12, 1) = cse176 - cse624 - cse65*cse658;
  jac(12, 2) = -cse104 - cse189*cse658 - cse624;
  jac(12, 3) = -cse209*cse658 - cse588;
  jac(12, 4) = -cse257*cse658 + cse260 - cse261 - cse504 + cse629 - cse630;
  jac(12, 5) = -cse274*cse658 - cse630;
  jac(12, 6) = -cse280*cse658;
  jac(12, 7) = -cse281*cse658 + cse616;
  jac(12, 8) = -cse289*cse658;
  jac(12, 9) = -cse290*cse658;
  jac(12, 10) = -cse292*cse658 + cse653;
  jac(12, 11) = -cse293*cse658 - cse624;
  jac(12, 12) = cse298 - cse299*cse658 - cse302 - cse524;
  jac(12, 13) = -cse303*cse658;
  jac(12, 14) = -cse304*cse658;
  jac(12, 15) = -cse313*cse658 - cse654;
  jac(12, 16) = -cse315*cse658 + cse322 + cse625;
  jac(12, 17) = -cse330*cse658 - cse535 - cse641;
  jac(12, 18) = cse336 + cse337 - cse339*cse658;
  jac(12, 19) = cse658;
  jac(12, 20) = cse347;
  jac(12, 22) = cse352;
  jac(12, 24) = cse659;
  jac(12, 26) = cse660;
  jac(13, 1) = cse470 - cse65*cse662;
  jac(13, 2) = -cse189*cse662 + cse470;
  jac(13, 3) = -cse209*cse662 + cse481;
  jac(13, 4) = -cse257*cse662 + cse494;
  jac(13, 5) = -cse274*cse662 + cse493;
  jac(13, 6) = -cse280*cse662;
  jac(13, 7) = -cse281*cse662;
  jac(13, 8) = -cse289*cse662;
  jac(13, 9) = -cse290*cse662;
  jac(13, 10) = -cse292*cse662;
  jac(13, 11) = -cse293*cse662 + cse470;
  jac(13, 12) = -cse299*cse662;
  jac(13, 13) = -cse303*cse662 - cse530;
  jac(13, 14) = -cse304*cse662 + cse531;
  jac(13, 15) = -cse313*cse662 + cse470;
  jac(13, 16) = -cse315*cse662 + cse470;
  jac(13, 17) = -cse330*cse662 + cse542;
  jac(13, 18) = -cse339*cse662;
  jac(13, 19) = cse662;
  jac(13, 20) = -cse663;
  jac(13, 22) = -cse664;
  jac(13, 24) = -cse665;
  jac(13, 26) = -cse666;
  jac(14, 1) = -cse470 - cse65*cse667;
  jac(14, 2) = -cse189*cse667 - cse470;
  jac(14, 3) = -cse209*cse667 - cse481;
  jac(14, 4) = -cse257*cse667 - cse494;
  jac(14, 5) = -cse274*cse667 - cse493;
  jac(14, 6) = -cse280*cse667;
  jac(14, 7) = -cse281*cse667;
  jac(14, 8) = -cse289*cse667;
  jac(14, 9) = -cse290*cse667;
  jac(14, 10) = -cse292*cse667;
  jac(14, 11) = -cse293*cse667 - cse470;
  jac(14, 12) = -cse299*cse667;
  jac(14, 13) = -cse303*cse667 + cse530;
  jac(14, 14) = -cse304*cse667 - cse531;
  jac(14, 15) = -cse313*cse667 - cse470;
  jac(14, 16) = -cse315*cse667 - cse470;
  jac(14, 17) = -cse330*cse667 - cse542;
  jac(14, 18) = -cse339*cse667;
  jac(14, 19) = cse667;
  jac(14, 20) = cse663;
  jac(14, 22) = cse664;
  jac(14, 24) = cse665;
  jac(14, 26) = cse666;
  jac(15, 1) = -cse1 - cse65*cse668;
  jac(15, 2) = -cse189*cse668;
  jac(15, 3) = -cse209*cse668;
  jac(15, 4) = -cse257*cse668 + cse669*state.xn[8];
  jac(15, 5) = -cse274*cse668;
  jac(15, 6) = -cse280*cse668;
  jac(15, 7) = -cse281*cse668;
  jac(15, 8) = -cse289*cse668 + cse514;
  jac(15, 9) = -cse290*cse668 + cse669*state.xn[3];
  jac(15, 10) = -cse292*cse668;
  jac(15, 11) = -cse293*cse668;
  jac(15, 12) = -cse299*cse668 - cse301;
  jac(15, 13) = -cse303*cse668;
  jac(15, 14) = -cse304*cse668;
  jac(15, 15) = -cse310 - cse313*cse668 - cse639;
  jac(15, 16) = -cse315*cse668;
  jac(15, 17) = -cse330*cse668 + cse534;
  jac(15, 18) = -cse339*cse668;
  jac(15, 19) = cse668;
  jac(15, 20) = -cse345;
  jac(15, 22) = -cse351;
  jac(15, 24) = -cse354;
  jac(15, 26) = -cse357;
  jac(16, 1) = -cse65*cse670;
  jac(16, 2) = -cse189*cse670;
  jac(16, 3) = -cse209*cse670 + cse485 - cse486 - cse589;
  jac(16, 4) = -cse257*cse670 + cse505;
  jac(16, 5) = -cse274*cse670;
  jac(16, 6) = -cse280*cse670;
  jac(16, 7) = -cse281*cse670 - cse285;
  jac(16, 8) = -cse288 - cse289*cse670;
  jac(16, 9) = -cse290*cse670 - cse649;
  jac(16, 10) = -cse292*cse670;
  jac(16, 11) = -cse293*cse670;
  jac(16, 12) = -cse298 - cse299*cse670 + cse524;
  jac(16, 13) = -cse303*cse670;
  jac(16, 14) = -cse304*cse670;
  jac(16, 15) = -cse313*cse670;
  jac(16, 16) = -cse315*cse670 - cse323 - cse650;
  jac(16, 17) = -cse330*cse670 + cse535;
  jac(16, 18) = cse338 - cse339*cse670;
  jac(16, 19) = cse670;
  jac(16, 20) = -cse347;
  jac(16, 22) = -cse352;
  jac(16, 24) = -cse355;
  jac(16, 26) = -cse358;
  jac(17, 1) = -cse65*cse671;
  jac(17, 2) = -cse189*cse671;
  jac(17, 3) = -cse209*cse671 - cse488;
  jac(17, 4) = -cse257*cse671 + cse598;
  jac(17, 5) = cse273 - cse274*cse671;
  jac(17, 6) = -cse280*cse671;
  jac(17, 7) = -cse281*cse671;
  jac(17, 8) = -cse289*cse671 - cse514;
  jac(17, 9) = -cse290*cse671;
  jac(17, 10) = -cse292*cse671 - cse603;
  jac(17, 11) = -cse293*cse671;
  jac(17, 12) = -cse299*cse671 - cse525;
  jac(17, 13) = -cse303*cse671;
  jac(17, 14) = -cse304*cse671;
  jac(17, 15) = -cse313*cse671;
  jac(17, 16) = -cse315*cse671;
  jac(17, 17) = -cse330*cse671 - cse536 - cse607;
  jac(17, 18) = -cse339*cse671;
  jac(17, 19) = cse671;
  jac(18, 1) = -cse176 - cse65*cse672;
  jac(18, 2) = cse104 - cse189*cse672;
  jac(18, 3) = -cse209*cse672 - cse482;
  jac(18, 4) = -cse257*cse672 - cse599;
  jac(18, 5) = -cse274*cse672;
  jac(18, 6) = -cse280*cse672;
  jac(18, 7) = -cse281*cse672 + cse285;
  jac(18, 8) = -cse289*cse672;
  jac(18, 9) = -cse290*cse672;
  jac(18, 10) = -cse292*cse672;
  jac(18, 11) = -cse293*cse672;
  jac(18, 12) = -cse299*cse672 + cse300;
  jac(18, 13) = -cse303*cse672;
  jac(18, 14) = -cse304*cse672;
  jac(18, 15) = -cse313*cse672;
  jac(18, 16) = -cse315*cse672 + cse316;
  jac(18, 17) = -cse330*cse672;
  jac(18, 18) = -cse336 - cse339*cse672 - cse543;
  jac(18, 19) = cse672;
  jac(19, 1) = -1.6737729999999998e-24*cse1019 + cse1020*(1.602177e-12*crate*cse677 + 2.0187430200000004e-26*cse10*cse1087*d2g*state.xn[0]*(-cse1069*cse726 - cse1088 - cse739*(cse1089 - cse725/std::pow(state.xn[0], 2))) + 3.0000000000000002e-15*cse10*cse741*d2g + 2.5634832e-11*cse1062*cse594*state.xn[3]*(cse1063 - cse1069*cse799) - cse1085*(-cse1083 - cse769*cse785 - cse794*(cse1072*cse1075 + cse1080*(-cse1076*cse1078 + cse769*cse789))) - 4.39089220777172e-29*cse1086*cse752 + 4.39089220777172e-29*cse1091*cse752*cse757*cse759*cse763*state.xn[8] - cse1093*(cse1090*cse1092 - cse1090*cse760) - cse1110 - 7.1777529600000012e-12*cse575 - 7.1777529600000012e-12*cse576 - cse678 - cse684*cse693 - cse769*cse798 - cse802 - state.xn[11]*(cse1029*cse858 + cse1034*cse859 + cse1037*cse1038 + cse1037*cse1039) - state.xn[3]*((cse889) ? (
   0
)
: (
   cse1042*cse920
)) - state.xn[7]*((cse1017) ? (
   cse1015*cse1051 + cse1016*cse1056 + cse1059*cse1060 + cse1059*cse1061
)
: (
   0
))) - cse1204*cse65;
  jac(19, 2) = -1.6728620616289998e-24*cse1019 + cse1020*(2.0187430200000004e-26*cse10*cse1087*cse1211*d2g*state.xn[0] + 2.5634832e-11*cse1062*cse1208*cse594*state.xn[3] - cse1110 - cse1205 - cse1206 - cse1210 - state.xn[3]*((cse889) ? (
   0
)
: (
   cse1042*cse911
))) - cse1204*cse189;
  jac(19, 3) = -9.1093837099999994e-28*cse1019 + cse1020*(5.6705896183685846e-35*state.c_hat*cse10*cse1120*cse142*cse705*d2g*(8.429953871621411e-15*cse1122*cse210 + 1.2210344531082231e-12*cse1124*cse210) + 4.22974728e-11*crate*cse10*cse106*cse674*cse676*state.xn[0]*(0.5*cse11*cse675 - cse1105*cse1107) - cse1085*(-cse1079*cse768*cse794 - cse768*cse785) - 2.6241084038530201e-25*cse1086*cse71 + 2.6241084038530201e-25*cse1091*cse71*cse757*cse759*cse763*state.xn[8] - cse1093*(cse1092*cse1217 - cse1217*cse760) + 6.0732876035249664e-19*cse1115*cse1117*cse1212*cse687*cse691*state.xn[0]*state.xn[2] - cse1128*(5.9301787935659931e-17*cse1115*cse1117*cse1212*cse687*state.xn[2] - cse682*cse689) - 2.4067011396745388e-39*cse1214*cse1216*cse143*cse426 - 2.1789607199999999e-11*cse220 - cse694*state.xn[0] - cse721 - cse722*(2.8857630567877724e-12*state.c_hat*cse1126*cse1213*cse142*cse2*cse715 - cse1130*cse1214*cse713) - cse768*cse798 - state.xn[11]*(cse1038*cse1224 + cse1039*cse1224 + cse1221*cse858 + cse1223*cse859) - state.xn[3]*((cse889) ? (
   0
)
: (
   cse1042*cse928
)) - state.xn[7]*((cse1017) ? (
   cse1015*cse1231 + cse1016*cse1234 + cse1060*cse1237 + cse1061*cse1237
)
: (
   0
))) - cse1204*cse209;
  jac(19, 4) = -3.3475459999999996e-24*cse1019 + cse1020*(1.1341179236737169e-34*state.c_hat*cse142*cse705*cse709*d2g + 4.22974728e-11*crate*cse10*cse106*cse1244*cse674*cse676*state.xn[0] + 3.2043540000000001e-12*crate*cse1245*state.xn[3] + 3.2043540000000001e-12*crate*cse747 + 2.0187430200000004e-26*cse10*cse1087*d2g*state.xn[0]*(-cse1248*cse726 - cse1250) + 2.5634832e-11*cse1062*cse594*state.xn[3]*(2*cse1063 - cse1248*cse799) - cse1085*(-cse1253 - cse785 - cse794*(cse1075*cse1251 + cse1080*(-cse1076*cse1252 + cse789))) - cse1093*(-cse1145*cse1246 - cse756*cse761) - cse1136*cse756 - cse1139*cse1246 - cse1239 - cse1240 - cse1241 - 7.1777529600000012e-12*cse230 - 7.1777529600000012e-12*cse240 - 1.4355505920000002e-11*cse241*state.xn[3] - 7.1777529600000012e-12*cse590 + cse594*cse801 + 6.0000000000000005e-15*cse741*d2g*state.xn[0] - cse798 - cse933 - state.xn[11]*(cse1038*cse1259 + cse1039*cse1259 + cse1255*cse858 + cse1258*cse859) - state.xn[3]*((cse889) ? (
   0
)
: (
   cse1042*cse907
)) - state.xn[7]*((cse1017) ? (
   cse1015*cse1261 + cse1016*cse1264 + cse1060*cse1265 + cse1061*cse1265
)
: (
   0
))) - cse1204*cse257;
  jac(19, 5) = -3.3466350616289996e-24*cse1019 + cse1020*(-cse1085*(-cse1253 - cse794*(cse1073*cse1074*cse1251 - cse1209*cse1252)) + cse1144*(-cse1271*cse799 + 2*cse35*cse738*cse799) - cse1239 - cse1240 - cse1241 + cse1244*cse1269 + cse1245*cse1270 + 1.1341179236737169e-34*cse1268 + cse1272*(-cse1250 - cse1271*cse726) + cse57*cse741 - 7.1777529600000012e-12*cse600 - 7.1777529600000012e-12*cse601) - cse1204*cse274;
  jac(19, 6) = -6.6464729999999996e-24*cse1019 - cse1204*cse280 - cse1273;
  jac(19, 7) = -6.6455620616289996e-24*cse1019 - cse1204*cse281 - cse1273;
  jac(19, 8) = -cse1018*cse1020 - 1.9944730000000003e-23*cse1019 - cse1204*cse289;
  jac(19, 9) = -1.9943819061629003e-23*cse1019 - cse1020*cse763*cse766 - cse1204*cse290;
  jac(19, 10) = -4.6512360000000008e-23*cse1019 + cse1020*(-cse1085*(cse1081*cse1082*cse1274*cse770 - cse793*(cse1081*cse1082*cse1274*cse774*cse786 - cse1275*cse787) - cse794*(cse1073*cse1074*cse1274 - cse1209*cse1275)) - cse770*cse797) - cse1204*cse292;
  jac(19, 11) = -4.8185222061629005e-23*cse1019 - cse1204*cse293 - cse1277;
  jac(19, 12) = -2.6567630000000005e-23*cse1019 - cse1020*cse860 - cse1204*cse299;
  jac(19, 13) = -4.6637070000000001e-23*cse1019 - cse1204*cse303;
  jac(19, 14) = -4.6636159061629004e-23*cse1019 - cse1204*cse304;
  jac(19, 15) = -2.1618503000000003e-23*cse1019 - cse1204*cse313 - cse1277;
  jac(19, 16) = -2.8241403000000005e-23*cse1019 - cse1204*cse315 - cse1277;
  jac(19, 17) = -5.0204080616289998e-24*cse1019 + cse1020*(-cse1085*(cse1081*cse1082*cse1279*cse770 - cse793*(cse1081*cse1082*cse1279*cse774*cse786 - cse1280*cse787) - cse794*(cse1073*cse1074*cse1279 - cse1209*cse1280)) - cse1096*(300.0*cse1094 + 300.0*cse696) - cse1103*(6*state.xn[0] + 6*state.xn[1] + 6*state.xn[10] + 6*state.xn[14] + 6*state.xn[15] + 18*state.xn[16] + 12*state.xn[3] + 12*state.xn[4]) + cse1144*(-cse1278*cse799 + 3*cse35*cse738*cse799) - 5.6994379667262474e-29*cse1238 + 1.7011768855105754e-34*cse1268 + cse1269*(-1.5*cse1242 + 1.5*cse1243) + cse1270*((cse743) ? (
   0
)
: ((cse744) ? (
   (9.0/2.0)*cse1098
)
: ((cse745) ? (
   4*cse1098
)
: ((cse746) ? (
   cse1098
)
: (
   0
))))) + cse1272*(-9.0000000000000011e-15*cse1249*d2g - cse1278*cse726 + 3*cse35*cse726*cse738) - 7.1777529600000012e-12*cse328 + cse604*cse741 - 7.1777529600000012e-12*cse605) - cse1204*cse330;
  jac(19, 18) = -2.6566719061629008e-23*cse1019 - cse1204*cse339;
  jac(19, 19) = cse1204;
  jac(19, 20) = cse1020*(7.2520441538364322e-33*cse1281 + 4.2141157754966708e-36*cse1282 + 3.0778991435866862e-37*cse1284 + cse1288*(-1.078095961904418e-12*cse1286 - 1.5615652627396295e-10*cse1287) - cse722*(-3.6905652698873601e-10*cse1289 + 7.2520441538364316e-9*cse1290));
  jac(19, 22) = cse1020*(9.932866933823022e-33*cse1281 + 5.7719244882960692e-36*cse1282 + 4.2156889809890617e-37*cse1284 + cse1288*(-1.4766296928602576e-12*cse1286 - 2.1388203979794697e-10*cse1287) - 8.9777440182952746e-38*cse1291 + cse608*cse801 - cse722*(-5.0548359826225349e-10*cse1289 + 9.9328669338230208e-9*cse1290));
  jac(19, 24) = cse1020*(1.1956682352839107e-32*cse1281 + 6.9479504891109339e-36*cse1282 + 5.0746329715150791e-37*cse1284 + cse1288*(-1.7774920682950379e-12*cse1286 - 2.5746037149991978e-10*cse1287) - 3.5474571214840408e-29*cse1291 + cse609*cse801 - cse722*(-6.0847556493598216e-10*cse1289 + 1.1956682352839107e-8*cse1290));
  jac(19, 26) = cse1020*(1.3176625050428103e-32*cse1281 + 7.6568512704709209e-36*cse1282 + 5.5923987909837356e-37*cse1284 + cse1288*(-1.9588499403826908e-12*cse1286 - 2.8372910482085621e-10*cse1287) - 1.6552354925381238e-28*cse1291 + cse610*cse801 - cse722*(-6.7055844881628607e-10*cse1289 + 1.3176625050428102e-8*cse1290));
  jac(20, 1) = cse1294;
  jac(20, 2) = cse1294;
  jac(20, 4) = cse1295;
  jac(20, 5) = cse1295;
  jac(20, 11) = cse1294;
  jac(20, 13) = -cse529;
  jac(20, 15) = -cse1293 - cse306;
  jac(20, 16) = -cse1293 - cse318;
  jac(20, 17) = -3.4881978485786016e-21*cse1292;
  jac(20, 20) = cse1298;
  jac(21, 1) = cse1301;
  jac(21, 2) = cse1301;
  jac(21, 4) = cse1303;
  jac(21, 5) = cse1303;
  jac(21, 11) = cse1301;
  jac(21, 13) = -6.5847540874715911e-18*cse1299;
  jac(21, 15) = -5.4356586967624969e-18*cse1299 - cse1300;
  jac(21, 16) = -1.1464502398585298e-18*cse1299 - cse1300;
  jac(21, 17) = -3.4881978485786016e-21*cse1302;
  jac(21, 21) = cse1298;
  jac(22, 1) = cse1306;
  jac(22, 2) = cse1306;
  jac(22, 4) = -cse1307 - cse1308*cse247;
  jac(22, 5) = -cse1307;
  jac(22, 11) = cse1306;
  jac(22, 13) = -cse527;
  jac(22, 15) = -cse1305 - cse307;
  jac(22, 16) = -cse1305 - cse319;
  jac(22, 17) = -4.2626128386451148e-21*cse1304;
  jac(22, 22) = cse1310;
  jac(23, 1) = cse1313;
  jac(23, 2) = cse1313;
  jac(23, 4) = -cse1308*cse1311 - cse1315;
  jac(23, 5) = -cse1315;
  jac(23, 11) = cse1313;
  jac(23, 13) = -4.4805860217826794e-17*cse1311;
  jac(23, 15) = -1.9473397080211686e-18*cse1311 - cse1312;
  jac(23, 16) = -2.7737849364247527e-18*cse1311 - cse1312;
  jac(23, 17) = -4.2626128386451148e-21*cse1314;
  jac(23, 23) = cse1310;
  jac(24, 1) = cse1318;
  jac(24, 2) = cse1318;
  jac(24, 4) = -cse1319 - cse1320*cse251;
  jac(24, 5) = -cse1319;
  jac(24, 8) = -cse519;
  jac(24, 10) = -cse637;
  jac(24, 11) = cse1318;
  jac(24, 13) = -cse528;
  jac(24, 15) = -cse1317 - cse308;
  jac(24, 16) = -cse1317 - cse320;
  jac(24, 17) = -5.6628985663562247e-21*cse1316;
  jac(24, 24) = cse1322;
  jac(25, 1) = cse1325;
  jac(25, 2) = cse1325;
  jac(25, 4) = -cse1320*cse1323 - cse1327;
  jac(25, 5) = -cse1327;
  jac(25, 8) = -cse1323*cse518;
  jac(25, 10) = -cse1323*cse636;
  jac(25, 11) = cse1325;
  jac(25, 13) = -2.062878503133515e-17*cse1323;
  jac(25, 15) = -1.0937796810239276e-18*cse1323 - cse1324;
  jac(25, 16) = -6.87721070339377e-18*cse1323 - cse1324;
  jac(25, 17) = -5.6628985663562247e-21*cse1326;
  jac(25, 25) = cse1322;
  jac(26, 1) = cse1330;
  jac(26, 2) = cse1330;
  jac(26, 4) = -cse1331 - cse1332*cse253;
  jac(26, 5) = -cse1331;
  jac(26, 8) = -cse517;
  jac(26, 10) = -cse635;
  jac(26, 11) = cse1330;
  jac(26, 13) = -cse526;
  jac(26, 15) = -cse1329 - cse309;
  jac(26, 16) = -cse1329 - cse321;
  jac(26, 17) = -6.797628717797286e-21*cse1328;
  jac(26, 26) = cse1334;
  jac(27, 1) = cse1337;
  jac(27, 2) = cse1337;
  jac(27, 4) = -cse1332*cse1335 - cse1339;
  jac(27, 5) = -cse1339;
  jac(27, 8) = -cse1335*cse516;
  jac(27, 10) = -cse1335*cse634;
  jac(27, 11) = cse1337;
  jac(27, 13) = -1.7217538808319277e-17*cse1335;
  jac(27, 15) = -1.2582308138615626e-18*cse1335 - cse1336;
  jac(27, 16) = -8.8938650996318353e-19*cse1335 - cse1336;
  jac(27, 17) = -6.797628717797286e-21*cse1338;
  jac(27, 27) = cse1334;
  // $$JAFF END
}

AMREX_INLINE
void actual_rhs_init() {}

#endif
