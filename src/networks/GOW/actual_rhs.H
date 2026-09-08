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


  // $JAFF REPEAT idx, rhs, cse IN rhses $[REPLACE radeden\[\s*(\d+)\s*\] state.rn[2*\1] REPLACE photden\[\s*(\d+)\s*\] state.rn[2*\1] REPLACE rflux\[\s*(\d+)\s*\] state.rn[2*\1+1] REPLACE nden\[\s*(\d+)\s*\] state.xn[\1] REPLACE tgas state.T REPLACE c_hat state.c_hat  RADIATION True]$
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
  const amrex::Real cse12 = state.c_hat*state.rn[2*0];
  const amrex::Real cse13 = 1.7141251340095381e-18*cse12;
  const amrex::Real cse14 = state.c_hat*state.rn[2*1];
  const amrex::Real cse15 = 4.1730320353928637e-18*cse14;
  const amrex::Real cse16 = (cse13 + cse15)*state.xn[15];
  const amrex::Real cse17 = 3.5000000000000002e-11*state.xn[11]*state.xn[15];
  const amrex::Real cse18 = 1.0/state.T;
  const amrex::Real cse19 = 7.9513696852843427e-10*std::pow(state.T, -0.33900000000000002)*std::exp(0.11*cse18)*state.xn[7]*state.xn[15];
  const amrex::Real cse20 = M_LN10;
  const amrex::Real cse21 = 1.0/cse20;
  const amrex::Real cse22 = 2*state.xn[3];
  const amrex::Real cse23 = cse22 + state.xn[0] + state.xn[1] + 2*state.xn[4] + state.xn[10] + state.xn[14] + state.xn[15] + 3*state.xn[16];
  const amrex::Real cse24 = std::log((1.0/10000.0)*state.T);
  const amrex::Real cse25 = std::pow(cse20, -2);
  const amrex::Real cse26 = std::pow(cse24, 2)*cse25;
  const amrex::Real cse27 = 1.0/cse23;
  const amrex::Real cse28 = cse27*state.xn[3];
  const amrex::Real cse29 = cse27*state.xn[0];
  const amrex::Real cse30 = std::pow(10, 0.41599999999999998*cse21*cse24 + 0.32700000000000001*cse26 - 3)*cse29 + std::pow(10, 1.3*cse21*cse24 - 1.6200000000000001*cse26 - 4.8449999999999998)*cse28;
  const amrex::Real cse31 = cse23*((cse30 < 1.0e-50) ? (
   1.0e-50
)
: (
   cse30
));
  const amrex::Real cse32 = cse21/(cse31 + 1);
  const amrex::Real cse33 = cse31*cse32;
  const amrex::Real cse34 = state.T > 700.0;
  const amrex::Real cse35 = ((cse34) ? (
   std::pow(10, cse32*std::log(5.9959999999999997e-30*std::pow(state.T, 4.1881000000000004)*std::pow(6.7610000000000002e-6*state.T + 1, -5.6881000000000004)*std::exp(-54657.400000000001*cse18)) + cse33*std::log(1.3000000000000001e-9*std::exp(-53300*cse18)))
)
: (
   0.0
))*std::pow(state.xn[3], 2);
  const amrex::Real cse36 = std::max(1.0e-35, 2.0000000000000002e-15*ncol_H2);
  const amrex::Real cse37 = 0.035000000000000003*std::pow(cse36 + 1, -0.5)*std::exp(-0.00084999999999999995*std::sqrt(cse36 + 1.0)) + 0.96499999999999997*std::pow(cse36/std::max(9.9999999999999995e-21, 1.4142135623730953e-5*vdisp) + 1.0, -2.0);
  const amrex::Real cse38 = cse12*cse37;
  const amrex::Real cse39 = cse14*cse37;
  const amrex::Real cse40 = 1.2216858566394236e-27*cse38 + 3.6749360303422164e-18*cse39;
  const amrex::Real cse41 = (7.5e-10*std::pow(state.T, -0.45800000000000002) + 4.9899999999999997e-11*std::pow(state.T, 0.40500000000000003))*state.xn[0]*state.xn[17];
  const amrex::Real cse42 = ((cse34) ? (
   std::pow(10, cse32*std::log(6.6699999999999996e-12*cse2*std::exp(-63590*cse18 - 1)) + cse33*std::log(3.5199999999999998e-9*std::exp(-43900*cse18)))
)
: (
   0.0
))*state.xn[0];
  const amrex::Real cse43 = std::pow(state.T, -0.52000000000000002);
  const amrex::Real cse44 = cse43*state.xn[2]*state.xn[16];
  const amrex::Real cse45 = cse23*d2g;
  const amrex::Real cse46 = (4.0000000000000001e-10*std::pow(state.T, 0.0066899999999999998) + 1.1000000000000001e-11*std::pow(state.T, 0.51700000000000002))*std::exp(-227*cse18)*state.xn[1]*state.xn[11];
  const amrex::Real cse47 = std::pow(state.T, -1.3);
  const amrex::Real cse48 = std::exp(-23*cse18);
  const amrex::Real cse49 = cse47*cse48*state.xn[3]*state.xn[8];
  const amrex::Real cse50 = 4.2188032355736621e-18*cse12;
  const amrex::Real cse51 = 1.168047934563505e-18*cse14;
  const amrex::Real cse52 = (cse50 + cse51)*state.xn[14];
  const amrex::Real cse53 = 7.0000000000000004e-11*state.xn[11]*state.xn[14];
  const amrex::Real cse54 = cse52 + cse53;
  const amrex::Real cse55 = (26.218500000000002*std::sqrt(3)*std::sqrt(cse18) + 0.62)*state.xn[15];
  const amrex::Real cse56 = 1.3500000000000001e-9*cse55*state.xn[6];
  const amrex::Real cse57 = state.xn[3]*state.xn[6];
  const amrex::Real cse58 = 1.2599999999999999e-13*cse57*std::exp(-22.5*cse18);
  const amrex::Real cse59 = cse56 + cse58;
  const amrex::Real cse60 = 2.8142984982406731e-11*std::pow(state.T, 0.26000000000000001)*state.xn[0]*state.xn[14];
  const amrex::Real cse61 = 1.7599999999999999e-9*std::pow(state.T, 0.042000000000000003)*std::exp(-2.145922746781116e-5*state.T)*state.xn[3]*state.xn[4];
  const amrex::Real cse62 = -cse60 + cse61;
  const amrex::Real cse63 = crate*state.xn[9];
  const amrex::Real cse64 = 6.5199999999999996*cse63;
  const amrex::Real cse65 = crate*(2.2999999999999998*cse28 + 1.5*cse29);
  const amrex::Real cse66 = cse65*state.xn[0];
  const amrex::Real cse67 = std::log(8.6173000000000003e-5*state.T);
  const amrex::Real cse68 = state.xn[0]*state.xn[2];
  const amrex::Real cse69 = cse68*((cse34) ? (
   5.9078447946382801e-70*std::pow(state.T, 13.536555999999999)*std::exp(-2.0391498499999999e-6*std::pow(cse67, 8) + 0.000111954395*std::pow(cse67, 7) - 0.00263197617*std::pow(cse67, 6) + 0.034825597700000002*std::pow(cse67, 5) - 0.28770560000000001*std::pow(cse67, 4) + 1.56315498*std::pow(cse67, 3) - 5.7393287500000003*std::pow(cse67, 2))
)
: (
   0.0
));
  const amrex::Real cse70 = 1.05849862170431e-5*std::pow(state.T, -0.64000000000000001)*state.xn[2]*state.xn[10];
  const amrex::Real cse71 = 4.8813573300130879e-6*std::pow(cse18, 1.5)*std::pow(114.81344434488582*std::pow(cse18, 0.40699999999999997) + 1, -2.242)*state.xn[1]*state.xn[2];
  const amrex::Real cse72 = 119.95506989726393*state.rn[2*0] + 37.314487691080771*state.rn[2*1];
  const amrex::Real cse73 = state.c_hat*cse72;
  const amrex::Real cse74 = cse3*cse73;
  const amrex::Real cse75 = 5.670589618368584e-11*cse74;
  const amrex::Real cse76 = std::log(state.T);
  const amrex::Real cse77 = 1.2250000000000001e-11*cse45*state.xn[1]/(6.1318063394789369e-20*std::pow(cse74, 1.3779999999999999)*(508.69999999999999*std::pow(cse75, -1.102e-5*cse76 - 0.4723)*std::pow(state.T, 0.015859999999999999) + 1) + 1);
  const amrex::Real cse78 = -cse64 - cse66 - cse69 + cse70 + cse71 + cse77;
  const amrex::Real cse79 = cse41 - cse46;
  const amrex::Real cse80 = 2.1635567548454671e-17*cse12;
  const amrex::Real cse81 = 1.913494189605312e-17*cse14;
  const amrex::Real cse82 = (cse80 + cse81)*state.xn[12];
  const amrex::Real cse83 = cse22*cse65;
  const amrex::Real cse84 = 1.4595000290776864e-10*std::pow(state.T, -0.62)*state.xn[2]*state.xn[13];
  const amrex::Real cse85 = 2.1659999999999999e-12*cse45*state.xn[13]/(3.5685736926834518e-27*std::pow(cse74, 1.8740000000000001)*(43750.0*std::pow(cse75, -7.538e-5*cse76 - 0.89639999999999997)*std::pow(state.T, 1.635e-6) + 1) + 1);
  const amrex::Real cse86 = 16800.0*crate*cse28*state.xn[12];
  const amrex::Real cse87 = (1.0537934368240927e-9*std::pow(cse18, 0.00231) + std::pow(state.T, -1.5)*(0.00015100000000000001*std::exp(-8110.0*cse18) + 1.31e-7*std::exp(-26.600000000000001*cse18) + 3.4e-8*std::exp(-7.6200000000000001*cse18) + 6.9699999999999997e-9*std::exp(-1.3799999999999999*cse18)))*state.xn[7]*state.xn[16];
  const amrex::Real cse88 = cse27*state.xn[2];
  const amrex::Real cse89 = ((cse88 < 1.0e-50) ? (
   10000000000.0
)
: (
   cse4
));
  const amrex::Real cse90 = cse10*cse89/(cse89 + 1);
  const amrex::Real cse91 = cse11 + 1.3007021411393064e-6*cse44 + cse87 + cse90;
  const amrex::Real cse92 = cse21*cse76;
  const amrex::Real cse93 = -1.1000000000000001*crate*state.xn[5] + 5.5720000000000001e-12*cse45*state.xn[6]/(1.0246966183388124e-22*std::pow(cse74, 1.512)*(5115.0*std::pow(cse75, -5.4939999999999998e-7*cse76 - 0.49559999999999998)*std::pow(state.T, 3.9029999999999999e-7) + 1) + 1) + 9.9999999999999994e-12*std::pow(state.T, -0.5)*(cse92*(cse92*(0.044330000000000001*cse92 - 0.28520000000000001) - 1.6759999999999999) + 11.19)*state.xn[2]*state.xn[6];
  const amrex::Real cse94 = crate*state.xn[7];
  const amrex::Real cse95 = 3.8500000000000001*cse94;
  const amrex::Real cse96 = 0.15970000000000001*std::exp(-49550.0*cse18);
  const amrex::Real cse97 = ((1.6339999999999999e-6*std::exp(-15230.0*cse18) + 9.7930000000000004e-9*std::exp(-73.799999999999997*cse18) + 6.3460000000000002e-9*std::exp(-12.17*cse18))/std::pow(state.T, 3.0/2.0) + 2.4460185352936309e-10*std::pow(0.00071740367932898903*cse2 + 1, -cse96 - 1.7848999999999999)*std::pow(12.244387999457523*cse2 + 1, cse96 - 0.21509999999999996)/cse2)*state.xn[2]*state.xn[8];
  const amrex::Real cse98 = 4.5579999999999993e-11*cse45*state.xn[8]/(1.6851501542381549e-14*std::pow(cse74, 1.1279999999999999)*(433.10000000000002*std::pow(cse75, -0.00013329999999999999*cse76 - 0.81200000000000006)*std::pow(state.T, 0.04845) + 1) + 1);
  const amrex::Real cse99 = 1040.0*cse28*cse94;
  const amrex::Real cse100 = 1.6927240490436929e-17*interp_8_shielding_H2(ncol_H2)*interp_8_shielding_self(ncol_C)*state.xn[7];
  const amrex::Real cse101 = cse100*cse14;
  const amrex::Real cse102 = -cse101 - cse95 + cse97 + cse98 - cse99;
  const amrex::Real cse103 = 3.2999999999999996e-13*cse49;
  const amrex::Real cse104 = cse8 + cse9;
  const amrex::Real cse105 = cse103 + cse104;
  const amrex::Real cse106 = -cse87;
  const amrex::Real cse107 = -cse90;
  const amrex::Real cse108 = 1.6999999999999999e-9*state.xn[9]*state.xn[16];
  const amrex::Real cse109 = cse42*state.xn[3];
  const amrex::Real cse110 = 7.2000000000000002e-15*cse57;
  const amrex::Real cse111 = cse1 + cse110 + cse83;
  const amrex::Real cse112 = -cse61;
  const amrex::Real cse113 = 1.6000000000000001e-9*state.xn[6]*state.xn[9];
  const amrex::Real cse114 = cse110 + cse113 + cse59 + cse93;
  const amrex::Real cse115 = cse106 + cse60;
  const amrex::Real cse116 = 1.6797535700197549e-17*interp_10_shielding_H2(ncol_H2)*interp_10_shielding_self(ncol_CO)*state.xn[9];
  const amrex::Real cse117 = cse116*cse14;
  const amrex::Real cse118 = 184.0*cse28*cse63;
  const amrex::Real cse119 = cse117 + cse118 - cse19;
  const amrex::Real cse120 = 9.1500000000000005e-10*cse55*state.xn[8];
  const amrex::Real cse121 = cse108 + cse64 - cse70;
  const amrex::Real cse122 = cse113 - cse53;
  const amrex::Real cse123 = cse107 + cse16 + cse17;
  const amrex::Real cse124 = cse82 - cse84 - cse85 + cse86;
  const amrex::Real cse125 = std::pow(state.T, 0.14999999999999999)/(0.00088240812266906664*std::pow(state.T, 0.65000000000000002) + 1);
  const amrex::Real cse126 = cse125*state.xn[2];
  const amrex::Real cse127 = std::exp(-118400.0*cse18)/(4.4460389837719569e-9*cse126 + 626500000.0);
  const amrex::Real cse128 = cse75 + 50;
  const amrex::Real cse129 = std::log(5.670589618368584e-11*cse2*cse73/(state.xn[2] + 1.0e-50) + 50);
  const amrex::Real cse130 = std::sqrt(cse18)*state.xn[2];
  const amrex::Real cse131 = (1.0/100.0)*state.T;
  const amrex::Real cse132 = std::log(cse131);
  const amrex::Real cse133 = state.T < 500;
  const amrex::Real cse134 = std::pow(state.T, 0.070000000000000007);
  const amrex::Real cse135 = 4.5299999999999998e-6*cse130 + 7.5799999999999997e-10*std::pow(cse131, 0.0086999999999999994*cse132 + 0.12809999999999999)*state.xn[0] + (0.75*((cse133) ? (
   1.1000000000000001e-13*state.T + 5.3300000000000002e-10
)
: (
   3.74757785025e-10*cse134
)) + 0.25*((cse133) ? (
   3.3000000000000001e-13*state.T + 4.4299999999999997e-10
)
: (
   3.8899728635600003e-10*cse134
)))*state.xn[3];
  const amrex::Real cse136 = cse135*std::exp(-91.209999999999994*cse18)/(cse135 + 2.3e-6);
  const amrex::Real cse137 = 0.001*state.T;
  const amrex::Real cse138 = cse137 + 1;
  const amrex::Real cse139 = std::pow(cse138, -2);
  const amrex::Real cse140 = 1.0/cse138;
  const amrex::Real cse141 = cse27/(std::pow(10, -2.3900000000000001*cse139 + 0.055500000000000001*cse140 - 11.06)*cse29 + std::pow(10, -2.0230000000000001*cse139 - 3.6709999999999998*cse140 - 11.08)*cse28);
  const amrex::Real cse142 = std::pow(state.T, 1.0/4.0);
  const amrex::Real cse143 = 1.75*cse142*state.xn[0] + state.xn[3] + 680.10000000000002*state.xn[2]/cse142;
  const amrex::Real cse144 = l0_co_interp1d(state.T);
  const amrex::Real cse145 = 1.0/cse144;
  const amrex::Real cse146 = ((ncol_H <= 0) ? (
   100000.0*state.xn[9]/std::max(gradv, 5.7445531974402559e-18*M_SQRT2*cse2)
)
: (
   1756701454.4310031*ncol_CO/std::sqrt(std::max(1.0000000000000001e-30, cse27*ncol_H))
));
  const amrex::Real cse147 = 1.0/llte_co_interp2d(state.T, cse146);
  const amrex::Real cse148 = nhalf_co_interp2d(state.T, cse146);
  const amrex::Real cse149 = cse21*std::log(cse23);
  const amrex::Real cse150 = 2.0*cse28*((cse149 < 2) ? (
   10.0
)
: ((cse149 < 4) ? (
   (3.0/2.0)*cse149 + 7
)
: ((cse149 < 7) ? (
   (4.0/3.0)*cse149 + 23.0/3.0
)
: ((cse149 < 10) ? (
   (1.0/3.0)*cse149 + 44.0/3.0
)
: (
   18.0
))))) + cse29*(26.399999999999999*std::sqrt(cse88/(cse88 + 0.070000000000000007)) + 6.5);
  const amrex::Real cse151 = 0.042999999999999997*cse132;
  const amrex::Real cse152 = 4.34e-10*std::pow(cse131, 0.755 - 0.16*cse132)*state.xn[0] + 1.08e-14*std::pow(state.T, 0.92600000000000005)*state.xn[2] + (2.2500000000000003e-12*std::pow(cse131, 0.52500000000000002*cse132 + 1.198) + 5.2499999999999998e-13*std::pow(cse131, cse151 + 0.88900000000000001))*state.xn[3];
  const amrex::Real cse153 = 3.5700000000000001e-10*std::pow(cse131, 0.41899999999999998 - 0.0030000000000000001*cse132)*state.xn[0] + 5.1199999999999999e-10*std::pow(state.T, -0.074999999999999997)*state.xn[2] + (3.7250000000000001e-11*std::pow(cse131, 0.025000000000000001*cse132 + 0.26400000000000001) + 1.0274999999999999e-10*std::pow(cse131, cse151 + 0.29599999999999999))*state.xn[3];
  const amrex::Real cse154 = (1.0/5.0)*std::exp(-326.60000000000002*cse18);
  const amrex::Real cse155 = 3.1899999999999998e-10*std::pow(cse131, 0.36899999999999999 - 0.0060000000000000001*cse132)*state.xn[0] + 4.8599999999999998e-10*std::pow(state.T, -0.025999999999999999)*state.xn[2] + (4.7499999999999998e-11*std::pow(cse131, 0.041000000000000002*cse132 + 0.20300000000000001) + 1.6725000000000002e-10*std::pow(cse131, 0.058000000000000003*cse132 + 0.23699999999999999))*state.xn[3];
  const amrex::Real cse156 = cse153 + 8.9099999999999997e-5;
  const amrex::Real cse157 = cse154*cse155;
  const amrex::Real cse158 = cse152*cse153*cse154 + (1.0/15.0)*cse152*cse155*std::exp(-425.40000000000003*cse18) + cse156*cse157;
  const amrex::Real cse159 = cse152 + 1.7499999999999998e-5;
  const amrex::Real cse160 = cse155 + 1.34e-10;
  const amrex::Real cse161 = (3.0/5.0)*cse153*std::exp(-227.80000000000001*cse18);
  const amrex::Real cse162 = cse157*cse159 + cse159*cse161 + cse160*cse161;
  const amrex::Real cse163 = 1.0/((1.0/3.0)*cse152*cse160*std::exp(-98.800000000000011*cse18) + cse156*cse159 + cse156*cse160 + cse158 + cse162);
  const amrex::Real cse164 = state.T < 100.0;
  const amrex::Real cse165 = state.T < 6000.0;
  const amrex::Real cse166 = state.T < 10;
  const amrex::Real cse167 = std::log(cse137);
  const amrex::Real cse168 = cse167*cse21;
  const amrex::Real cse169 = std::pow(cse20, -8);
  const amrex::Real cse170 = std::pow(cse167, 8)*cse169;
  const amrex::Real cse171 = std::pow(cse20, -7);
  const amrex::Real cse172 = std::pow(cse167, 7)*cse171;
  const amrex::Real cse173 = std::pow(cse20, -6);
  const amrex::Real cse174 = std::pow(cse167, 6)*cse173;
  const amrex::Real cse175 = std::pow(cse20, -5);
  const amrex::Real cse176 = std::pow(cse167, 5)*cse175;
  const amrex::Real cse177 = std::pow(cse20, -4);
  const amrex::Real cse178 = std::pow(cse167, 4)*cse177;
  const amrex::Real cse179 = std::pow(cse20, -3);
  const amrex::Real cse180 = std::pow(cse167, 3);
  const amrex::Real cse181 = cse179*cse180;
  const amrex::Real cse182 = std::pow(cse167, 2)*cse25;
  const amrex::Real cse183 = std::pow(10.0, 16.815729999999999*cse168 + 70.609154000000004*cse170 + 364.14445999999998*cse172 + 801.81246999999996*cse174 + 983.67575999999997*cse176 + 734.71650999999997*cse178 + 343.1918*cse181 + 96.743155000000002*cse182 - 21.928795999999998)*state.xn[2];
  const amrex::Real cse184 = state.xn[5] + state.xn[6];
  const amrex::Real cse185 = std::pow(10.0, 1.5714710999999999*cse168 + 0.32168730000000001*cse176 - 0.51002221000000003*cse178 - 0.23619984999999999*cse181 + 0.015391166*cse182 - 22.089523)*state.xn[1] + std::pow(10.0, 2.1892372*cse168 + 0.19191374999999999*cse176 - 0.16596184*cse178 + 0.29036281000000003*cse181 - 0.81520437999999995*cse182 - 23.689236999999999)*cse184 + std::pow(10.0, 2.0943374000000001*cse167*cse21 - 0.033638326000000003*cse176 - 0.14913216000000001*cse178 + 0.43693353000000001*cse179*cse180 - 0.77151435999999995*cse182 - 23.962112000000001)*state.xn[3];
  const amrex::Real cse186 = std::pow(10, 37.383713*cse168 + 3.8479610000000002*cse176 + 20.159831000000001*cse178 + 48.656103000000002*cse181 + 58.145166000000003*cse182 - 16.818342000000001)*state.xn[0] + cse185;
  const amrex::Real cse187 = std::pow(10.0, 1.6802758*cse168 - 6.3701156000000001*cse170 + 6.4380698000000001*cse172 + 8.9167182999999994*cse174 - 8.8077017000000009*cse176 - 4.7274035999999997*cse178 + 4.0406627000000004*cse181 + 0.93310621999999999*cse182 - 22.921188999999998)*state.xn[2];
  const amrex::Real cse188 = state.T < 1000.0;
  const amrex::Real cse189 = std::pow(10.0, 3.5692468000000002*cse167*cse21 - 4.2519023000000002*cse176 - 21.328264000000001*cse178 - 27.850082*cse181 - 11.33286*cse182 - 24.311209000000002)*state.xn[0] + cse185;
  const amrex::Real cse190 = std::pow(10.0, 4.6450521*cse168 + 1.5538288*cse176 - 5.5108049000000001*cse178 + 5.9369081000000001*cse181 - 3.7209846*cse182 - 24.311209000000002)*state.xn[0] + cse185;
  const amrex::Real cse191 = std::pow(10.0, -0.62120210234156448*cse175 - 1.5370575959566981*cse177 + 2.5133588394664677*cse179 + 3.7525488682084647*cse21 - 2.4768712409536775*cse25 - 23.962112000000001)*state.xn[3] + std::pow(10.0, 3.5440891133599637*cse175 - 1.7105157385968939*cse177 + 1.6702447513374001*cse179 + 3.9225864834863131*cse21 - 2.6171337683481011*cse25 - 23.689236999999999)*cse184 + std::pow(10.0, 5.9406293599919788*cse175 - 5.2566362076906961*cse177 - 1.3586848802337363*cse179 + 2.8156982240432273*cse21 + 0.049411830040524525*cse25 - 22.089523)*state.xn[1] + std::pow(10.0, 28.69470131298657*cse175 - 56.79810800956944*cse177 + 34.150687525022562*cse179 + 8.3228160852326614*cse21 - 11.945856385319289*cse25 - 24.311209000000002)*state.xn[0];
  const amrex::Real cse192 = ((cse165) ? (
   cse18
)
: (
   0.00016666666666666666
));
  const amrex::Real cse193 = 1.6e-18*std::exp(-11700.0*cse192) + 6.7e-19*std::exp(-5860.0*cse192) + 3.0e-24*std::exp(-510.0*cse192) + 4.985670872372847e-33*((cse165) ? (
   std::pow(state.T, 3.7599999999999998)
)
: (
   160638152785302.53
))*std::exp(-2197000.0*((cse165) ? (
   std::pow(state.T, -3)
)
: (
   4.6296296296296294e-12
)))/(6.0142468035272636e-8*((cse165) ? (
   std::pow(state.T, 2.1000000000000001)
)
: (
   85924621.118090734
)) + 1.0);
  const amrex::Real cse194 = std::log(0.001*((cse165) ? (
   state.T
)
: (
   6000.0
)));
  const amrex::Real cse195 = cse175*std::pow(cse194, 5);
  const amrex::Real cse196 = cse177*std::pow(cse194, 4);
  const amrex::Real cse197 = std::pow(cse194, 3);
  const amrex::Real cse198 = std::pow(cse194, 2)*cse25;
  const amrex::Real cse199 = cse194*cse21;
  const amrex::Real cse200 = cse179*cse197;
  const amrex::Real cse201 = cse169*std::pow(cse194, 8);
  const amrex::Real cse202 = cse171*std::pow(cse194, 7);
  const amrex::Real cse203 = cse173*std::pow(cse194, 6);
  const amrex::Real cse204 = 0.045999999999999999*cse132;
  const amrex::Real cse205 = 1.7258000000000001e-6*cse130;
  const amrex::Real cse206 = 8.9000000000000003e-11*std::pow(cse131, cse204 + 0.22800000000000001)*state.xn[0] + cse205*std::exp(((cse188) ? (
   cse76*(cse76*(cse76*(0.0070527699999999999*cse76 - 0.11133800000000001) + 0.69763799999999998) - 1.3074300000000001) - 7.6973500000000001
)
: (
   cse76*(cse76*(cse76*(0.093813800000000003*cse76 - 3.0328300000000001) + 36.180300000000003) - 187.47399999999999) + 350.60899999999998
))) + (5.1749999999999996e-11*std::pow(cse131, 0.037999999999999999*cse132 + 0.16900000000000001) + 2.15e-11*std::pow(cse131, 0.048000000000000001*cse132 - 0.01))*state.xn[3];
  const amrex::Real cse207 = 2.6400000000000002e-10*std::pow(cse131, cse204 + 0.23100000000000001)*state.xn[0] + cse205*std::exp(((cse188) ? (
   cse76*(cse76*(cse76*(0.00235272*cse76 - 0.041816600000000002) + 0.35826400000000003) - 0.57443) - 7.4386999999999999
)
: (
   cse76*(cse76*(cse76*(0.097857299999999994*cse76 - 3.1926800000000002) + 38.504899999999999) - 202.19300000000001) + 386.18599999999998
))) + (1.11e-10*std::pow(cse131, 0.031*cse132 + 0.26300000000000001) + 4.3749999999999998e-11*std::pow(cse131, 0.064000000000000001*cse132 + 0.071999999999999995))*state.xn[3];
  const amrex::Real cse208 = 2.8763333333333333e-6*cse130*std::exp(((cse188) ? (
   cse76*(cse76*(cse76*(-0.00065632500000000003*cse76 - 0.015089200000000001) + 0.36118400000000001) - 0.77378199999999997) - 9.2514099999999999
)
: (
   cse76*(cse76*(cse76*(0.10508000000000001*cse76 - 3.4762) + 42.594999999999999) - 227.91300000000001) + 444.60000000000002
))) + 1.26e-10*std::pow(cse131, 0.057000000000000002*cse132 + 0.115)*state.xn[0] + (5.3249999999999997e-11*std::pow(cse131, 0.049000000000000002*cse132 - 0.0040000000000000001) + 1.675e-11*std::pow(cse131, 0.10199999999999999*cse132 - 0.085000000000000006))*state.xn[3];
  const amrex::Real cse209 = 5*std::exp(-62.460000000000001*cse18);
  const amrex::Real cse210 = cse208 + 7.8800000000000004e-8;
  const amrex::Real cse211 = cse206*cse209;
  const amrex::Real cse212 = (25.0/3.0)*cse206*cse207*std::exp(-101.30000000000001*cse18) + cse207*cse208*cse209 + cse210*cse211;
  const amrex::Real cse213 = cse207 + 2.65e-7;
  const amrex::Real cse214 = cse206 + 1.81e-14;
  const amrex::Real cse215 = 3*cse208*std::exp(-23.619999999999997*cse18);
  const amrex::Real cse216 = cse211*cse213 + cse213*cse215 + cse214*cse215;
  const amrex::Real cse217 = 1.0/((5.0/3.0)*cse207*cse214*std::exp(-38.840000000000003*cse18) + cse210*cse213 + cse210*cse214 + cse212 + cse216);
  const amrex::Real cse218 = 1.8689621674132958e-20*state.xn[3];
  const amrex::Real cse219 = Zd*cse23;
  const amrex::Real cse220 = 1.252779946046115e-21*cse219;
  const amrex::Real cse221 = state.c_hat*state.rn[2*0+1];
  const amrex::Real cse222 = 2.6551134042892157e-17*state.xn[3];
  const amrex::Real cse223 = 2.0584539190345922e-21*cse219;
  const amrex::Real cse224 = state.c_hat*state.rn[2*1+1];

  ydot(1) = cse1 + cse11 + cse16 + cse17 + cse19 + cse22*cse40 + cse22*cse42 + 2*cse35 - cse41 + 2.9935562710997168e-6*cse44 - 6.0000000000000005e-15*cse45*state.xn[0] + cse46 + 4.2899999999999994e-13*cse49 + cse54 + cse59 + cse62 + 3.2000000000000001e-9*cse7 + cse78 + cse9;
  ydot(2) = cse0 + cse58 + cse66 + cse69 - cse71 - cse77 + cse79;
  ydot(3) = -cse102 - cse105 - cse78 + cse82 + cse83 - cse84 - cse85 + cse86 - cse91 - cse93;
  ydot(4) = -cse105 - cse106 - cse107 + cse108 - cse109 + cse11 - cse111 + 3.0000000000000002e-15*cse23*d2g*state.xn[0] - cse35 - cse40*state.xn[3] + 4.5427507615910105e-7*cse43*state.xn[2]*state.xn[16] - cse58 - cse62;
  ydot(5) = cse111 + cse112;
  ydot(6) = cse114;
  ydot(7) = -cse114;
  ydot(8) = cse102 + cse115 + cse119 + 9.8999999999999995e-14*cse49 + cse52;
  ydot(9) = cse101 - cse103 + cse113 - cse120 + cse95 - cse97 - cse98 + cse99;
  ydot(10) = -cse119 - cse121 - cse122;
  ydot(11) = cse120 + cse121;
  ydot(12) = cse117 + cse118 + cse122 + cse123 + cse79 + cse8;
  ydot(13) = -cse124;
  ydot(14) = cse124;
  ydot(15) = -cse115 + 2.3099999999999997e-13*cse47*cse48*state.xn[3]*state.xn[8] - cse54;
  ydot(16) = -cse120 - cse123 - cse19 - cse56 + cse9;
  ydot(17) = -cse108 - cse112 - cse91;
  ydot(18) = -cse104 + cse56 - cse79;
  ydot(19) = 5.6705896183685846e-35*state.c_hat*cse23*cse72*d2g*(2.25*std::pow(state.T, 0.14699999999999999) + 5.2199999999999998)/(0.049959999999999997*std::pow(cse128, 0.43099999999999999)*(0.0043*std::pow(cse128, 0.69199999999999995) + 1) + 1) + 1.602177e-12*crate*cse150*state.xn[0] + 3.2043540000000001e-12*crate*cse150*state.xn[3] - 7.1777529600000012e-12*cse109 - 1.3659996292638161e-10*cse125*cse127*cse68/(1.3338116951315871e-8*cse126*cse127 + 1) - 5.7927337833400002e-20*cse136*state.xn[8]/(2*cse136 + 1) - cse143*std::pow(1 - std::exp(-state.T), 1000)*state.xn[9]/(cse143*cse147 + cse145*std::pow(cse143/cse148, alpha_co_interp2d(state.T, cse146))*(-cse144*cse147*cse148 + 1) + cse145) + 3.0000000000000002e-15*cse23*d2g*(3.2043540000000002e-13 + 6.7291434000000005e-12/(cse141*(3.0000000000000002e-15*cse45/state.xn[0] + 1.9999999999999999e-7) + 1))*state.xn[0] - 7.1777529600000012e-12*cse35 + cse40*(6.4087080000000003e-13 + 2.5634832e-11/(cse141*(cse40 + 1.9999999999999999e-7) + 1))*state.xn[3] - 1.8998126555754159e-29*cse45*std::pow(state.T, 0.45350000000000001 + 2.234/cse129)*std::exp(cse129*(1.4419999999999999 - 0.050889999999999998*cse129))*state.xn[2] - 2.1789607199999999e-11*cse69 - (2.3872025442750958e-19*cse158*cse163 + 2.8023005140020001e-18*cse162*cse163)*state.xn[11] - ((((cse133 && cse164 && cse165) ? (
   cse166 || cse183 + cse186 < 1.0e-100
)
: (
   ((cse164 && cse165) ? (
      cse166 || cse186 + cse187 < 1.0e-100
   )
   : (
      ((cse133 && cse165 && cse188) ? (
         cse166 || cse183 + cse189 < 1.0e-100
      )
      : (
         ((cse165 && cse188) ? (
            cse166 || cse187 + cse189 < 1.0e-100
         )
         : (
            ((cse133 && cse165) ? (
               cse166 || cse183 + cse190 < 1.0e-100
            )
            : (
               ((cse165) ? (
                  cse166 || cse187 + cse190 < 1.0e-100
               )
               : (
                  ((cse133) ? (
                     cse166 || std::pow(10.0, 7500.646099484944*cse169 + 21588.957206746247*cse171 + 26530.843406310876*cse173 + 18165.631967965237*cse175 + 7572.4886742758581*cse177 + 1974.131269262874*cse179 + 30.129743459482278*cse21 + 310.58441786958315*cse25 - 21.928795999999998)*state.xn[2] + cse191 < 1.0e-100
                  )
                  : (
                     cse166 || std::pow(10.0, -676.68255490510751*cse169 + 381.69251128040054*cse171 + 295.04162851880631*cse173 - 162.65264843551881*cse175 - 48.723840736518795*cse177 + 23.242975457496804*cse179 + 3.0106500755647456*cse21 + 2.9956460707652877*cse25 - 22.921188999999998)*state.xn[2] + cse191 < 1.0e-100
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
   cse193/(cse193/(std::pow(10.0, 0.19191374999999999*cse195 - 0.16596184*cse196 - 0.81520437999999995*cse198 + 2.1892372*cse199 + 0.29036281000000003*cse200 - 23.689236999999999)*cse184 + std::pow(10.0, 0.32168730000000001*cse195 - 0.51002221000000003*cse196 + 0.015391166*cse198 + 1.5714710999999999*cse199 - 0.23619984999999999*cse200 - 22.089523)*state.xn[1] + std::pow(10.0, 0.43693353000000001*cse179*cse197 + 2.0943374000000001*cse194*cse21 - 0.033638326000000003*cse195 - 0.14913216000000001*cse196 - 0.77151435999999995*cse198 - 23.962112000000001)*state.xn[3] + ((cse133) ? (
      std::pow(10.0, 983.67575999999997*cse195 + 734.71650999999997*cse196 + 96.743155000000002*cse198 + 16.815729999999999*cse199 + 343.1918*cse200 + 70.609154000000004*cse201 + 364.14445999999998*cse202 + 801.81246999999996*cse203 - 21.928795999999998)
   )
   : (
      std::pow(10.0, -8.8077017000000009*cse195 - 4.7274035999999997*cse196 + 0.93310621999999999*cse198 + 1.6802758*cse199 + 4.0406627000000004*cse200 - 6.3701156000000001*cse201 + 6.4380698000000001*cse202 + 8.9167182999999994*cse203 - 22.921188999999998)
   ))*state.xn[2] + ((((cse165) ? (
      cse164
   )
   : (
      false
   ))) ? (
      std::pow(10, 3.8479610000000002*cse195 + 20.159831000000001*cse196 + 58.145166000000003*cse198 + 37.383713*cse199 + 48.656103000000002*cse200 - 16.818342000000001)
   )
   : ((((cse165) ? (
      cse188
   )
   : (
      false
   ))) ? (
      std::pow(10.0, 3.5692468000000002*cse194*cse21 - 4.2519023000000002*cse195 - 21.328264000000001*cse196 - 11.33286*cse198 - 27.850082*cse200 - 24.311209000000002)
   )
   : (
      std::pow(10.0, 1.5538288*cse195 - 5.5108049000000001*cse196 - 3.7209846*cse198 + 4.6450521*cse199 + 5.9369081000000001*cse200 - 24.311209000000002)
   )))*state.xn[0]) + 1)
))*state.xn[3] - ((state.T < 1000000.0) ? (
   1.4210469458259591e-21*cse212*cse217 + 2.5697412351439998e-22*cse216*cse217
)
: (
   0.0
))*state.xn[7] - std::min(100.0*d2g*psigd_coll_interp2d(cse23, state.T), 3.2000000000000002e-32*cse2*cse45*(state.T - 2.1544346900318838*std::pow(2.8677558904086741e-6*cse73 + 4.1397668473788896, 1.0/6.0)));
  ydot(20) = -cse12*cse220 - cse13*state.xn[15] - cse218*cse38 - cse50*state.xn[14] - cse80*state.xn[12];
  ydot(21) = -cse218*cse221*cse37 - cse220*cse221 - 2.1635567548454671e-17*cse221*state.xn[12] - 4.2188032355736621e-18*cse221*state.xn[14] - 1.7141251340095381e-18*cse221*state.xn[15];
  ydot(22) = -cse101 - cse117 - cse14*cse223 - cse15*state.xn[15] - cse222*cse39 - cse51*state.xn[14] - cse81*state.xn[12];
  ydot(23) = -cse100*cse224 - cse116*cse224 - cse222*cse224*cse37 - cse223*cse224 - 1.913494189605312e-17*cse224*state.xn[12] - 1.168047934563505e-18*cse224*state.xn[14] - 4.1730320353928637e-18*cse224*state.xn[15];
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


  // $JAFF REPEAT idx, expr, cse IN jacobian $[REPLACE radeden\[\s*(\d+)\s*\] state.rn[2*\1] REPLACE photden\[\s*(\d+)\s*\] state.rn[2*\1] REPLACE rflux\[\s*(\d+)\s*\] state.rn[2*\1+1] REPLACE nden\[\s*(\d+)\s*\] state.xn[\1] REPLACE tgas state.T REPLACE c_hat state.c_hat RADIATION True USE_DEDT True]$
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
  const amrex::Real cse64 = crate*(1.5*cse11*state.xn[0] + 2.2999999999999998*cse11*state.xn[3]);
  const amrex::Real cse65 = 1.0/state.xn[2];
  const amrex::Real cse66 = cse2*cse65;
  const amrex::Real cse67 = 119.95506989726393*state.rn[2*0] + 37.314487691080771*state.rn[2*1];
  const amrex::Real cse68 = state.c_hat*cse67;
  const amrex::Real cse69 = cse66*cse68;
  const amrex::Real cse70 = 5.670589618368584e-11*cse69;
  const amrex::Real cse71 = std::log(state.T);
  const amrex::Real cse72 = 1.102e-5*cse71 + 0.4723;
  const amrex::Real cse73 = std::pow(cse70, -cse72);
  const amrex::Real cse74 = cse73*std::pow(state.T, 0.015859999999999999);
  const amrex::Real cse75 = 508.69999999999999*cse74;
  const amrex::Real cse76 = cse75 + 1;
  const amrex::Real cse77 = std::pow(cse69, 1.3779999999999999);
  const amrex::Real cse78 = 6.1318063394789369e-20*cse77;
  const amrex::Real cse79 = cse76*cse78 + 1;
  const amrex::Real cse80 = 1.0/cse79;
  const amrex::Real cse81 = 1.2250000000000001e-11*cse80*d2g;
  const amrex::Real cse82 = cse81*state.xn[1];
  const amrex::Real cse83 = -cse82;
  const amrex::Real cse84 = 2.2999999999999998*cse36 + 1.5*cse38;
  const amrex::Real cse85 = 1.5*cse11 - cse84;
  const amrex::Real cse86 = crate*state.xn[0];
  const amrex::Real cse87 = cse85*cse86;
  const amrex::Real cse88 = std::log(8.6173000000000003e-5*state.T);
  const amrex::Real cse89 = std::pow(cse88, 2);
  const amrex::Real cse90 = std::pow(cse88, 3);
  const amrex::Real cse91 = std::pow(cse88, 4);
  const amrex::Real cse92 = std::pow(cse88, 5);
  const amrex::Real cse93 = std::pow(cse88, 6);
  const amrex::Real cse94 = std::pow(cse88, 7);
  const amrex::Real cse95 = std::exp(-2.0391498499999999e-6*std::pow(cse88, 8) - 5.7393287500000003*cse89 + 1.56315498*cse90 - 0.28770560000000001*cse91 + 0.034825597700000002*cse92 - 0.00263197617*cse93 + 0.000111954395*cse94);
  const amrex::Real cse96 = 5.9078447946382801e-70*cse95*std::pow(state.T, 13.536555999999999);
  const amrex::Real cse97 = ((cse28) ? (
   cse96
)
: (
   0.0
));
  const amrex::Real cse98 = cse97*state.xn[2];
  const amrex::Real cse99 = cse61 + cse63 + cse64 + cse83 + cse87 + cse98;
  const amrex::Real cse100 = std::exp(-227*cse3);
  const amrex::Real cse101 = cse100*(4.0000000000000001e-10*std::pow(state.T, 0.0066899999999999998) + 1.1000000000000001e-11*std::pow(state.T, 0.51700000000000002));
  const amrex::Real cse102 = cse101*state.xn[11];
  const amrex::Real cse103 = std::pow(cse3, 1.5);
  const amrex::Real cse104 = 114.81344434488582*std::pow(cse3, 0.40699999999999997) + 1;
  const amrex::Real cse105 = std::pow(cse104, -2.242);
  const amrex::Real cse106 = 4.8813573300130879e-6*cse103*cse105;
  const amrex::Real cse107 = -cse84;
  const amrex::Real cse108 = cse107*cse86;
  const amrex::Real cse109 = -cse108 + cse82;
  const amrex::Real cse110 = cse10*cse81 + cse106*state.xn[2] + cse109;
  const amrex::Real cse111 = cse102 + cse110;
  const amrex::Real cse112 = cse10*((cse20) ? (
   0
)
: (
   -cse40
));
  const amrex::Real cse113 = cse112*cse24;
  const amrex::Real cse114 = cse50*(-cse112 - cse21);
  const amrex::Real cse115 = cse114*cse22;
  const amrex::Real cse116 = cse30*((cse28) ? (
   cse54*(cse113*cse31 + cse114*cse49 + cse115*cse31 + cse33)
)
: (
   0
));
  const amrex::Real cse117 = -cse57;
  const amrex::Real cse118 = ((cse28) ? (
   cse59*(cse113*cse25 + cse114*cse6 + cse115*cse25 + cse58)
)
: (
   0
));
  const amrex::Real cse119 = cse9*state.xn[0];
  const amrex::Real cse120 = 2*cse116 + cse117 + cse118*cse119;
  const amrex::Real cse121 = std::pow(state.xn[2], 2);
  const amrex::Real cse122 = 1.0/cse121;
  const amrex::Real cse123 = 0.00011320754716981131*cse66;
  const amrex::Real cse124 = cse123*state.xn[3];
  const amrex::Real cse125 = std::min(10000000000.0, cse124);
  const amrex::Real cse126 = cse125 + 1;
  const amrex::Real cse127 = std::pow(cse126, -2);
  const amrex::Real cse128 = (((cse124 - 10000000000.0 > 0) ? (
   0
)
: ((cse124 - 10000000000.0 == 0) ? (
   1.0/2.0
)
: (
   1
))));
  const amrex::Real cse129 = 2.2528301886792451e-13*std::pow(state.T, 0.31);
  const amrex::Real cse130 = cse122*cse127*cse128*cse129*state.xn[11]*state.xn[16]*state.xn[3];
  const amrex::Real cse131 = std::pow(state.T, -0.52000000000000002);
  const amrex::Real cse132 = 2.9935562710997168e-6*cse131;
  const amrex::Real cse133 = cse122*cse2;
  const amrex::Real cse134 = cse128*state.xn[17];
  const amrex::Real cse135 = cse134*cse30;
  const amrex::Real cse136 = cse133*cse135;
  const amrex::Real cse137 = 3.6226415094339619e-13*cse127;
  const amrex::Real cse138 = 1.0/cse126;
  const amrex::Real cse139 = 1.811320754716981e-13*cse138;
  const amrex::Real cse140 = cse136*cse139;
  const amrex::Real cse141 = 1.811320754716981e-13*cse127;
  const amrex::Real cse142 = cse136*cse141;
  const amrex::Real cse143 = cse125*cse142;
  const amrex::Real cse144 = -cse140 + cse143;
  const amrex::Real cse145 = 1.05849862170431e-5*std::pow(state.T, -0.64000000000000001);
  const amrex::Real cse146 = cse145*state.xn[10];
  const amrex::Real cse147 = cse97*state.xn[0];
  const amrex::Real cse148 = cse76*cse77;
  const amrex::Real cse149 = -cse72;
  const amrex::Real cse150 = cse149*cse74*cse77;
  const amrex::Real cse151 = d2g*state.xn[1];
  const amrex::Real cse152 = 1.2250000000000001e-11*cse10*cse151/std::pow(cse79, 2);
  const amrex::Real cse153 = cse106*state.xn[1] - cse147 + cse152*(8.4496291358019747e-20*cse148*cse65 + 3.1192498848929353e-17*cse150*cse65);
  const amrex::Real cse154 = cse146 + cse153;
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
  const amrex::Real cse177 = 3.2000000000000001e-9*cse138;
  const amrex::Real cse178 = cse24*cse25;
  const amrex::Real cse179 = cse166*cse178;
  const amrex::Real cse180 = ((cse28) ? (
   cse59*(cse171*cse178 + cse172*cse6 + cse173*cse25 + cse179)
)
: (
   0
));
  const amrex::Real cse181 = state.c_hat*state.rn[2*1];
  const amrex::Real cse182 = std::max(1.0e-35, 2.0000000000000002e-15*ncol_H2);
  const amrex::Real cse183 = 0.035000000000000003*std::pow(cse182 + 1, -0.5)*std::exp(-0.00084999999999999995*std::sqrt(cse182 + 1.0)) + 0.96499999999999997*std::pow(cse182/std::max(9.9999999999999995e-21, 1.4142135623730953e-5*vdisp) + 1.0, -2.0);
  const amrex::Real cse184 = 7.3498720606844327e-18*cse183;
  const amrex::Real cse185 = state.c_hat*state.rn[2*0];
  const amrex::Real cse186 = 2.4433717132788471e-27*cse183;
  const amrex::Real cse187 = std::exp(-23*cse3);
  const amrex::Real cse188 = cse187*std::pow(state.T, -1.3);
  const amrex::Real cse189 = 4.2899999999999994e-13*cse188;
  const amrex::Real cse190 = cse134*cse66*state.xn[3];
  const amrex::Real cse191 = 1.6000000000000001e-9*cse138;
  const amrex::Real cse192 = cse191*state.xn[17];
  const amrex::Real cse193 = cse141*cse190;
  const amrex::Real cse194 = cse125*cse192 - cse125*cse193 + cse139*cse190;
  const amrex::Real cse195 = cse151*cse80;
  const amrex::Real cse196 = 2.4500000000000003e-11*cse195;
  const amrex::Real cse197 = 4.5999999999999996*cse36 + 3.0*cse38;
  const amrex::Real cse198 = 2.2999999999999998*cse11 - cse197;
  const amrex::Real cse199 = cse198*cse86;
  const amrex::Real cse200 = cse127*cse128*cse65*state.xn[11]*state.xn[16];
  const amrex::Real cse201 = cse129*cse200;
  const amrex::Real cse202 = cse196 - cse199 - cse201;
  const amrex::Real cse203 = cse10*((cse20) ? (
   0
)
: (
   -cse170
));
  const amrex::Real cse204 = cse50*(-cse166 - cse203);
  const amrex::Real cse205 = cse204*cse22;
  const amrex::Real cse206 = cse30*((cse28) ? (
   cse54*(cse167*cse203 + cse168 + cse204*cse49 + cse205*cse31)
)
: (
   0
));
  const amrex::Real cse207 = ((cse28) ? (
   cse59*(cse178*cse203 + cse179 + cse204*cse6 + cse205*cse25)
)
: (
   0
));
  const amrex::Real cse208 = cse158*cse161*state.xn[3];
  const amrex::Real cse209 = 6.3999999999999996e-10*state.xn[0];
  const amrex::Real cse210 = cse208 - cse209;
  const amrex::Real cse211 = -cse197;
  const amrex::Real cse212 = cse211*cse86;
  const amrex::Real cse213 = cse196 - cse212;
  const amrex::Real cse214 = cse156*state.xn[3];
  const amrex::Real cse215 = 26.218500000000002*std::sqrt(3)*std::sqrt(cse3) + 0.62;
  const amrex::Real cse216 = 1.3500000000000001e-9*cse215;
  const amrex::Real cse217 = cse216*state.xn[15];
  const amrex::Real cse218 = cse214 + cse217;
  const amrex::Real cse219 = std::exp(0.11*cse3);
  const amrex::Real cse220 = cse219*state.xn[15];
  const amrex::Real cse221 = 7.9513696852843427e-10*std::pow(state.T, -0.33900000000000002);
  const amrex::Real cse222 = cse220*cse221;
  const amrex::Real cse223 = 6.5199999999999996*crate;
  const amrex::Real cse224 = cse145*state.xn[2];
  const amrex::Real cse225 = cse109 + cse224;
  const amrex::Real cse226 = cse138*state.xn[16];
  const amrex::Real cse227 = 1.99e-9*std::pow(state.T, -0.19);
  const amrex::Real cse228 = cse226*cse227;
  const amrex::Real cse229 = 3.5000000000000002e-11*state.xn[15];
  const amrex::Real cse230 = 7.0000000000000004e-11*state.xn[14];
  const amrex::Real cse231 = cse101*state.xn[1];
  const amrex::Real cse232 = cse230 + cse231;
  const amrex::Real cse233 = 7.0000000000000004e-11*state.xn[11];
  const amrex::Real cse234 = 4.2188032355736621e-18*cse185;
  const amrex::Real cse235 = 1.168047934563505e-18*cse181;
  const amrex::Real cse236 = cse0*state.xn[0];
  const amrex::Real cse237 = -cse236;
  const amrex::Real cse238 = cse109 + cse120;
  const amrex::Real cse239 = cse216*state.xn[6];
  const amrex::Real cse240 = cse219*cse221*state.xn[7];
  const amrex::Real cse241 = 1.7141251340095381e-18*cse185;
  const amrex::Real cse242 = 4.1730320353928637e-18*cse181;
  const amrex::Real cse243 = cse241 + cse242 + 3.5000000000000002e-11*state.xn[11];
  const amrex::Real cse244 = cse239 + cse240 + cse243;
  const amrex::Real cse245 = 3*cse21;
  const amrex::Real cse246 = cse10*((cse20) ? (
   0
)
: (
   -3*cse37 - 3*cse39
));
  const amrex::Real cse247 = cse50*(-cse245 - cse246);
  const amrex::Real cse248 = cse22*cse247;
  const amrex::Real cse249 = cse30*((cse28) ? (
   cse54*(cse167*cse245 + cse167*cse246 + cse247*cse49 + cse248*cse31)
)
: (
   0
));
  const amrex::Real cse250 = ((cse28) ? (
   cse59*(cse178*cse245 + cse178*cse246 + cse247*cse6 + cse248*cse25)
)
: (
   0
));
  const amrex::Real cse251 = cse227*state.xn[11];
  const amrex::Real cse252 = cse138*cse251;
  const amrex::Real cse253 = -6.8999999999999995*cse36 - 4.5*cse38;
  const amrex::Real cse254 = cse253*cse86;
  const amrex::Real cse255 = 3.6750000000000002e-11*cse195;
  const amrex::Real cse256 = cse252 - cse254 + cse255;
  const amrex::Real cse257 = cse62*state.xn[0];
  const amrex::Real cse258 = cse191*state.xn[3];
  const amrex::Real cse259 = cse125*cse258;
  const amrex::Real cse260 = std::pow(state.T, -2);
  const amrex::Real cse261 = 2.8349999999999999e-12*cse155*cse260*state.xn[3]*state.xn[6];
  const amrex::Real cse262 = -cse261;
  const amrex::Real cse263 = 7.3171760954257497e-12*std::pow(state.T, -0.73999999999999999)*state.xn[0]*state.xn[14];
  const amrex::Real cse264 = std::sqrt(cse3);
  const amrex::Real cse265 = std::sqrt(3)*cse264*cse3*state.xn[15];
  const amrex::Real cse266 = 1.7697487500000004e-8*cse265*state.xn[6];
  const amrex::Real cse267 = cse26*cse260;
  const amrex::Real cse268 = cse13*cse3;
  const amrex::Real cse269 = cse12*cse268;
  const amrex::Real cse270 = cse3*cse8;
  const amrex::Real cse271 = ((cse20) ? (
   0
)
: (
   cse16*cse7*(-3.2400000000000002*cse269 + 1.3*cse3*cse8) + cse18*cse7*(0.65400000000000003*cse269 + 0.41599999999999998*cse270)
));
  const amrex::Real cse272 = cse10*cse271;
  const amrex::Real cse273 = cse272*cse50;
  const amrex::Real cse274 = cse21*cse271*cse34*cse50;
  const amrex::Real cse275 = ((cse28) ? (
   cse54*(cse167*cse272 + 1.6677785190126752e+29*cse24*cse45*(-2.3058963562359999e-34*cse43*std::pow(cse44, -6.6881000000000004)*cse47 + 3.2772577039999999e-25*cse48*std::pow(state.T, 2.1881000000000004) + 2.51118476e-29*cse48*std::pow(state.T, 3.1881000000000004))*std::exp(cse46)/cse43 + 53300.0*cse267 - cse273*cse49 - cse274*cse31)
)
: (
   0
));
  const amrex::Real cse276 = state.xn[0]*state.xn[17]*(-3.435e-10*std::pow(state.T, -1.458) + 2.0209499999999999e-11*std::pow(state.T, -0.59499999999999997));
  const amrex::Real cse277 = 1.0/cse2;
  const amrex::Real cse278 = std::pow(state.T, -3.0/2.0);
  const amrex::Real cse279 = ((cse28) ? (
   cse59*(cse178*cse272 + 149925037481.25937*cse24*cse277*(3.3349999999999998e-12*cse277*cse5 + 4.2414529999999999e-7*cse278*cse5)*std::exp(cse4) - cse25*cse274 + 43900.0*cse267 - cse273*cse6)
)
: (
   0
));
  const amrex::Real cse280 = std::pow(state.T, -1.52)*state.xn[16]*state.xn[2];
  const amrex::Real cse281 = cse100*state.xn[1]*state.xn[11]*(2.6759999999999999e-12*std::pow(state.T, -0.99331000000000003) + 5.6870000000000003e-12*std::pow(state.T, -0.48299999999999998));
  const amrex::Real cse282 = std::pow(state.T, -2.2999999999999998);
  const amrex::Real cse283 = cse187*state.xn[3]*state.xn[8];
  const amrex::Real cse284 = cse282*cse283;
  const amrex::Real cse285 = cse159*state.xn[3];
  const amrex::Real cse286 = 3.7768240343347639e-14*cse160*cse285;
  const amrex::Real cse287 = 7.3920000000000002e-11*cse285*std::pow(state.T, -0.95799999999999996);
  const amrex::Real cse288 = std::pow(state.T, -3.2999999999999998);
  const amrex::Real cse289 = 227*cse102*cse260*state.xn[1];
  const amrex::Real cse290 = cse277*cse65;
  const amrex::Real cse291 = cse135*cse290;
  const amrex::Real cse292 = 6.7743911789075847e-6*std::pow(state.T, -1.6400000000000001)*state.xn[10]*state.xn[2];
  const amrex::Real cse293 = cse220*state.xn[7];
  const amrex::Real cse294 = 8.7465066538127767e-11*cse293*std::pow(state.T, -2.339) + 2.6955143233113923e-10*cse293*std::pow(state.T, -1.339);
  const amrex::Real cse295 = cse292 + cse294;
  const amrex::Real cse296 = state.xn[0]*state.xn[2];
  const amrex::Real cse297 = cse296*((cse28) ? (
   7.9971871901929577e-69*cse95*std::pow(state.T, 12.536555999999999) + cse96*(-11.478657500000001*cse3*cse88 + 4.6894649399999997*cse3*cse89 - 1.1508224*cse3*cse90 + 0.1741279885*cse3*cse91 - 0.015791857020000001*cse3*cse92 + 0.00078368076500000001*cse3*cse93 - 1.6313198799999999e-5*cse3*cse94)
)
: (
   0
));
  const amrex::Real cse298 = cse3*state.xn[1]*state.xn[2];
  const amrex::Real cse299 = 7.3220359950196314e-6*cse103*cse105*cse298;
  const amrex::Real cse300 = 0.0005114031087522854*std::pow(cse104, -3.242)*cse298*std::pow(cse3, 1.907);
  const amrex::Real cse301 = 0.5*cse3;
  const amrex::Real cse302 = cse3*std::log(cse70);
  const amrex::Real cse303 = cse152*(-4.2248145679009874e-20*cse148*cse3 - cse78*(8.0679819999999989*cse73*std::pow(state.T, -0.98414000000000001) + cse75*(cse149*cse301 - 1.102e-5*cse302)));
  const amrex::Real cse304 = 9.0566037735849048e-14*cse291;
  const amrex::Real cse305 = cse138*cse304;
  const amrex::Real cse306 = cse127*cse304;
  const amrex::Real cse307 = cse125*cse306;
  const amrex::Real cse308 = -cse305 + cse307;
  const amrex::Real cse309 = 3.7810000000000002e-10*std::pow(state.T, -1.1899999999999999)*state.xn[11];
  const amrex::Real cse310 = 1.1264150943396225e-13*cse200*std::pow(state.T, -0.68999999999999995)*state.xn[3] + cse226*cse309;
  const amrex::Real cse311 = cse297 + cse299 - cse300 - cse303 + cse308 + cse310;
  const amrex::Real cse312 = 0.66666666666670005*(1.6737729999999998e-24*state.xn[0] + 1.6728620616289998e-24*state.xn[1] + 9.1093837099999994e-28*state.xn[2] + 3.3475459999999996e-24*state.xn[3] + 3.3466350616289996e-24*state.xn[4] + 6.6464729999999996e-24*state.xn[5] + 6.6455620616289996e-24*state.xn[6] + 1.9944730000000003e-23*state.xn[7] + 1.9943819061629003e-23*state.xn[8] + 4.6512360000000008e-23*state.xn[9] + 4.8185222061629005e-23*state.xn[10] + 2.6567630000000005e-23*state.xn[11] + 4.6637070000000001e-23*state.xn[12] + 4.6636159061629004e-23*state.xn[13] + 2.1618503000000003e-23*state.xn[14] + 2.8241403000000005e-23*state.xn[15] + 5.0204080616289998e-24*state.xn[16] + 2.6566719061629008e-23*state.xn[17])/(1.380649e-16*state.xn[0] + 1.380649e-16*state.xn[1] + 1.380649e-16*state.xn[2] + 1.380649e-16*state.xn[3] + 1.380649e-16*state.xn[4] + 1.380649e-16*state.xn[5] + 1.380649e-16*state.xn[6] + 1.380649e-16*state.xn[7] + 1.380649e-16*state.xn[8] + 1.380649e-16*state.xn[9] + 1.380649e-16*state.xn[10] + 1.380649e-16*state.xn[11] + 1.380649e-16*state.xn[12] + 1.380649e-16*state.xn[13] + 1.380649e-16*state.xn[14] + 1.380649e-16*state.xn[15] + 1.380649e-16*state.xn[16] + 1.380649e-16*state.xn[17]);
  const amrex::Real cse313 = state.c_hat*state.xn[3];
  const amrex::Real cse314 = 1.0/cse67;
  const amrex::Real cse315 = cse148*cse314;
  const amrex::Real cse316 = cse150*cse314;
  const amrex::Real cse317 = cse152*(-1.0135758535910836e-17*cse315 - 3.7416983796936446e-15*cse316);
  const amrex::Real cse318 = state.c_hat*state.xn[14];
  const amrex::Real cse319 = 4.2188032355736621e-18*cse318;
  const amrex::Real cse320 = state.c_hat*state.xn[15];
  const amrex::Real cse321 = 1.7141251340095381e-18*cse320;
  const amrex::Real cse322 = cse319 + cse321;
  const amrex::Real cse323 = 1.168047934563505e-18*cse318;
  const amrex::Real cse324 = 4.1730320353928637e-18*cse320;
  const amrex::Real cse325 = cse152*(-3.1529358238208022e-18*cse315 - 1.1639321143524254e-15*cse316);
  const amrex::Real cse326 = -cse196;
  const amrex::Real cse327 = cse108 + cse83;
  const amrex::Real cse328 = -cse276 + cse281 + cse289;
  const amrex::Real cse329 = crate*cse9;
  const amrex::Real cse330 = cse329*cse85;
  const amrex::Real cse331 = 5.4939999999999998e-7*cse71 + 0.49559999999999998;
  const amrex::Real cse332 = std::pow(cse70, -cse331);
  const amrex::Real cse333 = cse332*std::pow(state.T, 3.9029999999999999e-7);
  const amrex::Real cse334 = 5115.0*cse333;
  const amrex::Real cse335 = cse334 + 1;
  const amrex::Real cse336 = std::pow(cse69, 1.512);
  const amrex::Real cse337 = 1.0246966183388124e-22*cse336;
  const amrex::Real cse338 = cse335*cse337 + 1;
  const amrex::Real cse339 = 1.0/cse338;
  const amrex::Real cse340 = cse339*d2g*state.xn[6];
  const amrex::Real cse341 = 5.5720000000000001e-12*cse340;
  const amrex::Real cse342 = crate*cse36;
  const amrex::Real cse343 = cse342*state.xn[7];
  const amrex::Real cse344 = 0.00013329999999999999*cse71 + 0.81200000000000006;
  const amrex::Real cse345 = std::pow(cse70, -cse344);
  const amrex::Real cse346 = cse345*std::pow(state.T, 0.04845);
  const amrex::Real cse347 = 433.10000000000002*cse346;
  const amrex::Real cse348 = cse347 + 1;
  const amrex::Real cse349 = std::pow(cse69, 1.1279999999999999);
  const amrex::Real cse350 = 1.6851501542381549e-14*cse349;
  const amrex::Real cse351 = cse348*cse350 + 1;
  const amrex::Real cse352 = 1.0/cse351;
  const amrex::Real cse353 = cse352*d2g*state.xn[8];
  const amrex::Real cse354 = 1040.0*cse343 + 4.5579999999999993e-11*cse353;
  const amrex::Real cse355 = cse342*state.xn[12];
  const amrex::Real cse356 = 7.538e-5*cse71 + 0.89639999999999997;
  const amrex::Real cse357 = std::pow(cse70, -cse356);
  const amrex::Real cse358 = cse357*std::pow(state.T, 1.635e-6);
  const amrex::Real cse359 = 43750.0*cse358;
  const amrex::Real cse360 = cse359 + 1;
  const amrex::Real cse361 = std::pow(cse69, 1.8740000000000001);
  const amrex::Real cse362 = 3.5685736926834518e-27*cse361;
  const amrex::Real cse363 = cse360*cse362 + 1;
  const amrex::Real cse364 = 1.0/cse363;
  const amrex::Real cse365 = cse364*d2g*state.xn[13];
  const amrex::Real cse366 = 16800.0*cse355 + 2.1659999999999999e-12*cse365;
  const amrex::Real cse367 = cse341 + cse354 + cse366;
  const amrex::Real cse368 = cse107*cse329;
  const amrex::Real cse369 = cse367 - cse368;
  const amrex::Real cse370 = cse71*cse8;
  const amrex::Real cse371 = 0.044330000000000001*cse370 - 0.28520000000000001;
  const amrex::Real cse372 = cse370*cse371 - 1.6759999999999999;
  const amrex::Real cse373 = cse370*cse372 + 11.19;
  const amrex::Real cse374 = cse373*state.xn[6];
  const amrex::Real cse375 = 9.9999999999999994e-12*std::pow(state.T, -0.5);
  const amrex::Real cse376 = cse335*cse336;
  const amrex::Real cse377 = -cse331;
  const amrex::Real cse378 = cse333*cse336*cse377;
  const amrex::Real cse379 = cse10*d2g;
  const amrex::Real cse380 = 5.5720000000000001e-12*cse379;
  const amrex::Real cse381 = cse380*state.xn[6]/std::pow(cse338, 2);
  const amrex::Real cse382 = cse374*cse375 + cse381*(1.5493412869282843e-22*cse376*cse65 + 5.2413232028030254e-19*cse378*cse65);
  const amrex::Real cse383 = std::exp(-15230.0*cse3);
  const amrex::Real cse384 = std::exp(-73.799999999999997*cse3);
  const amrex::Real cse385 = std::exp(-12.17*cse3);
  const amrex::Real cse386 = 1.6339999999999999e-6*cse383 + 9.7930000000000004e-9*cse384 + 6.3460000000000002e-9*cse385;
  const amrex::Real cse387 = 0.00071740367932898903*cse2 + 1;
  const amrex::Real cse388 = std::exp(-49550.0*cse3);
  const amrex::Real cse389 = 0.15970000000000001*cse388;
  const amrex::Real cse390 = cse389 + 1.7848999999999999;
  const amrex::Real cse391 = 12.244387999457523*cse2 + 1;
  const amrex::Real cse392 = 0.21509999999999996 - cse389;
  const amrex::Real cse393 = std::pow(cse387, -cse390)*std::pow(cse391, -cse392);
  const amrex::Real cse394 = 2.4460185352936309e-10*cse277*cse393;
  const amrex::Real cse395 = cse278*cse386 + cse394;
  const amrex::Real cse396 = cse348*cse349;
  const amrex::Real cse397 = -cse344;
  const amrex::Real cse398 = cse346*cse349*cse397;
  const amrex::Real cse399 = 4.5579999999999993e-11*cse379;
  const amrex::Real cse400 = cse399*state.xn[8]/std::pow(cse351, 2);
  const amrex::Real cse401 = cse395*state.xn[8] + cse400*(1.9008493739806384e-14*cse396*cse65 + 7.2983853180054489e-12*cse398*cse65);
  const amrex::Real cse402 = 1.4595000290776864e-10*std::pow(state.T, -0.62);
  const amrex::Real cse403 = cse360*cse361;
  const amrex::Real cse404 = -cse356;
  const amrex::Real cse405 = cse358*cse361*cse404;
  const amrex::Real cse406 = 2.1659999999999999e-12*cse379;
  const amrex::Real cse407 = cse406*state.xn[13]/std::pow(cse363, 2);
  const amrex::Real cse408 = cse402*state.xn[13] + cse407*(6.687507100088789e-27*cse403*cse65 + 1.5612509905490101e-22*cse405*cse65);
  const amrex::Real cse409 = cse142 + cse144;
  const amrex::Real cse410 = 1.3007021411393064e-6*cse131;
  const amrex::Real cse411 = cse11*state.xn[2];
  const amrex::Real cse412 = cse411 < 1.0e-50;
  const amrex::Real cse413 = ((cse412) ? (
   0
)
: (
   -0.00011320754716981131*cse133*state.xn[3]
));
  const amrex::Real cse414 = ((cse412) ? (
   10000000000.0
)
: (
   cse124
));
  const amrex::Real cse415 = cse414 + 1;
  const amrex::Real cse416 = 1.0/cse415;
  const amrex::Real cse417 = cse251*state.xn[16];
  const amrex::Real cse418 = cse416*cse417;
  const amrex::Real cse419 = cse413*cse418;
  const amrex::Real cse420 = cse414*cse417/std::pow(cse415, 2);
  const amrex::Real cse421 = cse413*cse420;
  const amrex::Real cse422 = cse419 - cse421;
  const amrex::Real cse423 = cse130 + cse422;
  const amrex::Real cse424 = cse410*state.xn[16] + cse423;
  const amrex::Real cse425 = 1.1144e-11*cse340;
  const amrex::Real cse426 = 2*cse64;
  const amrex::Real cse427 = cse198*cse329;
  const amrex::Real cse428 = crate*cse11;
  const amrex::Real cse429 = 16800.0*cse428;
  const amrex::Real cse430 = 33600.0*cse355 + 4.3319999999999997e-12*cse365;
  const amrex::Real cse431 = -cse429*state.xn[12] + cse430;
  const amrex::Real cse432 = 3.2999999999999996e-13*cse188;
  const amrex::Real cse433 = cse432*state.xn[8];
  const amrex::Real cse434 = 1040.0*cse428;
  const amrex::Real cse435 = -cse434*state.xn[7];
  const amrex::Real cse436 = 2080.0*cse343 + 9.1159999999999986e-11*cse353;
  const amrex::Real cse437 = cse433 + cse435 + cse436;
  const amrex::Real cse438 = ((cse412) ? (
   0
)
: (
   cse123
));
  const amrex::Real cse439 = cse418*cse438;
  const amrex::Real cse440 = cse420*cse438;
  const amrex::Real cse441 = cse439 - cse440;
  const amrex::Real cse442 = cse194 + cse441;
  const amrex::Real cse443 = cse192 - cse193;
  const amrex::Real cse444 = cse211*cse329;
  const amrex::Real cse445 = -cse444;
  const amrex::Real cse446 = 1.1000000000000001*crate;
  const amrex::Real cse447 = cse375*state.xn[2];
  const amrex::Real cse448 = cse339*cse380 + cse373*cse447;
  const amrex::Real cse449 = std::pow(cse3, 0.00231);
  const amrex::Real cse450 = std::pow(state.T, -1.5);
  const amrex::Real cse451 = std::exp(-8110.0*cse3);
  const amrex::Real cse452 = std::exp(-26.600000000000001*cse3);
  const amrex::Real cse453 = std::exp(-7.6200000000000001*cse3);
  const amrex::Real cse454 = std::exp(-1.3799999999999999*cse3);
  const amrex::Real cse455 = 0.00015100000000000001*cse451 + 1.31e-7*cse452 + 3.4e-8*cse453 + 6.9699999999999997e-9*cse454;
  const amrex::Real cse456 = 1.0537934368240927e-9*cse449 + cse450*cse455;
  const amrex::Real cse457 = cse456*state.xn[16];
  const amrex::Real cse458 = -cse457;
  const amrex::Real cse459 = 1.6927240490436929e-17*interp_8_shielding_H2(ncol_H2)*interp_8_shielding_self(ncol_C);
  const amrex::Real cse460 = cse181*cse459;
  const amrex::Real cse461 = 3.8500000000000001*crate + cse434*state.xn[3] + cse460;
  const amrex::Real cse462 = cse432*state.xn[3];
  const amrex::Real cse463 = cse352*cse399 + cse395*state.xn[2];
  const amrex::Real cse464 = cse462 + cse463;
  const amrex::Real cse465 = cse414*cse416;
  const amrex::Real cse466 = cse465*state.xn[16];
  const amrex::Real cse467 = cse227*cse466;
  const amrex::Real cse468 = cse228 + cse467;
  const amrex::Real cse469 = -cse468;
  const amrex::Real cse470 = 2.1635567548454671e-17*cse185;
  const amrex::Real cse471 = 1.913494189605312e-17*cse181;
  const amrex::Real cse472 = cse429*state.xn[3] + cse470 + cse471;
  const amrex::Real cse473 = cse364*cse406 + cse402*state.xn[2];
  const amrex::Real cse474 = -cse473;
  const amrex::Real cse475 = -cse109 - cse369;
  const amrex::Real cse476 = cse251*cse465;
  const amrex::Real cse477 = cse456*state.xn[7];
  const amrex::Real cse478 = cse410*state.xn[2];
  const amrex::Real cse479 = cse253*cse329;
  const amrex::Real cse480 = -cse479;
  const amrex::Real cse481 = 1.6715999999999999e-11*cse340;
  const amrex::Real cse482 = 3120.0*cse343 + 1.3673999999999998e-10*cse353;
  const amrex::Real cse483 = 50400.0*cse355 + 6.4979999999999992e-12*cse365;
  const amrex::Real cse484 = cse258 + cse259;
  const amrex::Real cse485 = -cse484;
  const amrex::Real cse486 = state.xn[16]*state.xn[7]*(-2.4342628390636543e-12*cse3*cse449 + cse450*(1.2246100000000002*cse260*cse451 + 3.4846000000000001e-6*cse260*cse452 + 2.5908000000000001e-7*cse260*cse453 + 9.6185999999999982e-9*cse260*cse454) - 1.5*cse455*std::pow(state.T, -2.5));
  const amrex::Real cse487 = -cse486;
  const amrex::Real cse488 = 6.7636511339243933e-7*cse280;
  const amrex::Real cse489 = cse309*cse466;
  const amrex::Real cse490 = ((cse412) ? (
   0
)
: (
   5.6603773584905654e-5*cse290*state.xn[3]
));
  const amrex::Real cse491 = cse418*cse490;
  const amrex::Real cse492 = -cse491;
  const amrex::Real cse493 = cse420*cse490;
  const amrex::Real cse494 = 7913.1350000000002*cse260*cse388;
  const amrex::Real cse495 = state.xn[2]*state.xn[8]*(-1.2230092676468155e-10*cse278*cse393 + cse278*(0.024885819999999999*cse260*cse383 + 7.2272340000000005e-7*cse260*cse384 + 7.7230820000000007e-8*cse260*cse385) - 3.0/2.0*cse386/std::pow(state.T, 5.0/2.0) + cse394*(-0.00035870183966449452*cse277*cse390/cse387 - cse494*std::log(cse387)) + cse394*(-6.1221939997287613*cse277*cse392/cse391 + cse494*std::log(cse391)));
  const amrex::Real cse496 = 9.0489001802816564e-11*std::pow(state.T, -1.6200000000000001)*state.xn[13]*state.xn[2];
  const amrex::Real cse497 = 4.9999999999999997e-12*cse374*cse450*state.xn[2];
  const amrex::Real cse498 = cse283*cse288;
  const amrex::Real cse499 = 7.5899999999999998e-12*cse498;
  const amrex::Real cse500 = cse447*state.xn[6]*(cse270*cse372 + cse370*(0.044330000000000001*cse268*cse71 + cse270*cse371));
  const amrex::Real cse501 = cse407*(-3.3437535500443945e-27*cse3*cse403 - cse362*(0.071531250000000005*cse357*std::pow(state.T, -0.99999836499999994) + cse359*(cse301*cse404 - 7.538e-5*cse302)));
  const amrex::Real cse502 = cse381*(-7.7467064346414215e-23*cse3*cse376 - cse337*(0.0019963845000000001*cse332*std::pow(state.T, -0.99999960970000001) + cse334*(cse301*cse377 - 5.4939999999999998e-7*cse302)));
  const amrex::Real cse503 = cse400*(-9.5042468699031921e-15*cse3*cse396 - cse350*(20.983695000000001*cse345*std::pow(state.T, -0.95155000000000001) + cse347*(cse301*cse397 - 0.00013329999999999999*cse302)));
  const amrex::Real cse504 = state.c_hat*state.xn[12];
  const amrex::Real cse505 = 2.1635567548454671e-17*cse504;
  const amrex::Real cse506 = cse314*cse403;
  const amrex::Real cse507 = cse314*cse405;
  const amrex::Real cse508 = cse407*(-8.0220038162959944e-25*cse506 - 1.8727997169847905e-20*cse507);
  const amrex::Real cse509 = cse314*cse376;
  const amrex::Real cse510 = cse314*cse378;
  const amrex::Real cse511 = cse381*(-1.858513423681992e-20*cse509 - 6.2872329114638811e-17*cse510);
  const amrex::Real cse512 = cse314*cse396;
  const amrex::Real cse513 = cse314*cse398;
  const amrex::Real cse514 = cse400*(-2.2801651952001787e-12*cse512 - 8.7547832095850848e-10*cse513);
  const amrex::Real cse515 = 1.913494189605312e-17*cse504;
  const amrex::Real cse516 = cse407*(-2.4954090137027837e-25*cse506 - 5.8257280869528704e-21*cse507);
  const amrex::Real cse517 = cse381*(-5.7812876380368708e-21*cse509 - 1.9557729013596953e-17*cse510);
  const amrex::Real cse518 = state.c_hat*cse459*state.xn[7];
  const amrex::Real cse519 = cse400*(-7.0929220567999134e-13*cse512 - 2.7233550911347897e-10*cse513);
  const amrex::Real cse520 = -cse518 + cse519;
  const amrex::Real cse521 = -3.0000000000000002e-15*d2g*state.xn[0];
  const amrex::Real cse522 = cse29*state.xn[3];
  const amrex::Real cse523 = cse30*cse55;
  const amrex::Real cse524 = -cse1;
  const amrex::Real cse525 = state.xn[0]*state.xn[3];
  const amrex::Real cse526 = cse525*cse60;
  const amrex::Real cse527 = cse330 - cse61;
  const amrex::Real cse528 = cse118*cse525;
  const amrex::Real cse529 = cse116 + cse368 + cse521 + cse528;
  const amrex::Real cse530 = -cse529;
  const amrex::Real cse531 = 4.5427507615910105e-7*cse131;
  const amrex::Real cse532 = cse180*cse525;
  const amrex::Real cse533 = 1.2216858566394236e-27*cse183;
  const amrex::Real cse534 = 3.6749360303422164e-18*cse183;
  const amrex::Real cse535 = cse181*cse534 + cse185*cse533;
  const amrex::Real cse536 = 7.2000000000000002e-15*state.xn[6];
  const amrex::Real cse537 = cse157 + cse536;
  const amrex::Real cse538 = cse426 + cse427;
  const amrex::Real cse539 = cse162 + cse201 - cse439 + cse440;
  const amrex::Real cse540 = cse194 + cse443;
  const amrex::Real cse541 = cse207*cse525;
  const amrex::Real cse542 = 7.2000000000000002e-15*state.xn[3];
  const amrex::Real cse543 = 1.6999999999999999e-9*state.xn[16];
  const amrex::Real cse544 = 9.0000000000000011e-15*cse163;
  const amrex::Real cse545 = cse250*cse525;
  const amrex::Real cse546 = 1.6999999999999999e-9*state.xn[9];
  const amrex::Real cse547 = cse252 + cse476 + cse477 + cse546;
  const amrex::Real cse548 = -cse263;
  const amrex::Real cse549 = cse275*cse30;
  const amrex::Real cse550 = cse279*cse525;
  const amrex::Real cse551 = -4.2899999999999994e-13*cse187*cse282*state.xn[3]*state.xn[8] + cse499;
  const amrex::Real cse552 = cse305 - cse307;
  const amrex::Real cse553 = -cse286 + cse287;
  const amrex::Real cse554 = cse310 + cse487 + cse489 + cse492 + cse493 + cse553;
  const amrex::Real cse555 = cse313*cse533;
  const amrex::Real cse556 = cse313*cse534;
  const amrex::Real cse557 = cse425 + cse537;
  const amrex::Real cse558 = 1.6000000000000001e-9*state.xn[9];
  const amrex::Real cse559 = cse218 + cse448 + cse542 + cse558;
  const amrex::Real cse560 = 1.6000000000000001e-9*state.xn[6];
  const amrex::Real cse561 = cse239 + cse341;
  const amrex::Real cse562 = -cse266;
  const amrex::Real cse563 = cse261 - cse497 + cse500 + cse502 + cse562;
  const amrex::Real cse564 = -cse341;
  const amrex::Real cse565 = cse342*state.xn[9];
  const amrex::Real cse566 = 184.0*cse565;
  const amrex::Real cse567 = -cse566;
  const amrex::Real cse568 = cse354 + cse567;
  const amrex::Real cse569 = 184.0*cse428;
  const amrex::Real cse570 = cse569*state.xn[9];
  const amrex::Real cse571 = 9.8999999999999995e-14*cse188;
  const amrex::Real cse572 = 368.0*cse565;
  const amrex::Real cse573 = -cse572;
  const amrex::Real cse574 = cse436 + cse573;
  const amrex::Real cse575 = 1.6797535700197549e-17*interp_10_shielding_H2(ncol_H2)*interp_10_shielding_self(ncol_CO);
  const amrex::Real cse576 = cse181*cse575;
  const amrex::Real cse577 = cse569*state.xn[3] + cse576;
  const amrex::Real cse578 = cse234 + cse235 + cse236;
  const amrex::Real cse579 = 552.0*cse565;
  const amrex::Real cse580 = cse495 + cse503;
  const amrex::Real cse581 = state.c_hat*cse575*state.xn[9];
  const amrex::Real cse582 = -cse354;
  const amrex::Real cse583 = 9.1500000000000005e-10*cse215;
  const amrex::Real cse584 = cse583*state.xn[15];
  const amrex::Real cse585 = cse583*state.xn[8];
  const amrex::Real cse586 = 1.1994963750000002e-8*cse265*state.xn[8];
  const amrex::Real cse587 = -cse586;
  const amrex::Real cse588 = -cse570 + cse572;
  const amrex::Real cse589 = cse223 + cse543;
  const amrex::Real cse590 = cse560 + cse577;
  const amrex::Real cse591 = cse233 + cse566;
  const amrex::Real cse592 = -cse63;
  const amrex::Real cse593 = -cse229 + cse467;
  const amrex::Real cse594 = -cse489 + cse491 - cse493;
  const amrex::Real cse595 = cse306 + cse328;
  const amrex::Real cse596 = cse324 + cse581;
  const amrex::Real cse597 = -cse496 + cse501;
  const amrex::Real cse598 = cse505 - cse508;
  const amrex::Real cse599 = cse515 - cse516;
  const amrex::Real cse600 = -cse366;
  const amrex::Real cse601 = 2.3099999999999997e-13*cse188;
  const amrex::Real cse602 = (1.0/100.0)*state.T;
  const amrex::Real cse603 = std::log(cse602);
  const amrex::Real cse604 = 0.0060000000000000001*cse603;
  const amrex::Real cse605 = std::pow(cse602, 0.36899999999999999 - cse604);
  const amrex::Real cse606 = std::pow(state.T, 0.92600000000000005);
  const amrex::Real cse607 = 1.08e-14*cse606;
  const amrex::Real cse608 = 0.16*cse603;
  const amrex::Real cse609 = std::pow(cse602, 0.755 - cse608);
  const amrex::Real cse610 = 4.34e-10*cse609;
  const amrex::Real cse611 = cse610*state.xn[0];
  const amrex::Real cse612 = 0.042999999999999997*cse603;
  const amrex::Real cse613 = 5.2499999999999998e-13*std::pow(cse602, cse612 + 0.88900000000000001);
  const amrex::Real cse614 = 0.52500000000000002*cse603;
  const amrex::Real cse615 = 2.2500000000000003e-12*std::pow(cse602, cse614 + 1.198);
  const amrex::Real cse616 = cse613 + cse615;
  const amrex::Real cse617 = cse607*state.xn[2] + cse611 + cse616*state.xn[3];
  const amrex::Real cse618 = std::exp(-425.40000000000003*cse3);
  const amrex::Real cse619 = cse617*cse618;
  const amrex::Real cse620 = std::exp(-326.60000000000002*cse3);
  const amrex::Real cse621 = std::pow(state.T, -0.025999999999999999);
  const amrex::Real cse622 = 4.8599999999999998e-10*cse621;
  const amrex::Real cse623 = 3.1899999999999998e-10*cse605;
  const amrex::Real cse624 = cse623*state.xn[0];
  const amrex::Real cse625 = 0.041000000000000002*cse603;
  const amrex::Real cse626 = 4.7499999999999998e-11*std::pow(cse602, cse625 + 0.20300000000000001);
  const amrex::Real cse627 = 0.058000000000000003*cse603;
  const amrex::Real cse628 = 1.6725000000000002e-10*std::pow(cse602, cse627 + 0.23699999999999999);
  const amrex::Real cse629 = cse626 + cse628;
  const amrex::Real cse630 = cse622*state.xn[2] + cse624 + cse629*state.xn[3];
  const amrex::Real cse631 = cse620*cse630;
  const amrex::Real cse632 = 0.0030000000000000001*cse603;
  const amrex::Real cse633 = std::pow(cse602, 0.41899999999999998 - cse632);
  const amrex::Real cse634 = 7.1400000000000007e-11*cse633;
  const amrex::Real cse635 = cse617*cse620;
  const amrex::Real cse636 = std::pow(state.T, -0.074999999999999997);
  const amrex::Real cse637 = 5.1199999999999999e-10*cse636;
  const amrex::Real cse638 = 3.5700000000000001e-10*cse633;
  const amrex::Real cse639 = cse638*state.xn[0];
  const amrex::Real cse640 = 0.025000000000000001*cse603;
  const amrex::Real cse641 = 3.7250000000000001e-11*std::pow(cse602, cse640 + 0.26400000000000001);
  const amrex::Real cse642 = 1.0274999999999999e-10*std::pow(cse602, cse612 + 0.29599999999999999);
  const amrex::Real cse643 = cse641 + cse642;
  const amrex::Real cse644 = cse637*state.xn[2] + cse639 + cse643*state.xn[3];
  const amrex::Real cse645 = cse620*cse644;
  const amrex::Real cse646 = 8.68e-11*cse609;
  const amrex::Real cse647 = cse618*cse630;
  const amrex::Real cse648 = cse644 + 8.9099999999999997e-5;
  const amrex::Real cse649 = cse620*cse648;
  const amrex::Real cse650 = 6.3800000000000002e-11*cse605;
  const amrex::Real cse651 = 2.1266666666666666e-11*cse605*cse619 + 2.8933333333333332e-11*cse609*cse647 + cse631*cse634 + cse634*cse635 + cse645*cse646 + cse649*cse650;
  const amrex::Real cse652 = cse617 + 1.7499999999999998e-5;
  const amrex::Real cse653 = cse630 + 1.34e-10;
  const amrex::Real cse654 = std::exp(-98.800000000000011*cse3);
  const amrex::Real cse655 = cse617*cse654;
  const amrex::Real cse656 = (1.0/3.0)*cse655;
  const amrex::Real cse657 = (1.0/5.0)*cse620;
  const amrex::Real cse658 = cse617*cse657;
  const amrex::Real cse659 = (1.0/15.0)*cse619;
  const amrex::Real cse660 = cse630*cse657;
  const amrex::Real cse661 = cse630*cse659 + cse644*cse658 + cse648*cse660;
  const amrex::Real cse662 = std::exp(-227.80000000000001*cse3);
  const amrex::Real cse663 = cse644*cse662;
  const amrex::Real cse664 = (3.0/5.0)*cse663;
  const amrex::Real cse665 = cse652*cse660 + cse652*cse664 + cse653*cse664;
  const amrex::Real cse666 = cse648*cse652 + cse648*cse653 + cse653*cse656 + cse661 + cse665;
  const amrex::Real cse667 = 1.0/cse666;
  const amrex::Real cse668 = 2.3872025442750958e-19*cse667;
  const amrex::Real cse669 = cse620*cse652;
  const amrex::Real cse670 = 2.1419999999999999e-10*cse633;
  const amrex::Real cse671 = cse653*cse662;
  const amrex::Real cse672 = cse652*cse662;
  const amrex::Real cse673 = 1.9139999999999998e-10*cse605*cse663 + 2.604e-10*cse609*cse663 + cse631*cse646 + cse650*cse669 + cse670*cse671 + cse670*cse672;
  const amrex::Real cse674 = 2.8023005140020001e-18*cse667;
  const amrex::Real cse675 = std::pow(cse666, -2);
  const amrex::Real cse676 = cse653*cse654;
  const amrex::Real cse677 = cse675*(-1.0633333333333333e-10*cse605*cse655 - 1.4466666666666666e-10*cse609*cse676 - cse610*cse648 - cse623*cse648 - cse638*cse652 - cse638*cse653 - cse651 - cse673);
  const amrex::Real cse678 = 2.3872025442750958e-19*cse661;
  const amrex::Real cse679 = 2.8023005140020001e-18*cse665;
  const amrex::Real cse680 = state.T < 100.0;
  const amrex::Real cse681 = state.T < 6000.0;
  const amrex::Real cse682 = state.T < 500;
  const amrex::Real cse683 = state.T < 10;
  const amrex::Real cse684 = 0.001*state.T;
  const amrex::Real cse685 = std::log(cse684);
  const amrex::Real cse686 = cse685*cse8;
  const amrex::Real cse687 = std::pow(cse7, -8);
  const amrex::Real cse688 = std::pow(cse685, 8)*cse687;
  const amrex::Real cse689 = std::pow(cse7, -7);
  const amrex::Real cse690 = std::pow(cse685, 7)*cse689;
  const amrex::Real cse691 = std::pow(cse7, -6);
  const amrex::Real cse692 = std::pow(cse685, 6)*cse691;
  const amrex::Real cse693 = std::pow(cse7, -5);
  const amrex::Real cse694 = std::pow(cse685, 5)*cse693;
  const amrex::Real cse695 = std::pow(cse7, -4);
  const amrex::Real cse696 = std::pow(cse685, 4)*cse695;
  const amrex::Real cse697 = std::pow(cse7, -3);
  const amrex::Real cse698 = std::pow(cse685, 3);
  const amrex::Real cse699 = cse697*cse698;
  const amrex::Real cse700 = cse13*std::pow(cse685, 2);
  const amrex::Real cse701 = std::pow(10.0, 16.815729999999999*cse686 + 70.609154000000004*cse688 + 364.14445999999998*cse690 + 801.81246999999996*cse692 + 983.67575999999997*cse694 + 734.71650999999997*cse696 + 343.1918*cse699 + 96.743155000000002*cse700 - 21.928795999999998)*state.xn[2];
  const amrex::Real cse702 = state.xn[5] + state.xn[6];
  const amrex::Real cse703 = std::pow(10.0, 1.5714710999999999*cse686 + 0.32168730000000001*cse694 - 0.51002221000000003*cse696 - 0.23619984999999999*cse699 + 0.015391166*cse700 - 22.089523)*state.xn[1] + std::pow(10.0, 2.1892372*cse686 + 0.19191374999999999*cse694 - 0.16596184*cse696 + 0.29036281000000003*cse699 - 0.81520437999999995*cse700 - 23.689236999999999)*cse702 + std::pow(10.0, 2.0943374000000001*cse685*cse8 - 0.033638326000000003*cse694 - 0.14913216000000001*cse696 + 0.43693353000000001*cse697*cse698 - 0.77151435999999995*cse700 - 23.962112000000001)*state.xn[3];
  const amrex::Real cse704 = std::pow(10, 37.383713*cse686 + 3.8479610000000002*cse694 + 20.159831000000001*cse696 + 48.656103000000002*cse699 + 58.145166000000003*cse700 - 16.818342000000001)*state.xn[0] + cse703;
  const amrex::Real cse705 = std::pow(10.0, 1.6802758*cse686 - 6.3701156000000001*cse688 + 6.4380698000000001*cse690 + 8.9167182999999994*cse692 - 8.8077017000000009*cse694 - 4.7274035999999997*cse696 + 4.0406627000000004*cse699 + 0.93310621999999999*cse700 - 22.921188999999998)*state.xn[2];
  const amrex::Real cse706 = state.T < 1000.0;
  const amrex::Real cse707 = std::pow(10.0, 3.5692468000000002*cse685*cse8 - 4.2519023000000002*cse694 - 21.328264000000001*cse696 - 27.850082*cse699 - 11.33286*cse700 - 24.311209000000002)*state.xn[0] + cse703;
  const amrex::Real cse708 = std::pow(10.0, 4.6450521*cse686 + 1.5538288*cse694 - 5.5108049000000001*cse696 + 5.9369081000000001*cse699 - 3.7209846*cse700 - 24.311209000000002)*state.xn[0] + cse703;
  const amrex::Real cse709 = std::pow(10.0, -11.945856385319289*cse13 + 28.69470131298657*cse693 - 56.79810800956944*cse695 + 34.150687525022562*cse697 + 8.3228160852326614*cse8 - 24.311209000000002)*state.xn[0] + std::pow(10.0, -2.6171337683481011*cse13 + 3.5440891133599637*cse693 - 1.7105157385968939*cse695 + 1.6702447513374001*cse697 + 3.9225864834863131*cse8 - 23.689236999999999)*cse702 + std::pow(10.0, -2.4768712409536775*cse13 - 0.62120210234156448*cse693 - 1.5370575959566981*cse695 + 2.5133588394664677*cse697 + 3.7525488682084647*cse8 - 23.962112000000001)*state.xn[3] + std::pow(10.0, 0.049411830040524525*cse13 + 5.9406293599919788*cse693 - 5.2566362076906961*cse695 - 1.3586848802337363*cse697 + 2.8156982240432273*cse8 - 22.089523)*state.xn[1];
  const amrex::Real cse710 = ((cse680 && cse681 && cse682) ? (
   cse683 || cse701 + cse704 < 1.0e-100
)
: (
   ((cse680 && cse681) ? (
      cse683 || cse704 + cse705 < 1.0e-100
   )
   : (
      ((cse681 && cse682 && cse706) ? (
         cse683 || cse701 + cse707 < 1.0e-100
      )
      : (
         ((cse681 && cse706) ? (
            cse683 || cse705 + cse707 < 1.0e-100
         )
         : (
            ((cse681 && cse682) ? (
               cse683 || cse701 + cse708 < 1.0e-100
            )
            : (
               ((cse681) ? (
                  cse683 || cse705 + cse708 < 1.0e-100
               )
               : (
                  ((cse682) ? (
                     cse683 || std::pow(10.0, 310.58441786958315*cse13 + 7500.646099484944*cse687 + 21588.957206746247*cse689 + 26530.843406310876*cse691 + 18165.631967965237*cse693 + 7572.4886742758581*cse695 + 1974.131269262874*cse697 + 30.129743459482278*cse8 - 21.928795999999998)*state.xn[2] + cse709 < 1.0e-100
                  )
                  : (
                     cse683 || std::pow(10.0, 2.9956460707652877*cse13 - 676.68255490510751*cse687 + 381.69251128040054*cse689 + 295.04162851880631*cse691 - 162.65264843551881*cse693 - 48.723840736518795*cse695 + 23.242975457496804*cse697 + 3.0106500755647456*cse8 - 22.921188999999998)*state.xn[2] + cse709 < 1.0e-100
                  ))
               ))
            ))
         ))
      ))
   ))
));
  const amrex::Real cse711 = std::log(0.001*((cse681) ? (
   state.T
)
: (
   6000.0
)));
  const amrex::Real cse712 = cse711*cse8;
  const amrex::Real cse713 = std::pow(cse711, 5);
  const amrex::Real cse714 = cse693*cse713;
  const amrex::Real cse715 = std::pow(cse711, 4);
  const amrex::Real cse716 = cse695*cse715;
  const amrex::Real cse717 = std::pow(cse711, 3);
  const amrex::Real cse718 = cse697*cse717;
  const amrex::Real cse719 = std::pow(cse711, 2);
  const amrex::Real cse720 = cse13*cse719;
  const amrex::Real cse721 = std::pow(10, 37.383713*cse712 + 3.8479610000000002*cse714 + 20.159831000000001*cse716 + 48.656103000000002*cse718 + 58.145166000000003*cse720 - 16.818342000000001);
  const amrex::Real cse722 = ((cse681) ? (
   cse680
)
: (
   false
));
  const amrex::Real cse723 = std::pow(10.0, 3.5692468000000002*cse711*cse8 - 4.2519023000000002*cse714 - 21.328264000000001*cse716 - 27.850082*cse718 - 11.33286*cse720 - 24.311209000000002);
  const amrex::Real cse724 = ((cse681) ? (
   cse706
)
: (
   false
));
  const amrex::Real cse725 = std::pow(10.0, 4.6450521*cse712 + 1.5538288*cse714 - 5.5108049000000001*cse716 + 5.9369081000000001*cse718 - 3.7209846*cse720 - 24.311209000000002);
  const amrex::Real cse726 = ((cse722) ? (
   cse721
)
: ((cse724) ? (
   cse723
)
: (
   cse725
)));
  const amrex::Real cse727 = std::pow(10.0, 0.43693353000000001*cse697*cse717 + 2.0943374000000001*cse711*cse8 - 0.033638326000000003*cse714 - 0.14913216000000001*cse716 - 0.77151435999999995*cse720 - 23.962112000000001);
  const amrex::Real cse728 = cse727*state.xn[3];
  const amrex::Real cse729 = std::pow(10.0, 1.5714710999999999*cse712 + 0.32168730000000001*cse714 - 0.51002221000000003*cse716 - 0.23619984999999999*cse718 + 0.015391166*cse720 - 22.089523);
  const amrex::Real cse730 = cse729*state.xn[1];
  const amrex::Real cse731 = std::pow(10.0, 2.1892372*cse712 + 0.19191374999999999*cse714 - 0.16596184*cse716 + 0.29036281000000003*cse718 - 0.81520437999999995*cse720 - 23.689236999999999);
  const amrex::Real cse732 = cse702*cse731;
  const amrex::Real cse733 = cse687*std::pow(cse711, 8);
  const amrex::Real cse734 = std::pow(cse711, 7);
  const amrex::Real cse735 = cse689*cse734;
  const amrex::Real cse736 = std::pow(cse711, 6);
  const amrex::Real cse737 = cse691*cse736;
  const amrex::Real cse738 = std::pow(10.0, 16.815729999999999*cse712 + 983.67575999999997*cse714 + 734.71650999999997*cse716 + 343.1918*cse718 + 96.743155000000002*cse720 + 70.609154000000004*cse733 + 364.14445999999998*cse735 + 801.81246999999996*cse737 - 21.928795999999998);
  const amrex::Real cse739 = std::pow(10.0, 1.6802758*cse712 - 8.8077017000000009*cse714 - 4.7274035999999997*cse716 + 4.0406627000000004*cse718 + 0.93310621999999999*cse720 - 6.3701156000000001*cse733 + 6.4380698000000001*cse735 + 8.9167182999999994*cse737 - 22.921188999999998);
  const amrex::Real cse740 = ((cse682) ? (
   cse738
)
: (
   cse739
));
  const amrex::Real cse741 = cse726*state.xn[0] + cse728 + cse730 + cse732 + cse740*state.xn[2];
  const amrex::Real cse742 = std::pow(cse741, -2);
  const amrex::Real cse743 = ((cse681) ? (
   cse3
)
: (
   0.00016666666666666666
));
  const amrex::Real cse744 = std::exp(-11700.0*cse743);
  const amrex::Real cse745 = std::exp(-5860.0*cse743);
  const amrex::Real cse746 = std::exp(-510.0*cse743);
  const amrex::Real cse747 = std::exp(-2197000.0*((cse681) ? (
   std::pow(state.T, -3)
)
: (
   4.6296296296296294e-12
)));
  const amrex::Real cse748 = cse747*((cse681) ? (
   std::pow(state.T, 3.7599999999999998)
)
: (
   160638152785302.53
));
  const amrex::Real cse749 = 6.0142468035272636e-8*((cse681) ? (
   std::pow(state.T, 2.1000000000000001)
)
: (
   85924621.118090734
)) + 1.0;
  const amrex::Real cse750 = 1.0/cse749;
  const amrex::Real cse751 = 1.6e-18*cse744 + 6.7e-19*cse745 + 3.0e-24*cse746 + 4.985670872372847e-33*cse748*cse750;
  const amrex::Real cse752 = 1.0/cse741;
  const amrex::Real cse753 = cse751*cse752 + 1;
  const amrex::Real cse754 = std::pow(cse753, -2);
  const amrex::Real cse755 = cse742*std::pow(cse751, 2)*cse754;
  const amrex::Real cse756 = 0.045999999999999999*cse603;
  const amrex::Real cse757 = std::pow(cse602, cse756 + 0.22800000000000001);
  const amrex::Real cse758 = std::pow(cse602, cse756 + 0.23100000000000001);
  const amrex::Real cse759 = 2.6400000000000002e-10*cse758;
  const amrex::Real cse760 = cse759*state.xn[0];
  const amrex::Real cse761 = 0.031*cse603;
  const amrex::Real cse762 = 1.11e-10*std::pow(cse602, cse761 + 0.26300000000000001);
  const amrex::Real cse763 = 0.064000000000000001*cse603;
  const amrex::Real cse764 = 4.3749999999999998e-11*std::pow(cse602, cse763 + 0.071999999999999995);
  const amrex::Real cse765 = cse762 + cse764;
  const amrex::Real cse766 = 0.00235272*cse71;
  const amrex::Real cse767 = cse766 - 0.041816600000000002;
  const amrex::Real cse768 = cse71*cse767 + 0.35826400000000003;
  const amrex::Real cse769 = cse71*cse768 - 0.57443;
  const amrex::Real cse770 = 0.097857299999999994*cse71;
  const amrex::Real cse771 = cse770 - 3.1926800000000002;
  const amrex::Real cse772 = cse71*cse771 + 38.504899999999999;
  const amrex::Real cse773 = cse71*cse772 - 202.19300000000001;
  const amrex::Real cse774 = std::exp(((cse706) ? (
   cse71*cse769 - 7.4386999999999999
)
: (
   cse71*cse773 + 386.18599999999998
)));
  const amrex::Real cse775 = cse264*state.xn[2];
  const amrex::Real cse776 = 1.7258000000000001e-6*cse775;
  const amrex::Real cse777 = cse774*cse776;
  const amrex::Real cse778 = cse760 + cse765*state.xn[3] + cse777;
  const amrex::Real cse779 = std::exp(-101.30000000000001*cse3);
  const amrex::Real cse780 = cse778*cse779;
  const amrex::Real cse781 = 8.9000000000000003e-11*cse757;
  const amrex::Real cse782 = cse781*state.xn[0];
  const amrex::Real cse783 = 0.037999999999999999*cse603;
  const amrex::Real cse784 = 5.1749999999999996e-11*std::pow(cse602, cse783 + 0.16900000000000001);
  const amrex::Real cse785 = 0.048000000000000001*cse603;
  const amrex::Real cse786 = 2.15e-11*std::pow(cse602, cse785 - 0.01);
  const amrex::Real cse787 = cse784 + cse786;
  const amrex::Real cse788 = 0.0070527699999999999*cse71;
  const amrex::Real cse789 = cse788 - 0.11133800000000001;
  const amrex::Real cse790 = cse71*cse789 + 0.69763799999999998;
  const amrex::Real cse791 = cse71*cse790 - 1.3074300000000001;
  const amrex::Real cse792 = 0.093813800000000003*cse71;
  const amrex::Real cse793 = cse792 - 3.0328300000000001;
  const amrex::Real cse794 = cse71*cse793 + 36.180300000000003;
  const amrex::Real cse795 = cse71*cse794 - 187.47399999999999;
  const amrex::Real cse796 = std::exp(((cse706) ? (
   cse71*cse791 - 7.6973500000000001
)
: (
   cse71*cse795 + 350.60899999999998
)));
  const amrex::Real cse797 = cse776*cse796;
  const amrex::Real cse798 = cse782 + cse787*state.xn[3] + cse797;
  const amrex::Real cse799 = cse779*cse798;
  const amrex::Real cse800 = 0.057000000000000002*cse603;
  const amrex::Real cse801 = std::pow(cse602, cse800 + 0.115);
  const amrex::Real cse802 = 1.26e-10*cse801;
  const amrex::Real cse803 = cse802*state.xn[0];
  const amrex::Real cse804 = 0.049000000000000002*cse603;
  const amrex::Real cse805 = 5.3249999999999997e-11*std::pow(cse602, cse804 - 0.0040000000000000001);
  const amrex::Real cse806 = 0.10199999999999999*cse603;
  const amrex::Real cse807 = 1.675e-11*std::pow(cse602, cse806 - 0.085000000000000006);
  const amrex::Real cse808 = cse805 + cse807;
  const amrex::Real cse809 = 0.00065632500000000003*cse71;
  const amrex::Real cse810 = -cse809 - 0.015089200000000001;
  const amrex::Real cse811 = cse71*cse810 + 0.36118400000000001;
  const amrex::Real cse812 = cse71*cse811 - 0.77378199999999997;
  const amrex::Real cse813 = 0.10508000000000001*cse71;
  const amrex::Real cse814 = cse813 - 3.4762;
  const amrex::Real cse815 = cse71*cse814 + 42.594999999999999;
  const amrex::Real cse816 = cse71*cse815 - 227.91300000000001;
  const amrex::Real cse817 = std::exp(((cse706) ? (
   cse71*cse812 - 9.2514099999999999
)
: (
   cse71*cse816 + 444.60000000000002
)));
  const amrex::Real cse818 = 2.8763333333333333e-6*cse817;
  const amrex::Real cse819 = cse775*cse818;
  const amrex::Real cse820 = cse803 + cse808*state.xn[3] + cse819;
  const amrex::Real cse821 = std::exp(-62.460000000000001*cse3);
  const amrex::Real cse822 = cse820*cse821;
  const amrex::Real cse823 = 1.32e-9*cse758;
  const amrex::Real cse824 = 6.3e-10*cse801;
  const amrex::Real cse825 = cse798*cse821;
  const amrex::Real cse826 = cse778*cse821;
  const amrex::Real cse827 = cse820 + 7.8800000000000004e-8;
  const amrex::Real cse828 = cse821*cse827;
  const amrex::Real cse829 = 4.4500000000000001e-10*cse757;
  const amrex::Real cse830 = 7.4166666666666679e-10*cse757*cse780 + 2.2000000000000003e-9*cse758*cse799 + cse822*cse823 + cse824*cse825 + cse824*cse826 + cse828*cse829;
  const amrex::Real cse831 = cse778 + 2.65e-7;
  const amrex::Real cse832 = cse798 + 1.81e-14;
  const amrex::Real cse833 = std::exp(-38.840000000000003*cse3);
  const amrex::Real cse834 = cse778*cse833;
  const amrex::Real cse835 = (5.0/3.0)*cse832;
  const amrex::Real cse836 = (25.0/3.0)*cse799;
  const amrex::Real cse837 = 5*cse821;
  const amrex::Real cse838 = cse778*cse837;
  const amrex::Real cse839 = cse798*cse837;
  const amrex::Real cse840 = cse778*cse836 + cse820*cse838 + cse827*cse839;
  const amrex::Real cse841 = std::exp(-23.619999999999997*cse3);
  const amrex::Real cse842 = cse820*cse841;
  const amrex::Real cse843 = 3*cse842;
  const amrex::Real cse844 = cse831*cse839 + cse831*cse843 + cse832*cse843;
  const amrex::Real cse845 = cse827*cse831 + cse827*cse832 + cse834*cse835 + cse840 + cse844;
  const amrex::Real cse846 = 1.0/cse845;
  const amrex::Real cse847 = 1.4210469458259591e-21*cse846;
  const amrex::Real cse848 = cse821*cse831;
  const amrex::Real cse849 = 3.7800000000000004e-10*cse801;
  const amrex::Real cse850 = cse832*cse841;
  const amrex::Real cse851 = cse831*cse841;
  const amrex::Real cse852 = 2.6700000000000001e-10*cse757*cse842 + 7.9200000000000006e-10*cse758*cse842 + cse823*cse825 + cse829*cse848 + cse849*cse850 + cse849*cse851;
  const amrex::Real cse853 = 2.5697412351439998e-22*cse846;
  const amrex::Real cse854 = std::pow(cse845, -2);
  const amrex::Real cse855 = cse832*cse833;
  const amrex::Real cse856 = cse854*(-1.4833333333333334e-10*cse757*cse834 - 4.4000000000000003e-10*cse758*cse855 - cse759*cse827 - cse781*cse827 - cse802*cse831 - cse802*cse832 - cse830 - cse852);
  const amrex::Real cse857 = 1.4210469458259591e-21*cse840;
  const amrex::Real cse858 = 2.5697412351439998e-22*cse844;
  const amrex::Real cse859 = state.T < 1000000.0;
  const amrex::Real cse860 = cse411 + 0.070000000000000007;
  const amrex::Real cse861 = 1.0/cse860;
  const amrex::Real cse862 = std::sqrt(cse411*cse861);
  const amrex::Real cse863 = 26.399999999999999*cse862;
  const amrex::Real cse864 = cse863 + 6.5;
  const amrex::Real cse865 = cse11*cse864;
  const amrex::Real cse866 = cse8*std::log(cse10);
  const amrex::Real cse867 = cse866 < 2;
  const amrex::Real cse868 = cse866 < 4;
  const amrex::Real cse869 = cse866 < 7;
  const amrex::Real cse870 = cse866 < 10;
  const amrex::Real cse871 = ((cse867) ? (
   10.0
)
: ((cse868) ? (
   (3.0/2.0)*cse866 + 7
)
: ((cse869) ? (
   (4.0/3.0)*cse866 + 23.0/3.0
)
: ((cse870) ? (
   (1.0/3.0)*cse866 + 44.0/3.0
)
: (
   18.0
)))));
  const amrex::Real cse872 = 2.0*cse11;
  const amrex::Real cse873 = cse871*cse872;
  const amrex::Real cse874 = cse865*state.xn[0] + cse873*state.xn[3];
  const amrex::Real cse875 = 1.0/state.xn[0];
  const amrex::Real cse876 = 3.0000000000000002e-15*d2g;
  const amrex::Real cse877 = cse10*cse876;
  const amrex::Real cse878 = cse875*cse877 + 1.9999999999999999e-7;
  const amrex::Real cse879 = cse684 + 1;
  const amrex::Real cse880 = std::pow(cse879, -2);
  const amrex::Real cse881 = 1.0/cse879;
  const amrex::Real cse882 = std::pow(10, -2.0230000000000001*cse880 - 3.6709999999999998*cse881 - 11.08);
  const amrex::Real cse883 = cse11*cse882;
  const amrex::Real cse884 = cse883*state.xn[3];
  const amrex::Real cse885 = std::pow(10, -2.3900000000000001*cse880 + 0.055500000000000001*cse881 - 11.06);
  const amrex::Real cse886 = cse11*cse885;
  const amrex::Real cse887 = cse886*state.xn[0];
  const amrex::Real cse888 = cse884 + cse887;
  const amrex::Real cse889 = 1.0/cse888;
  const amrex::Real cse890 = cse11*cse889;
  const amrex::Real cse891 = cse878*cse890 + 1;
  const amrex::Real cse892 = 3.2043540000000002e-13 + 6.7291434000000005e-12/cse891;
  const amrex::Real cse893 = cse38*cse864;
  const amrex::Real cse894 = cse35*state.xn[2];
  const amrex::Real cse895 = std::pow(cse860, -2);
  const amrex::Real cse896 = 0.5*cse895;
  const amrex::Real cse897 = cse121/std::pow(cse10, 3);
  const amrex::Real cse898 = -0.5*cse861*cse894 + cse896*cse897;
  const amrex::Real cse899 = cse65*cse860*cse863*state.xn[0];
  const amrex::Real cse900 = cse11*cse8;
  const amrex::Real cse901 = ((cse867) ? (
   0
)
: ((cse868) ? (
   (3.0/2.0)*cse900
)
: ((cse869) ? (
   (4.0/3.0)*cse900
)
: ((cse870) ? (
   (1.0/3.0)*cse900
)
: (
   0
)))));
  const amrex::Real cse902 = cse872*state.xn[3];
  const amrex::Real cse903 = cse36*cse871;
  const amrex::Real cse904 = 2.0*cse903;
  const amrex::Real cse905 = cse865 - cse893 + cse898*cse899 + cse901*cse902 - cse904;
  const amrex::Real cse906 = std::pow(state.T, 1.0/4.0);
  const amrex::Real cse907 = 1.75*cse906;
  const amrex::Real cse908 = std::exp(-state.T);
  const amrex::Real cse909 = 1 - cse908;
  const amrex::Real cse910 = std::pow(cse909, 1000);
  const amrex::Real cse911 = l0_co_interp1d(state.T);
  const amrex::Real cse912 = 1.0/cse911;
  const amrex::Real cse913 = 680.10000000000002/cse906;
  const amrex::Real cse914 = cse907*state.xn[0] + cse913*state.xn[2] + state.xn[3];
  const amrex::Real cse915 = M_SQRT2;
  const amrex::Real cse916 = std::max(gradv, 5.7445531974402559e-18*cse2*cse915);
  const amrex::Real cse917 = 100000.0/cse916;
  const amrex::Real cse918 = ncol_H <= 0;
  const amrex::Real cse919 = cse11*ncol_H;
  const amrex::Real cse920 = std::max(1.0000000000000001e-30, cse919);
  const amrex::Real cse921 = 1756701454.4310031*ncol_CO;
  const amrex::Real cse922 = ((cse918) ? (
   cse917*state.xn[9]
)
: (
   cse921/std::sqrt(cse920)
));
  const amrex::Real cse923 = llte_co_interp2d(state.T, cse922);
  const amrex::Real cse924 = 1.0/cse923;
  const amrex::Real cse925 = nhalf_co_interp2d(state.T, cse922);
  const amrex::Real cse926 = cse911*cse924;
  const amrex::Real cse927 = -cse925*cse926 + 1;
  const amrex::Real cse928 = 1.0/cse925;
  const amrex::Real cse929 = cse914*cse928;
  const amrex::Real cse930 = alpha_co_interp2d(state.T, cse922);
  const amrex::Real cse931 = std::pow(cse929, cse930);
  const amrex::Real cse932 = cse912*cse931;
  const amrex::Real cse933 = cse927*cse932;
  const amrex::Real cse934 = cse912 + cse914*cse924 + cse933;
  const amrex::Real cse935 = 1.0/cse934;
  const amrex::Real cse936 = cse910*cse935;
  const amrex::Real cse937 = cse936*state.xn[9];
  const amrex::Real cse938 = cse535 + 1.9999999999999999e-7;
  const amrex::Real cse939 = cse890*cse938 + 1;
  const amrex::Real cse940 = std::pow(cse939, -2);
  const amrex::Real cse941 = cse35*cse889*cse938;
  const amrex::Real cse942 = cse36*cse882;
  const amrex::Real cse943 = cse38*cse885;
  const amrex::Real cse944 = cse942 + cse943;
  const amrex::Real cse945 = std::pow(cse888, -2);
  const amrex::Real cse946 = cse11*cse945;
  const amrex::Real cse947 = cse946*(-cse886 + cse944);
  const amrex::Real cse948 = cse35*ncol_H*(((cse919 - 1.0000000000000001e-30 < 0) ? (
   0
)
: ((cse919 - 1.0000000000000001e-30 == 0) ? (
   1.0/2.0
)
: (
   1
))))/std::pow(cse920, 3.0/2.0);
  const amrex::Real cse949 = cse948*ncol_CO;
  const amrex::Real cse950 = ((cse918) ? (
   0
)
: (
   878350727.21550155*cse949
));
  const amrex::Real cse951 = alpha_co_interp2d_partial_1(state.T, cse922);
  const amrex::Real cse952 = std::log(cse929);
  const amrex::Real cse953 = cse951*cse952;
  const amrex::Real cse954 = cse914/std::pow(cse925, 2);
  const amrex::Real cse955 = nhalf_co_interp2d_partial_1(state.T, cse922);
  const amrex::Real cse956 = cse950*cse955;
  const amrex::Real cse957 = cse930/cse914;
  const amrex::Real cse958 = cse925*cse957;
  const amrex::Real cse959 = std::pow(cse923, -2);
  const amrex::Real cse960 = llte_co_interp2d_partial_1(state.T, cse922);
  const amrex::Real cse961 = -cse914*cse950*cse959*cse960 + cse932*(cse911*cse925*cse950*cse959*cse960 - cse926*cse956);
  const amrex::Real cse962 = cse914*state.xn[9];
  const amrex::Real cse963 = cse910*cse962/std::pow(cse934, 2);
  const amrex::Real cse964 = 0.0086999999999999994*cse603;
  const amrex::Real cse965 = std::pow(cse602, cse964 + 0.12809999999999999);
  const amrex::Real cse966 = 7.5799999999999997e-10*cse965*state.xn[0];
  const amrex::Real cse967 = std::pow(state.T, 0.070000000000000007);
  const amrex::Real cse968 = 0.75*((cse682) ? (
   1.1000000000000001e-13*state.T + 5.3300000000000002e-10
)
: (
   3.74757785025e-10*cse967
)) + 0.25*((cse682) ? (
   3.3000000000000001e-13*state.T + 4.4299999999999997e-10
)
: (
   3.8899728635600003e-10*cse967
));
  const amrex::Real cse969 = 4.5299999999999998e-6*cse775 + cse966 + cse968*state.xn[3];
  const amrex::Real cse970 = cse969 + 2.3e-6;
  const amrex::Real cse971 = std::exp(-91.209999999999994*cse3);
  const amrex::Real cse972 = cse971/cse970;
  const amrex::Real cse973 = 2*cse972;
  const amrex::Real cse974 = cse969*cse973 + 1;
  const amrex::Real cse975 = 1.0/cse974;
  const amrex::Real cse976 = cse972*state.xn[8];
  const amrex::Real cse977 = cse975*cse976;
  const amrex::Real cse978 = std::pow(cse891, -2);
  const amrex::Real cse979 = -cse35*cse878*cse889;
  const amrex::Real cse980 = cse875*cse876;
  const amrex::Real cse981 = std::pow(state.xn[0], 2);
  const amrex::Real cse982 = 1.5159999999999999e-9*cse965;
  const amrex::Real cse983 = std::pow(cse970, -2);
  const amrex::Real cse984 = cse969*cse971*cse983;
  const amrex::Real cse985 = cse969*cse972;
  const amrex::Real cse986 = 5.7927337833400002e-20*cse985*state.xn[8]/std::pow(cse974, 2);
  const amrex::Real cse987 = std::pow(state.T, 0.14999999999999999);
  const amrex::Real cse988 = 0.00088240812266906664*std::pow(state.T, 0.65000000000000002) + 1;
  const amrex::Real cse989 = 1.0/cse988;
  const amrex::Real cse990 = cse989*state.xn[2];
  const amrex::Real cse991 = cse987*cse990;
  const amrex::Real cse992 = 4.4460389837719569e-9*cse991 + 626500000.0;
  const amrex::Real cse993 = 1.0/cse992;
  const amrex::Real cse994 = std::exp(-118400.0*cse3);
  const amrex::Real cse995 = cse993*cse994;
  const amrex::Real cse996 = 1.3338116951315871e-8*cse995;
  const amrex::Real cse997 = cse991*cse996 + 1;
  const amrex::Real cse998 = 1.0/cse997;
  const amrex::Real cse999 = cse995*cse998;
  const amrex::Real cse1000 = 1.3659996292638161e-10*cse999;
  const amrex::Real cse1001 = 100.0*d2g;
  const amrex::Real cse1002 = psigd_coll_interp2d(cse10, state.T);
  const amrex::Real cse1003 = 2.8677558904086741e-6*cse68 + 4.1397668473788896;
  const amrex::Real cse1004 = -2.1544346900318838*std::pow(cse1003, 1.0/6.0) + state.T;
  const amrex::Real cse1005 = 3.2000000000000002e-32*cse2;
  const amrex::Real cse1006 = cse1005*cse379;
  const amrex::Real cse1007 = -cse1001*cse1002 + cse1004*cse1006;
  const amrex::Real cse1008 = (((cse1007 < 0) ? (
   0
)
: ((cse1007 == 0) ? (
   1.0/2.0
)
: (
   1
))));
  const amrex::Real cse1009 = cse1008*psigd_coll_interp2d_partial_0(cse10, state.T);
  const amrex::Real cse1010 = cse1001*cse1009;
  const amrex::Real cse1011 = (((cse1007 > 0) ? (
   0
)
: ((cse1007 == 0) ? (
   1.0/2.0
)
: (
   1
))));
  const amrex::Real cse1012 = cse1004*cse1011*d2g;
  const amrex::Real cse1013 = cse1005*cse1012;
  const amrex::Real cse1014 = state.xn[2] + 1.0e-50;
  const amrex::Real cse1015 = 1.0/cse1014;
  const amrex::Real cse1016 = cse1015*cse2;
  const amrex::Real cse1017 = 5.670589618368584e-11*cse68;
  const amrex::Real cse1018 = cse1016*cse1017 + 50;
  const amrex::Real cse1019 = std::log(cse1018);
  const amrex::Real cse1020 = 0.45350000000000001 + 2.234/cse1019;
  const amrex::Real cse1021 = std::pow(state.T, cse1020);
  const amrex::Real cse1022 = 1.4419999999999999 - 0.050889999999999998*cse1019;
  const amrex::Real cse1023 = std::exp(cse1019*cse1022);
  const amrex::Real cse1024 = 1.8998126555754159e-29*cse1021*cse1023;
  const amrex::Real cse1025 = d2g*state.xn[2];
  const amrex::Real cse1026 = cse1024*cse1025;
  const amrex::Real cse1027 = 2.25*std::pow(state.T, 0.14699999999999999) + 5.2199999999999998;
  const amrex::Real cse1028 = cse70 + 50;
  const amrex::Real cse1029 = 0.0043*std::pow(cse1028, 0.69199999999999995) + 1;
  const amrex::Real cse1030 = 0.049959999999999997*std::pow(cse1028, 0.43099999999999999)*cse1029 + 1;
  const amrex::Real cse1031 = 1.0/cse1030;
  const amrex::Real cse1032 = -5.6705896183685846e-35*state.c_hat*cse1027*cse1031*cse67*d2g + cse1010 + cse1013 + cse1026 - 3.0000000000000002e-15*cse892*d2g*state.xn[0];
  const amrex::Real cse1033 = 7.1777529600000012e-12*cse116;
  const amrex::Real cse1034 = 7.1777529600000012e-12*cse528;
  const amrex::Real cse1035 = 2.0*cse11*cse901*state.xn[3] + 26.399999999999999*cse65*cse860*cse862*cse898*state.xn[0] - cse893 - cse904;
  const amrex::Real cse1036 = cse944*cse946;
  const amrex::Real cse1037 = -cse1036*cse938 + cse35*cse889*cse938;
  const amrex::Real cse1038 = cse928*cse930;
  const amrex::Real cse1039 = cse963*(-cse933*(-cse1038*cse956 + cse950*cse951*cse952) - cse961);
  const amrex::Real cse1040 = -cse1036*cse878 - cse890*cse980 - cse979;
  const amrex::Real cse1041 = cse987*cse989;
  const amrex::Real cse1042 = std::pow(state.T, 0.29999999999999999);
  const amrex::Real cse1043 = std::pow(cse988, -2);
  const amrex::Real cse1044 = std::pow(cse992, -2);
  const amrex::Real cse1045 = cse1024*cse379;
  const amrex::Real cse1046 = 1.3659996292638161e-10*cse1041*cse296;
  const amrex::Real cse1047 = cse1046*cse995/std::pow(cse997, 2);
  const amrex::Real cse1048 = std::pow(cse1014, -2);
  const amrex::Real cse1049 = cse1048*cse2;
  const amrex::Real cse1050 = cse1021*cse1023;
  const amrex::Real cse1051 = 1.0/cse1018;
  const amrex::Real cse1052 = cse1051*cse71/std::pow(cse1019, 2);
  const amrex::Real cse1053 = cse1050*cse1052*state.xn[2];
  const amrex::Real cse1054 = 0.5*cse11*cse861 - cse894*cse896;
  const amrex::Real cse1055 = std::pow(cse1030, -2);
  const amrex::Real cse1056 = std::pow(cse1028, 0.12299999999999994);
  const amrex::Real cse1057 = cse1056*cse68;
  const amrex::Real cse1058 = std::pow(cse1028, -0.56899999999999995)*cse1029;
  const amrex::Real cse1059 = cse1058*cse68;
  const amrex::Real cse1060 = cse1022*cse1051;
  const amrex::Real cse1061 = cse1045*state.xn[2];
  const amrex::Real cse1062 = 9.0599999999999997e-6*cse264;
  const amrex::Real cse1063 = 1.024e-10*cse636;
  const amrex::Real cse1064 = 2.1600000000000004e-15*cse606;
  const amrex::Real cse1065 = 9.7199999999999998e-11*cse621;
  const amrex::Real cse1066 = cse1063*cse631 + cse1063*cse635 + cse1064*cse645 + cse1065*cse649 + 7.1999999999999997e-16*cse606*cse647 + 3.2399999999999999e-11*cse619*cse621;
  const amrex::Real cse1067 = 3.0719999999999998e-10*cse636;
  const amrex::Real cse1068 = cse1064*cse631 + cse1065*cse669 + cse1067*cse671 + cse1067*cse672 + 6.4799999999999996e-15*cse606*cse663 + 2.9159999999999995e-10*cse621*cse663;
  const amrex::Real cse1069 = cse675*(-cse1066 - cse1068 - 3.6000000000000001e-15*cse606*cse676 - cse607*cse648 - 1.6199999999999998e-10*cse621*cse655 - cse622*cse648 - cse637*cse652 - cse637*cse653);
  const amrex::Real cse1070 = cse264*cse774;
  const amrex::Real cse1071 = cse264*cse817;
  const amrex::Real cse1072 = 1.4381666666666666e-5*cse1071;
  const amrex::Real cse1073 = cse264*cse796;
  const amrex::Real cse1074 = 8.6289999999999999e-6*cse1070;
  const amrex::Real cse1075 = 8.6289999999999999e-6*cse1073;
  const amrex::Real cse1076 = 1.4381666666666668e-5*cse1070*cse799 + cse1072*cse825 + cse1072*cse826 + 1.4381666666666668e-5*cse1073*cse780 + cse1074*cse822 + cse1075*cse828;
  const amrex::Real cse1077 = 5.1774000000000006e-6*cse842;
  const amrex::Real cse1078 = 8.6289999999999999e-6*cse1071;
  const amrex::Real cse1079 = cse1070*cse1077 + cse1073*cse1077 + cse1074*cse825 + cse1075*cse848 + cse1078*cse850 + cse1078*cse851;
  const amrex::Real cse1080 = cse264*cse818;
  const amrex::Real cse1081 = 1.7258000000000001e-6*cse827;
  const amrex::Real cse1082 = cse854*(-cse1070*cse1081 - 2.8763333333333337e-6*cse1070*cse855 - cse1073*cse1081 - 2.8763333333333337e-6*cse1073*cse834 - cse1076 - cse1079 - cse1080*cse831 - cse1080*cse832);
  const amrex::Real cse1083 = cse1025*cse1050;
  const amrex::Real cse1084 = 3.7996253111508318e-29*cse1083;
  const amrex::Real cse1085 = cse1012*cse2;
  const amrex::Real cse1086 = 6.4000000000000004e-32*cse1085;
  const amrex::Real cse1087 = cse1009*d2g;
  const amrex::Real cse1088 = 200.0*cse1087;
  const amrex::Real cse1089 = 5.7927337833400002e-20*cse975;
  const amrex::Real cse1090 = cse1089*cse976;
  const amrex::Real cse1091 = 6.4087080000000003e-13 + 2.5634832e-11/cse939;
  const amrex::Real cse1092 = -cse968;
  const amrex::Real cse1093 = cse1089*cse984*state.xn[8];
  const amrex::Real cse1094 = 2*cse984;
  const amrex::Real cse1095 = cse169*cse882 + 2*cse943;
  const amrex::Real cse1096 = cse946*(cse1095 - cse883);
  const amrex::Real cse1097 = cse875*cse890;
  const amrex::Real cse1098 = cse1097*cse56 - 2*cse35*cse878*cse889;
  const amrex::Real cse1099 = 2*cse893;
  const amrex::Real cse1100 = cse861*cse894;
  const amrex::Real cse1101 = cse895*cse897;
  const amrex::Real cse1102 = -1.0*cse1100 + 1.0*cse1101;
  const amrex::Real cse1103 = ((cse867) ? (
   0
)
: ((cse868) ? (
   3*cse900
)
: ((cse869) ? (
   (8.0/3.0)*cse900
)
: ((cse870) ? (
   (2.0/3.0)*cse900
)
: (
   0
)))));
  const amrex::Real cse1104 = 4.0*cse903;
  const amrex::Real cse1105 = -cse1099 + cse1102*cse899 + cse1103*cse902 - cse1104 + cse873;
  const amrex::Real cse1106 = ((cse918) ? (
   0
)
: (
   cse921*cse948
));
  const amrex::Real cse1107 = cse1106*cse955;
  const amrex::Real cse1108 = -cse1106*cse914*cse959*cse960 + cse932*(cse1106*cse911*cse925*cse959*cse960 - cse1107*cse926);
  const amrex::Real cse1109 = 1.0/cse753;
  const amrex::Real cse1110 = cse644*cse657;
  const amrex::Real cse1111 = (1.0/15.0)*cse647;
  const amrex::Real cse1112 = cse629*cse648;
  const amrex::Real cse1113 = cse1110*cse616 + cse1111*cse616 + cse1112*cse657 + cse629*cse659 + cse643*cse658 + cse643*cse660;
  const amrex::Real cse1114 = cse643*cse653;
  const amrex::Real cse1115 = (3.0/5.0)*cse662;
  const amrex::Real cse1116 = cse643*cse652;
  const amrex::Real cse1117 = cse652*cse657;
  const amrex::Real cse1118 = cse1114*cse1115 + cse1115*cse1116 + cse1117*cse629 + cse616*cse660 + cse616*cse664 + cse629*cse664;
  const amrex::Real cse1119 = (1.0/3.0)*cse676;
  const amrex::Real cse1120 = cse675*(-cse1112 - cse1113 - cse1114 - cse1116 - cse1118 - cse1119*cse616 - cse616*cse648 - cse629*cse656);
  const amrex::Real cse1121 = cse820*cse837;
  const amrex::Real cse1122 = (25.0/3.0)*cse780;
  const amrex::Real cse1123 = cse787*cse827;
  const amrex::Real cse1124 = cse1121*cse765 + cse1122*cse787 + cse1123*cse837 + cse765*cse836 + cse808*cse838 + cse808*cse839;
  const amrex::Real cse1125 = cse831*cse837;
  const amrex::Real cse1126 = cse808*cse832;
  const amrex::Real cse1127 = 3*cse841;
  const amrex::Real cse1128 = cse808*cse831;
  const amrex::Real cse1129 = cse1125*cse787 + cse1126*cse1127 + cse1127*cse1128 + cse765*cse839 + cse765*cse843 + cse787*cse843;
  const amrex::Real cse1130 = cse833*cse835;
  const amrex::Real cse1131 = (5.0/3.0)*cse834;
  const amrex::Real cse1132 = cse854*(-cse1123 - cse1124 - cse1126 - cse1128 - cse1129 - cse1130*cse765 - cse1131*cse787 - cse765*cse827);
  const amrex::Real cse1133 = cse1027*cse1031;
  const amrex::Real cse1134 = cse1133*d2g;
  const amrex::Real cse1135 = cse1134*cse68;
  const amrex::Real cse1136 = cse1095*cse946;
  const amrex::Real cse1137 = 2.0187430200000004e-26*cse163*cse978;
  const amrex::Real cse1138 = cse10*cse1137;
  const amrex::Real cse1139 = -cse1099 + 2.0*cse11*cse1103*state.xn[3] + 26.399999999999999*cse1102*cse65*cse860*cse862*state.xn[0] - cse1104;
  const amrex::Real cse1140 = 1.602177e-12*cse86;
  const amrex::Real cse1141 = 3.2043540000000001e-12*crate*state.xn[3];
  const amrex::Real cse1142 = cse535*cse940;
  const amrex::Real cse1143 = 2.5634832e-11*cse1142*state.xn[3];
  const amrex::Real cse1144 = -state.xn[3]*((cse710) ? (
   0
)
: (
   cse731*cse755
));
  const amrex::Real cse1145 = ((cse918) ? (
   cse917
)
: (
   0
));
  const amrex::Real cse1146 = cse1145*cse955;
  const amrex::Real cse1147 = 5.6705896183685846e-35*cse68;
  const amrex::Real cse1148 = -cse1010 - cse1013 - cse1026 - cse1033 - cse1034 + cse1035*cse1140 + cse1035*cse1141 + cse1037*cse1143 - cse1039 + cse1040*cse1138 + cse1134*cse1147 + 3.0000000000000002e-15*cse163*cse892;
  const amrex::Real cse1149 = cse946*(3*cse942 + 3*cse943);
  const amrex::Real cse1150 = 2.0*cse11*state.xn[3]*((cse867) ? (
   0
)
: ((cse868) ? (
   (9.0/2.0)*cse900
)
: ((cse869) ? (
   4*cse900
)
: ((cse870) ? (
   cse900
)
: (
   0
))))) + 26.399999999999999*cse65*cse860*cse862*state.xn[0]*(-1.5*cse1100 + 1.5*cse1101) - 3*cse893 - 6.0*cse903;
  const amrex::Real cse1151 = ((cse918) ? (
   0
)
: (
   2635052181.6465044*cse949
));
  const amrex::Real cse1152 = cse1151*cse955;
  const amrex::Real cse1153 = cse995*std::pow(state.T, -1.8500000000000001);
  const amrex::Real cse1154 = cse296*cse989;
  const amrex::Real cse1155 = std::pow(state.T, -0.84999999999999998);
  const amrex::Real cse1156 = std::pow(state.T, -0.19999999999999998);
  const amrex::Real cse1157 = cse1155*cse990;
  const amrex::Real cse1158 = cse1044*cse994*(2.5500935934394033e-12*cse1043*cse1156*state.xn[2] - 6.6690584756579352e-10*cse1157);
  const amrex::Real cse1159 = cse1015*cse277;
  const amrex::Real cse1160 = cse1159*cse68;
  const amrex::Real cse1161 = cse1019*cse1051;
  const amrex::Real cse1162 = cse260*cse985;
  const amrex::Real cse1163 = cse3*cse775;
  const amrex::Real cse1164 = std::pow(state.T, -0.92999999999999994);
  const amrex::Real cse1165 = 100*cse3;
  const amrex::Real cse1166 = -2.2649999999999999e-6*cse1163 + cse966*(cse1165*(8.7000000000000001e-5*cse603 + 0.001281) + cse3*cse964) + state.xn[3]*(0.75*((cse682) ? (
   1.1000000000000001e-13
)
: (
   2.6233044951750002e-11*cse1164
)) + 0.25*((cse682) ? (
   3.3000000000000001e-13
)
: (
   2.7229810044920004e-11*cse1164
)));
  const amrex::Real cse1167 = 0.4375*state.xn[0]/std::pow(state.T, 3.0/4.0) - 170.02500000000001*state.xn[2]/std::pow(state.T, 5.0/4.0);
  const amrex::Real cse1168 = -cse1166;
  const amrex::Real cse1169 = std::pow(cse879, -3);
  const amrex::Real cse1170 = -cse7*cse884*(0.0040460000000000001*cse1169 + 0.0036709999999999998*cse880) - cse7*cse887*(0.0047800000000000004*cse1169 - 5.5500000000000001e-5*cse880);
  const amrex::Real cse1171 = l0_co_interp1d_partial_0(state.T);
  const amrex::Real cse1172 = std::pow(cse911, -2);
  const amrex::Real cse1173 = ((cse918) ? (
   -2.872276598720128e-13*cse277*cse915*state.xn[9]*(((5.7445531974402559e-18*cse2*cse915 - gradv < 0) ? (
      0
   )
   : ((5.7445531974402559e-18*cse2*cse915 - gradv == 0) ? (
      1.0/2.0
   )
   : (
      1
   ))))/std::pow(cse916, 2)
)
: (
   0
));
  const amrex::Real cse1174 = -cse1173*cse960 - llte_co_interp2d_partial_0(state.T, cse922);
  const amrex::Real cse1175 = cse1173*cse955 + nhalf_co_interp2d_partial_0(state.T, cse922);
  const amrex::Real cse1176 = ((cse681) ? (
   -cse260
)
: (
   0
));
  const amrex::Real cse1177 = -1.8719999999999998e-14*cse1176*cse744 - 3.9261999999999998e-15*cse1176*cse745 - 1.53e-21*cse1176*cse746 + 4.985670872372847e-33*cse747*cse750*((cse681) ? (
   3.7599999999999998*std::pow(state.T, 2.7599999999999998)
)
: (
   0
)) - 1.0953518906603145e-26*cse748*cse750*((cse681) ? (
   -3/std::pow(state.T, 4)
)
: (
   0
)) - 2.9985055107607381e-40*cse748*((cse681) ? (
   2.1000000000000001*std::pow(state.T, 1.1000000000000001)
)
: (
   0
))/std::pow(cse749, 2);
  const amrex::Real cse1178 = ((cse681) ? (
   1
)
: (
   0
));
  const amrex::Real cse1179 = cse1178*cse743;
  const amrex::Real cse1180 = cse1179*cse13*cse711;
  const amrex::Real cse1181 = cse1179*cse693*cse715;
  const amrex::Real cse1182 = cse1179*cse695*cse717;
  const amrex::Real cse1183 = cse1179*cse8;
  const amrex::Real cse1184 = cse1179*cse697*cse719;
  const amrex::Real cse1185 = cse1179*cse687*cse734;
  const amrex::Real cse1186 = cse1179*cse689*cse736;
  const amrex::Real cse1187 = cse1179*cse691*cse713;
  const amrex::Real cse1188 = 65.320000000000007*cse260;
  const amrex::Real cse1189 = cse3*cse612;
  const amrex::Real cse1190 = 0.00042999999999999999*cse603;
  const amrex::Real cse1191 = cse639*(cse1165*(0.0041900000000000001 - 3.0000000000000001e-5*cse603) - cse3*cse632) - 3.8399999999999998e-11*std::pow(state.T, -1.075)*state.xn[2] + state.xn[3]*(cse641*(cse1165*(0.00025000000000000001*cse603 + 0.00264) + cse3*cse640) + cse642*(cse1165*(cse1190 + 0.00296) + cse1189));
  const amrex::Real cse1192 = cse624*(cse1165*(0.0036900000000000001 - 6.0000000000000002e-5*cse603) - cse3*cse604) - 1.2635999999999999e-11*std::pow(state.T, -1.026)*state.xn[2] + state.xn[3]*(cse626*(cse1165*(0.00041000000000000005*cse603 + 0.0020300000000000001) + cse3*cse625) + cse628*(cse1165*(0.00058*cse603 + 0.0023700000000000001) + cse3*cse627));
  const amrex::Real cse1193 = cse611*(cse1165*(0.0075500000000000003 - 0.0016000000000000001*cse603) - cse3*cse608) + 1.0000800000000001e-14*std::pow(state.T, -0.073999999999999955)*state.xn[2] + state.xn[3]*(cse613*(cse1165*(cse1190 + 0.0088900000000000003) + cse1189) + cse615*(cse1165*(0.0052500000000000003*cse603 + 0.011979999999999999) + cse3*cse614));
  const amrex::Real cse1194 = cse1192*cse648;
  const amrex::Real cse1195 = cse1110*cse1193 + cse1111*cse1193 + cse1188*cse617*cse645 + cse1188*cse630*cse649 + cse1191*cse658 + cse1191*cse660 + cse1192*cse659 + cse1194*cse657 + 28.360000000000003*cse260*cse619*cse630;
  const amrex::Real cse1196 = 136.68000000000001*cse260*cse663;
  const amrex::Real cse1197 = cse1191*cse653;
  const amrex::Real cse1198 = cse1191*cse652;
  const amrex::Real cse1199 = cse1115*cse1197 + cse1115*cse1198 + cse1117*cse1192 + cse1188*cse631*cse652 + cse1192*cse664 + cse1193*cse660 + cse1193*cse664 + cse1196*cse652 + cse1196*cse653;
  const amrex::Real cse1200 = cse675*(-cse1119*cse1193 - cse1192*cse656 - cse1193*cse648 - cse1194 - cse1195 - cse1197 - cse1198 - cse1199 - 32.933333333333337*cse260*cse653*cse655);
  const amrex::Real cse1201 = 312.30000000000001*cse260;
  const amrex::Real cse1202 = cse3*cse756;
  const amrex::Real cse1203 = 0.00046000000000000001*cse603;
  const amrex::Real cse1204 = 8.6290000000000003e-7*cse1163;
  const amrex::Real cse1205 = -cse1204*cse774 + cse760*(cse1165*(cse1203 + 0.00231) + cse1202) + cse777*((cse706) ? (
   cse3*cse769 + cse71*(cse3*cse768 + cse71*(cse3*cse766 + cse3*cse767))
)
: (
   cse3*cse773 + cse71*(cse3*cse772 + cse71*(cse3*cse770 + cse3*cse771))
)) + state.xn[3]*(cse762*(cse1165*(0.00031*cse603 + 0.00263) + cse3*cse761) + cse764*(cse1165*(0.00064000000000000005*cse603 + 0.00071999999999999994) + cse3*cse763));
  const amrex::Real cse1206 = -1.4381666666666666e-6*cse1163*cse817 + cse803*(cse1165*(0.00057000000000000009*cse603 + 0.00115) + cse3*cse800) + cse819*((cse706) ? (
   cse3*cse812 + cse71*(cse3*cse811 + cse71*(-cse3*cse809 + cse3*cse810))
)
: (
   cse3*cse816 + cse71*(cse3*cse815 + cse71*(cse3*cse813 + cse3*cse814))
)) + state.xn[3]*(cse805*(cse1165*(0.00048999999999999998*cse603 - 4.0000000000000003e-5) + cse3*cse804) + cse807*(cse1165*(0.0010199999999999999*cse603 - 0.00085000000000000006) + cse3*cse806));
  const amrex::Real cse1207 = -cse1204*cse796 + cse782*(cse1165*(cse1203 + 0.0022800000000000003) + cse1202) + cse797*((cse706) ? (
   cse3*cse791 + cse71*(cse3*cse790 + cse71*(cse3*cse788 + cse3*cse789))
)
: (
   cse3*cse795 + cse71*(cse3*cse794 + cse71*(cse3*cse792 + cse3*cse793))
)) + state.xn[3]*(cse784*(cse1165*(0.00038000000000000002*cse603 + 0.0016900000000000001) + cse3*cse783) + cse786*(cse1165*(0.00048000000000000001*cse603 - 0.0001) + cse3*cse785));
  const amrex::Real cse1208 = cse1207*cse827;
  const amrex::Real cse1209 = cse1121*cse1205 + cse1122*cse1207 + cse1201*cse778*cse822 + cse1201*cse798*cse828 + cse1205*cse836 + cse1206*cse838 + cse1206*cse839 + cse1208*cse837 + 844.16666666666686*cse260*cse778*cse799;
  const amrex::Real cse1210 = 70.859999999999985*cse260*cse842;
  const amrex::Real cse1211 = cse1206*cse832;
  const amrex::Real cse1212 = cse1206*cse831;
  const amrex::Real cse1213 = cse1125*cse1207 + cse1127*cse1211 + cse1127*cse1212 + cse1201*cse825*cse831 + cse1205*cse839 + cse1205*cse843 + cse1207*cse843 + cse1210*cse831 + cse1210*cse832;
  const amrex::Real cse1214 = cse854*(-cse1130*cse1205 - cse1131*cse1207 - cse1205*cse827 - cse1208 - cse1209 - cse1211 - cse1212 - cse1213 - 64.733333333333348*cse260*cse832*cse834);
  const amrex::Real cse1215 = state.c_hat*cse379;
  const amrex::Real cse1216 = cse1133*cse1215;
  const amrex::Real cse1217 = cse1011*cse1215*cse2/std::pow(cse1003, 5.0/6.0);
  const amrex::Real cse1218 = cse1016*cse1053*cse1215;
  const amrex::Real cse1219 = state.c_hat*cse66;
  const amrex::Real cse1220 = cse1056*cse1219;
  const amrex::Real cse1221 = cse1058*cse1219;
  const amrex::Real cse1222 = cse1027*cse1055*cse1147*cse379;
  const amrex::Real cse1223 = state.c_hat*cse1016;
  const amrex::Real cse1224 = cse1161*cse1223;
  const amrex::Real cse1225 = cse1060*cse1223;
  const amrex::Real cse1226 = cse1142*cse183*cse313*cse890;
  const amrex::Real cse1227 = Zd*cse185;
  const amrex::Real cse1228 = 1.252779946046115e-21*cse1227;
  const amrex::Real cse1229 = -cse1228;
  const amrex::Real cse1230 = 2.50555989209223e-21*cse1227;
  const amrex::Real cse1231 = 1.8689621674132958e-20*cse183;
  const amrex::Real cse1232 = 1.252779946046115e-21*Zd;
  const amrex::Real cse1233 = state.c_hat*cse10;
  const amrex::Real cse1234 = -cse1231*cse313 - cse1232*cse1233 - cse322 - cse505;
  const amrex::Real cse1235 = state.c_hat*state.rn[2*0+1];
  const amrex::Real cse1236 = cse1232*cse1235;
  const amrex::Real cse1237 = -cse1236;
  const amrex::Real cse1238 = Zd*cse1235;
  const amrex::Real cse1239 = 2.50555989209223e-21*cse1238;
  const amrex::Real cse1240 = Zd*cse181;
  const amrex::Real cse1241 = 2.0584539190345922e-21*cse1240;
  const amrex::Real cse1242 = -cse1241;
  const amrex::Real cse1243 = 4.1169078380691844e-21*cse1240;
  const amrex::Real cse1244 = 2.6551134042892157e-17*cse183;
  const amrex::Real cse1245 = 2.0584539190345922e-21*Zd;
  const amrex::Real cse1246 = -cse1233*cse1245 - cse1244*cse313 - cse323 - cse515 - cse518 - cse596;
  const amrex::Real cse1247 = state.c_hat*state.rn[2*1+1];
  const amrex::Real cse1248 = cse1245*cse1247;
  const amrex::Real cse1249 = -cse1248;
  const amrex::Real cse1250 = Zd*cse1247;
  const amrex::Real cse1251 = 4.1169078380691844e-21*cse1250;

  jac(1, 1) = -cse1 - cse10*cse56 + 2*cse29*state.xn[3] + 2*cse30*cse55 - cse57 + 2*cse60*state.xn[0]*state.xn[3] - cse99;
  jac(1, 2) = cse111 + cse120;
  jac(1, 3) = cse130 + cse132*state.xn[16] + cse136*cse137 + cse144 + cse154;
  jac(1, 4) = cse119*cse180 - cse137*cse190 + cse157 + cse162 + cse164 + 2*cse165 + 2*cse174 + 4*cse176 + cse177*state.xn[17] + cse181*cse184 + cse185*cse186 + cse189*state.xn[8] + cse194 + cse202;
  jac(1, 5) = cse119*cse207 + cse164 + 2*cse206 + cse210 + cse213;
  jac(1, 7) = cse218;
  jac(1, 8) = cse222;
  jac(1, 9) = cse189*state.xn[3];
  jac(1, 10) = -cse223;
  jac(1, 11) = cse120 + cse225;
  jac(1, 12) = cse228 + cse229 + cse232;
  jac(1, 15) = cse233 + cse234 + cse235 + cse237 + cse238;
  jac(1, 16) = cse238 + cse244;
  jac(1, 17) = cse119*cse250 + cse132*state.xn[2] - 1.8000000000000002e-14*cse163 + 2*cse249 + cse256;
  jac(1, 18) = cse177*state.xn[3] - cse257 + cse259;
  jac(1, 19) = cse312*(-cse141*cse291 + 9.8669999999999991e-12*cse187*cse288*state.xn[3]*state.xn[8] - cse262 - cse263 - cse266 + 2*cse275*cse30 - cse276 + 2*cse279*state.xn[0]*state.xn[3] - 1.5566492609718528e-6*cse280 + cse281 - 5.5769999999999994e-13*cse284 - cse286 + cse287 + cse289 - cse295 - cse311);
  jac(1, 20) = cse186*cse313 + cse317 + cse322;
  jac(1, 22) = cse184*cse313 + cse323 + cse324 + cse325;
  jac(2, 1) = cse99;
  jac(2, 2) = -cse111;
  jac(2, 3) = -cse153;
  jac(2, 4) = cse157 + cse199 + cse326;
  jac(2, 5) = cse209 + cse212 + cse326;
  jac(2, 7) = cse214;
  jac(2, 11) = cse327;
  jac(2, 12) = -cse231;
  jac(2, 15) = cse327;
  jac(2, 16) = cse327;
  jac(2, 17) = cse254 - cse255;
  jac(2, 18) = cse257;
  jac(2, 19) = cse312*(-cse262 + cse297 + cse299 - cse300 - cse303 - cse328);
  jac(2, 20) = -cse317;
  jac(2, 22) = -cse325;
  jac(3, 1) = cse330 - cse367 + cse64 - cse82 + cse87 + cse98;
  jac(3, 2) = -cse110 - cse369;
  jac(3, 3) = -cse154 - cse382 - cse401 - cse408 - cse409 - cse424;
  jac(3, 4) = -cse202 - cse425 + cse426 + cse427 - cse431 - cse437 - cse442 - cse443;
  jac(3, 5) = -cse213 - cse425 - cse430 - cse436 - cse445;
  jac(3, 6) = cse446;
  jac(3, 7) = -cse448;
  jac(3, 8) = cse458 + cse461;
  jac(3, 9) = -cse464;
  jac(3, 10) = cse223;
  jac(3, 11) = -cse225 - cse369;
  jac(3, 12) = cse469;
  jac(3, 13) = cse472;
  jac(3, 14) = cse474;
  jac(3, 15) = cse475;
  jac(3, 16) = cse475;
  jac(3, 17) = -cse256 - cse476 - cse477 - cse478 - cse480 - cse481 - cse482 - cse483;
  jac(3, 18) = cse485;
  jac(3, 19) = cse312*(4.2899999999999994e-13*cse284 + cse292 + cse306 + cse311 + cse487 + cse488 + cse489 + cse492 + cse493 - cse495 + cse496 + cse497 - cse499 - cse500 - cse501 - cse502 - cse503);
  jac(3, 20) = -cse317 + cse505 - cse508 - cse511 - cse514;
  jac(3, 22) = -cse325 + cse515 - cse516 - cse517 - cse520;
  jac(4, 1) = 3.0000000000000002e-15*cse10*d2g - cse521 - cse522 - cse523 - cse524 - cse526 - cse527;
  jac(4, 2) = cse530;
  jac(4, 3) = cse140 - cse142 - cse143 + cse423 + cse531*state.xn[16];
  jac(4, 4) = -cse117 - cse165 - cse174 - cse175*cse9 - cse433 - cse532 - cse535 - cse537 - cse538 - cse539 - cse540;
  jac(4, 5) = -cse117 - cse206 - cse210 - cse444 - cse541;
  jac(4, 7) = -cse214 - cse542;
  jac(4, 8) = cse457;
  jac(4, 9) = -cse462;
  jac(4, 10) = cse543;
  jac(4, 11) = cse530;
  jac(4, 12) = cse468;
  jac(4, 15) = -cse237 - cse529;
  jac(4, 16) = cse530;
  jac(4, 17) = -cse249 + cse480 + cse531*state.xn[2] + cse544 - cse545 + cse547;
  jac(4, 18) = cse485;
  jac(4, 19) = cse312*(-cse261 - 2.3622303960273254e-7*cse280 + cse306 - cse548 - cse549 - cse550 - cse551 - cse552 - cse554);
  jac(4, 20) = -cse555;
  jac(4, 22) = -cse556;
  jac(5, 1) = cse527;
  jac(5, 2) = cse368;
  jac(5, 4) = -cse162 + cse536 + cse538;
  jac(5, 5) = -cse208 - cse209 - cse445;
  jac(5, 7) = cse542;
  jac(5, 11) = cse368;
  jac(5, 15) = cse368;
  jac(5, 16) = cse368;
  jac(5, 17) = cse479;
  jac(5, 19) = -cse312*cse553;
  jac(6, 1) = cse341;
  jac(6, 2) = cse341;
  jac(6, 3) = cse382;
  jac(6, 4) = cse557;
  jac(6, 5) = cse425;
  jac(6, 6) = -cse446;
  jac(6, 7) = cse559;
  jac(6, 10) = cse560;
  jac(6, 11) = cse341;
  jac(6, 15) = cse341;
  jac(6, 16) = cse561;
  jac(6, 17) = cse481;
  jac(6, 19) = cse312*cse563;
  jac(6, 20) = cse511;
  jac(6, 22) = cse517;
  jac(7, 1) = cse564;
  jac(7, 2) = cse564;
  jac(7, 3) = -cse382;
  jac(7, 4) = -cse557;
  jac(7, 5) = -cse425;
  jac(7, 6) = cse446;
  jac(7, 7) = -cse559;
  jac(7, 10) = -cse560;
  jac(7, 11) = cse564;
  jac(7, 15) = cse564;
  jac(7, 16) = -cse561;
  jac(7, 17) = -cse481;
  jac(7, 19) = -cse312*cse563;
  jac(7, 20) = -cse511;
  jac(7, 22) = -cse517;
  jac(8, 1) = cse1 + cse568;
  jac(8, 2) = cse568;
  jac(8, 3) = cse401;
  jac(8, 4) = cse435 + cse570 + cse571*state.xn[8] + cse574;
  jac(8, 5) = cse574;
  jac(8, 8) = -cse222 - cse457 - cse461;
  jac(8, 9) = cse463 + cse571*state.xn[3];
  jac(8, 10) = cse577;
  jac(8, 11) = cse568;
  jac(8, 15) = cse568 + cse578;
  jac(8, 16) = -cse240 + cse568;
  jac(8, 17) = -cse477 + cse482 - cse579;
  jac(8, 19) = cse312*(cse263 - 1.287e-13*cse284 + cse294 + cse487 + 2.2769999999999997e-12*cse498 + cse580);
  jac(8, 20) = cse319 + cse514;
  jac(8, 22) = cse323 + cse520 + cse581;
  jac(9, 1) = cse582;
  jac(9, 2) = cse582;
  jac(9, 3) = -cse401;
  jac(9, 4) = -cse437;
  jac(9, 5) = -cse436;
  jac(9, 7) = cse558;
  jac(9, 8) = cse461;
  jac(9, 9) = -cse464 - cse584;
  jac(9, 10) = cse560;
  jac(9, 11) = cse582;
  jac(9, 15) = cse582;
  jac(9, 16) = -cse354 - cse585;
  jac(9, 17) = -cse482;
  jac(9, 19) = cse312*(-cse551 - cse580 - cse587);
  jac(9, 20) = -cse514;
  jac(9, 22) = cse518 - cse519;
  jac(10, 1) = cse566;
  jac(10, 2) = cse566;
  jac(10, 3) = cse146;
  jac(10, 4) = cse588;
  jac(10, 5) = cse572;
  jac(10, 7) = -cse558;
  jac(10, 8) = cse222;
  jac(10, 10) = -cse589 - cse590;
  jac(10, 11) = cse224 + cse566;
  jac(10, 12) = cse230;
  jac(10, 15) = cse591;
  jac(10, 16) = cse240 + cse566;
  jac(10, 17) = -cse546 + cse579;
  jac(10, 19) = -cse295*cse312;
  jac(10, 22) = -cse581;
  jac(11, 3) = -cse146;
  jac(11, 9) = cse584;
  jac(11, 10) = cse589;
  jac(11, 11) = -cse224;
  jac(11, 16) = cse585;
  jac(11, 17) = cse546;
  jac(11, 19) = cse312*(cse292 + cse587);
  jac(12, 1) = -cse566 - cse592;
  jac(12, 2) = -cse102 - cse566;
  jac(12, 3) = cse142 - cse419 + cse421;
  jac(12, 4) = cse192 - cse193 - cse441 - cse588;
  jac(12, 5) = cse573;
  jac(12, 7) = cse558;
  jac(12, 10) = cse590;
  jac(12, 11) = cse567;
  jac(12, 12) = -cse232 - cse593;
  jac(12, 15) = -cse591;
  jac(12, 16) = cse243 + cse567;
  jac(12, 17) = -cse476 - cse579;
  jac(12, 18) = cse257 + cse258;
  jac(12, 19) = cse312*(-cse594 - cse595);
  jac(12, 20) = cse321;
  jac(12, 22) = cse596;
  jac(13, 1) = cse366;
  jac(13, 2) = cse366;
  jac(13, 3) = cse408;
  jac(13, 4) = cse431;
  jac(13, 5) = cse430;
  jac(13, 11) = cse366;
  jac(13, 13) = -cse472;
  jac(13, 14) = cse473;
  jac(13, 15) = cse366;
  jac(13, 16) = cse366;
  jac(13, 17) = cse483;
  jac(13, 19) = cse312*cse597;
  jac(13, 20) = -cse598;
  jac(13, 22) = -cse599;
  jac(14, 1) = cse600;
  jac(14, 2) = cse600;
  jac(14, 3) = -cse408;
  jac(14, 4) = -cse431;
  jac(14, 5) = -cse430;
  jac(14, 11) = cse600;
  jac(14, 13) = cse472;
  jac(14, 14) = cse474;
  jac(14, 15) = cse600;
  jac(14, 16) = cse600;
  jac(14, 17) = -cse483;
  jac(14, 19) = -cse312*cse597;
  jac(14, 20) = cse598;
  jac(14, 22) = cse599;
  jac(15, 1) = cse524;
  jac(15, 4) = cse601*state.xn[8];
  jac(15, 8) = cse457;
  jac(15, 9) = cse601*state.xn[3];
  jac(15, 12) = -cse230;
  jac(15, 15) = -cse233 - cse578;
  jac(15, 17) = cse477;
  jac(15, 19) = cse312*(-3.0029999999999999e-13*cse284 + cse486 + 5.3129999999999997e-12*cse498 + cse548);
  jac(15, 20) = -cse319;
  jac(15, 22) = -cse323;
  jac(16, 3) = cse144 + cse422;
  jac(16, 4) = cse442;
  jac(16, 7) = -cse217;
  jac(16, 8) = -cse222;
  jac(16, 9) = -cse584;
  jac(16, 12) = cse593;
  jac(16, 16) = -cse244 - cse585;
  jac(16, 17) = cse476;
  jac(16, 18) = cse259;
  jac(16, 19) = cse312*(cse266 + cse294 + cse552 + cse586 + cse594);
  jac(16, 20) = -cse321;
  jac(16, 22) = -cse324;
  jac(17, 3) = -cse424;
  jac(17, 4) = cse539;
  jac(17, 5) = cse208;
  jac(17, 8) = cse458;
  jac(17, 10) = -cse543;
  jac(17, 12) = cse469;
  jac(17, 17) = -cse478 - cse547;
  jac(17, 19) = cse312*(cse488 + cse554);
  jac(18, 1) = cse592;
  jac(18, 2) = cse102;
  jac(18, 3) = -cse409;
  jac(18, 4) = -cse540;
  jac(18, 7) = cse217;
  jac(18, 12) = cse231;
  jac(18, 16) = cse239;
  jac(18, 18) = -cse257 - cse484;
  jac(18, 19) = cse312*(cse308 + cse562 + cse595);
  jac(19, 1) = 1.602177e-12*crate*cse874 + 1.602177e-12*crate*cse905*state.xn[0] + 3.2043540000000001e-12*crate*cse905*state.xn[3] + 3.0000000000000002e-15*cse10*cse892*d2g + 2.0187430200000004e-26*cse10*cse978*d2g*state.xn[0]*(-cse878*cse947 - cse890*(-cse877/cse981 + cse980) - cse979) - cse1000*cse991 - cse1032 - 7.1777529600000012e-12*cse522 - 7.1777529600000012e-12*cse523 - 7.1777529600000012e-12*cse526 + 2.5634832e-11*cse535*cse940*state.xn[3]*(-cse938*cse947 + cse941) - cse907*cse937 - cse963*(-cse907*cse924 - cse933*(cse950*cse953 + cse958*(cse907*cse928 - cse954*cse956)) - cse961) + 4.39089220777172e-29*cse965*cse969*cse971*cse975*cse983*state.xn[8] - 4.39089220777172e-29*cse965*cse977 - 2.1789607199999999e-11*cse98 - cse986*(-cse972*cse982 + cse982*cse984) - state.xn[11]*(cse651*cse668 + cse673*cse674 + cse677*cse678 + cse677*cse679) - state.xn[3]*((cse710) ? (
   0
)
: (
   cse726*cse755
)) - state.xn[7]*((cse859) ? (
   cse830*cse847 + cse852*cse853 + cse856*cse857 + cse856*cse858
)
: (
   0
));
  jac(19, 2) = 1.602177e-12*crate*cse1035*state.xn[0] + 3.2043540000000001e-12*crate*cse1035*state.xn[3] + 2.0187430200000004e-26*cse10*cse1040*cse978*d2g*state.xn[0] - cse1032 - cse1033 - cse1034 + 2.5634832e-11*cse1037*cse535*cse940*state.xn[3] - cse1039 - state.xn[3]*((cse710) ? (
   0
)
: (
   cse729*cse755
));
  jac(19, 3) = 5.6705896183685846e-35*state.c_hat*cse10*cse1027*cse1055*cse67*d2g*(8.429953871621411e-15*cse1057*cse133 + 1.2210344531082231e-12*cse1059*cse133) + 4.22974728e-11*crate*cse1054*cse65*cse860*cse862*cse981 + 8.4594945600000001e-11*crate*cse1054*cse65*cse860*cse862*state.xn[0]*state.xn[3] - cse1000*cse1041*state.xn[0] + 6.0732876035249664e-19*cse1042*cse1043*cse1044*cse994*cse998*state.xn[0]*state.xn[2] - cse1045 - cse1047*(-cse1041*cse996 + 5.9301787935659931e-17*cse1042*cse1043*cse1044*cse994*state.xn[2]) - 2.4067011396745388e-39*cse1049*cse1053*cse379*cse68 - cse1061*(2.8857630567877724e-12*state.c_hat*cse1019*cse1048*cse1051*cse2*cse67 - cse1017*cse1049*cse1060) - 2.1789607199999999e-11*cse147 + 2.6241084038530201e-25*cse264*cse969*cse971*cse975*cse983*state.xn[8] - 2.6241084038530201e-25*cse264*cse977 - cse913*cse937 - cse963*(-cse913*cse924 - cse913*cse933*cse957) - cse986*(-cse1062*cse972 + cse1062*cse984) - state.xn[11]*(cse1066*cse668 + cse1068*cse674 + cse1069*cse678 + cse1069*cse679) - state.xn[3]*((cse710) ? (
   0
)
: (
   cse740*cse755
)) - state.xn[7]*((cse859) ? (
   cse1076*cse847 + cse1079*cse853 + cse1082*cse857 + cse1082*cse858
)
: (
   0
));
  jac(19, 4) = 1.1341179236737169e-34*state.c_hat*cse1027*cse1031*cse67*d2g + 1.602177e-12*crate*cse1105*state.xn[0] + 3.2043540000000001e-12*crate*cse1105*state.xn[3] + 3.2043540000000001e-12*crate*cse874 + 2.0187430200000004e-26*cse10*cse978*d2g*state.xn[0]*(-cse1096*cse878 - cse1098) - cse1084 - cse1086 - cse1088 - cse1090*cse968 + cse1091*cse535 - cse1092*cse1093 - 7.1777529600000012e-12*cse165 - 7.1777529600000012e-12*cse174 - 1.4355505920000002e-11*cse176 - 7.1777529600000012e-12*cse532 + 2.5634832e-11*cse535*cse940*state.xn[3]*(-cse1096*cse938 + 2*cse941) + 6.0000000000000005e-15*cse892*d2g*state.xn[0] - cse937 - cse963*(-cse1108 - cse924 - cse933*(cse1106*cse953 + cse958*(-cse1107*cse954 + cse928))) - cse986*(-cse1092*cse1094 - cse968*cse973) - state.xn[11]*(cse1113*cse668 + cse1118*cse674 + cse1120*cse678 + cse1120*cse679) - state.xn[3]*((cse710) ? (
   0
)
: (
   cse727*cse755
)) - state.xn[7]*((cse859) ? (
   cse1124*cse847 + cse1129*cse853 + cse1132*cse857 + cse1132*cse858
)
: (
   0
)) - ((cse710) ? (
   0.0
)
: (
   cse1109*cse751
));
  jac(19, 5) = -cse1084 - cse1086 - cse1088 + 1.1341179236737169e-34*cse1135 + cse1138*(-cse1098 - cse1136*cse878) + cse1139*cse1140 + cse1139*cse1141 + cse1143*(-cse1136*cse938 + 2*cse35*cse889*cse938) - 7.1777529600000012e-12*cse206 - 7.1777529600000012e-12*cse541 + cse57*cse892 - cse963*(-cse1108 - cse933*(-cse1038*cse1107 + cse1106*cse951*cse952));
  jac(19, 6) = cse1144;
  jac(19, 7) = cse1144;
  jac(19, 8) = -((cse859) ? (
   cse840*cse847 + cse844*cse853
)
: (
   0.0
));
  jac(19, 9) = -cse1089*cse985;
  jac(19, 10) = -cse914*cse936 - cse963*(cse1145*cse914*cse959*cse960 - cse932*(cse1145*cse911*cse925*cse959*cse960 - cse1146*cse926) - cse933*(-cse1038*cse1146 + cse1145*cse951*cse952));
  jac(19, 11) = cse1148;
  jac(19, 12) = -cse661*cse668 - cse665*cse674;
  jac(19, 15) = cse1148;
  jac(19, 16) = cse1148;
  jac(19, 17) = -5.6994379667262474e-29*cse1083 - 9.6000000000000005e-32*cse1085 - 300.0*cse1087 + 1.7011768855105754e-34*cse1135 + cse1138*(-9.0000000000000011e-15*cse1097*d2g - cse1149*cse878 + 3*cse35*cse878*cse889) + cse1140*cse1150 + cse1141*cse1150 + cse1143*(-cse1149*cse938 + 3*cse35*cse889*cse938) - 7.1777529600000012e-12*cse249 + cse544*cse892 - 7.1777529600000012e-12*cse545 - cse963*(cse1151*cse914*cse959*cse960 - cse932*(cse1151*cse911*cse925*cse959*cse960 - cse1152*cse926) - cse933*(-cse1038*cse1152 + cse1151*cse951*cse952));
  jac(19, 19) = cse312*(5.6705896183685846e-35*state.c_hat*cse10*cse1027*cse1055*cse67*d2g*(-4.2149769358107055e-15*cse1057*cse290 - 6.1051722655411154e-13*cse1059*cse290) + 1.8755475162754094e-35*state.c_hat*cse10*cse1031*cse67*d2g*std::pow(state.T, -0.85299999999999998) - cse1001*cse1008*psigd_coll_interp2d_partial_1(cse10, state.T) - cse1011*(1.6000000000000001e-32*cse1004*cse277*cse379 + cse1006) + 7.8348995947646122e-14*cse1043*cse1156*cse993*cse994*cse998*state.xn[0]*state.xn[2] - cse1046*cse1158*cse998 - cse1047*(7.6502807803182096e-12*cse1043*cse1156*cse993*cse994*state.xn[2] - 0.0015792330470357991*cse1153*cse990 - 2.0007175426973804e-9*cse1157*cse995 - 1.3338116951315871e-8*cse1158*cse991) - cse1061*(cse1020*cse3 - 6.3340486037177084e-11*cse1052*cse1160) - cse1061*(2.835294809184292e-11*cse1060*cse1160 - 1.4428815283938862e-12*cse1159*cse1161*cse68) - cse1090*cse1166 - cse1093*cse1168 - cse1137*cse1170*cse878*cse945 - cse1143*cse1170*cse938*cse946 - 1.6173435610483583e-5*cse1153*cse1154*cse998 - 2.0489994438957239e-11*cse1154*cse1155*cse999 - 5.2835524837844135e-18*cse1162*cse975*state.xn[8] - cse1167*cse937 - 2.1789607199999999e-11*cse297 - 7.1777529600000012e-12*cse549 - 7.1777529600000012e-12*cse550 - 1000*cse908*std::pow(cse909, 999)*cse935*cse962 - cse963*(-cse1167*cse924 + cse1171*cse1172*cse927*cse931 + cse1171*cse1172 - cse1174*cse914*cse959 - cse932*(-cse1171*cse924*cse925 - cse1174*cse911*cse925*cse959 - cse1175*cse926) - cse933*(cse952*(cse1173*cse951 + alpha_co_interp2d_partial_0(state.T, cse922)) + cse958*(cse1167*cse928 - cse1175*cse954))) - cse986*(-cse1094*cse1168 - 182.41999999999999*cse1162 - cse1166*cse973) - state.xn[11]*(cse1195*cse668 + cse1199*cse674 + cse1200*cse678 + cse1200*cse679) - state.xn[3]*((cse710) ? (
   0
)
: (
   cse1109*cse1177 + cse751*cse754*(-cse1177*cse752 - cse742*cse751*(-cse728*(3.0182298984217999*cse1178*cse697*cse719*cse743 + 4.8223900769399082*cse1178*cse743*cse8 - 3.5529549287336835*cse1180 - 0.3872755400043702*cse1181 - 1.3735579540080118*cse1182) - cse730*(0.0708789387907936*cse1180 + 3.7035619079275177*cse1181 - 4.6974781513675152*cse1182 + 3.6184459289309552*cse1183 - 1.6316107607322889*cse1184) - cse732*(-3.7541549062629067*cse1180 + 2.2094886994529301*cse1181 - 1.528565035159452*cse1182 + 5.0409049417480247*cse1183 + 2.0057552335975877*cse1184) - state.xn[0]*((cse722) ? (
      cse7*cse721*(116.29033200000001*cse1180 + 19.239805*cse1181 + 80.639324000000002*cse1182 + 37.383713*cse1183 + 145.968309*cse1184)
   )
   : ((cse724) ? (
      cse723*(8.2184944748967013*cse1178*cse743*cse8 - 52.189748993977005*cse1180 - 48.951834264235494*cse1181 - 196.44057098336626*cse1182 - 192.38155095558542*cse1184)
   )
   : (
      cse725*(-17.135767342440825*cse1180 + 17.889115159724135*cse1181 - 50.756388852554174*cse1182 + 10.695627721640689*cse1183 + 41.010708268606813*cse1184)
   ))) - state.xn[2]*((cse682) ? (
      cse738*(445.51869310442481*cse1180 + 11324.985706577943*cse1181 + 6766.9891340104432*cse1182 + 38.719649225812766*cse1183 + 2370.6849681533818*cse1184 + 1300.6686834345674*cse1185 + 5869.3152370465659*cse1186 + 11077.448644792414*cse1187)
   )
   : (
      cse739*(4.297112944704045*cse1180 - 101.40241318979159*cse1181 - 43.540996231705549*cse1182 + 3.8689780091986448*cse1183 + 27.91190909651122*cse1184 - 117.34186576967058*cse1185 + 103.76942484394611*cse1186 + 123.18901581604325*cse1187)
   ))))
)) - state.xn[7]*((cse859) ? (
   cse1209*cse847 + cse1213*cse853 + cse1214*cse857 + cse1214*cse858
)
: (
   0
)));
  jac(19, 20) = -cse1061*(-3.4616190918391922e-10*cse1224 + 6.8021597403010268e-9*cse1225) + cse1091*cse555 + 6.8021597403010268e-33*cse1216 + 3.9526908635666079e-36*cse1217 + 2.8869600343148412e-37*cse1218 + cse1222*(-1.011215705901057e-12*cse1220 - 1.4646927316956432e-10*cse1221) - 3.1317711691727707e-38*cse1226;
  jac(19, 22) = -cse1061*(-1.0768077006188295e-10*cse1224 + 2.1159514651578494e-9*cse1225) + cse1091*cse556 + 2.1159514651578495e-33*cse1216 + 1.2295656590548808e-36*cse1217 + 8.9804820052495635e-38*cse1218 + cse1222*(-3.1455940997899587e-13*cse1220 - 4.5562275070892329e-11*cse1221) - 9.4206367748569615e-29*cse1226;
  jac(20, 1) = cse1229;
  jac(20, 2) = cse1229;
  jac(20, 4) = -cse1230 - cse1231*cse185;
  jac(20, 5) = -cse1230;
  jac(20, 11) = cse1229;
  jac(20, 13) = -cse470;
  jac(20, 15) = -cse1228 - cse234;
  jac(20, 16) = -cse1228 - cse241;
  jac(20, 17) = -3.7583398381383454e-21*cse1227;
  jac(20, 20) = cse1234;
  jac(21, 1) = cse1237;
  jac(21, 2) = cse1237;
  jac(21, 4) = -cse1231*cse1235 - cse1239;
  jac(21, 5) = -cse1239;
  jac(21, 11) = cse1237;
  jac(21, 13) = -2.1635567548454671e-17*cse1235;
  jac(21, 15) = -4.2188032355736621e-18*cse1235 - cse1236;
  jac(21, 16) = -1.7141251340095381e-18*cse1235 - cse1236;
  jac(21, 17) = -3.7583398381383454e-21*cse1238;
  jac(21, 21) = cse1234;
  jac(22, 1) = cse1242;
  jac(22, 2) = cse1242;
  jac(22, 4) = -cse1243 - cse1244*cse181;
  jac(22, 5) = -cse1243;
  jac(22, 8) = -cse460;
  jac(22, 10) = -cse576;
  jac(22, 11) = cse1242;
  jac(22, 13) = -cse471;
  jac(22, 15) = -cse1241 - cse235;
  jac(22, 16) = -cse1241 - cse242;
  jac(22, 17) = -6.175361757103777e-21*cse1240;
  jac(22, 22) = cse1246;
  jac(23, 1) = cse1249;
  jac(23, 2) = cse1249;
  jac(23, 4) = -cse1244*cse1247 - cse1251;
  jac(23, 5) = -cse1251;
  jac(23, 8) = -cse1247*cse459;
  jac(23, 10) = -cse1247*cse575;
  jac(23, 11) = cse1249;
  jac(23, 13) = -1.913494189605312e-17*cse1247;
  jac(23, 15) = -1.168047934563505e-18*cse1247 - cse1248;
  jac(23, 16) = -4.1730320353928637e-18*cse1247 - cse1248;
  jac(23, 17) = -6.175361757103777e-21*cse1250;
  jac(23, 23) = cse1246;
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
