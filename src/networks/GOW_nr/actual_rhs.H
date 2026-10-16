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

AMREX_GPU_HOST_DEVICE AMREX_INLINE void actual_rhs(const burn_t &state, amrex::Array1D<amrex::Real, 1, neqs> &ydot)
{

	using namespace Rates;
	const amrex::Real ncol_H2 = network_rp::ncol_H2;
	const amrex::Real ncol_CO = network_rp::ncol_CO;
	const amrex::Real ncol_C = network_rp::ncol_C;
	const amrex::Real ncol_H = network_rp::ncol_H;
	const amrex::Real d2g = network_rp::dust2gas_ratio;
	const amrex::Real gradv = 3.0 * 3.0e-14;
	const amrex::Real vdisp = 3.0 * 1.0e5 / pow(2, 0.5);
	const amrex::Real av = network_rp::Av;

	// $JAFF REPEAT idx, rhs, cse IN rhses $[REPLACE radeden\[\s*(\d+)\s*\] state.rn[2*\1] REPLACE photden\[\s*(\d+)\s*\] state.rn[2*\1] REPLACE
	// rflux\[\s*(\d+)\s*\] state.rn[2*\1+1] REPLACE nden\[\s*(\d+)\s*\] state.xn[\1] REPLACE tgas state.T REPLACE c_hat state.c_hat THERMAL dedt DEDT_TYPE
	// specific]$
	const amrex::Real cse0 = 6.3999999999999996e-10 * state.xn[0] * state.xn[4];
	const amrex::Real cse1 = -cse0;
	const amrex::Real cse2 = std::sqrt(state.T);
	const amrex::Real cse3 = cse2 / state.xn[2];
	const amrex::Real cse4 = 0.00011320754716981131 * cse3 * state.xn[3];
	const amrex::Real cse5 = std::min(10000000000.0, cse4);
	const amrex::Real cse6 = 1.0 / (cse5 + 1);
	const amrex::Real cse7 = cse6 * state.xn[17] * state.xn[3];
	const amrex::Real cse8 = 1.6000000000000001e-9 * cse7;
	const amrex::Real cse9 = cse5 * cse8;
	const amrex::Real cse10 = std::pow(state.T, -0.19);
	const amrex::Real cse11 = cse10 * state.xn[11] * state.xn[16];
	const amrex::Real cse12 = cse11 * cse6;
	const amrex::Real cse13 = 1.99e-9 * cse12;
	const amrex::Real cse14 = state.xn[11] * state.xn[15];
	const amrex::Real cse15 = 3.5000000000000002e-11 * cse14;
	const amrex::Real cse16 = chi * std::exp(-2.6600000000000001 * av) * state.xn[15];
	const amrex::Real cse17 = 3.7999999999999998e-10 * cse16;
	const amrex::Real cse18 = 1.0 / state.T;
	const amrex::Real cse19 = std::pow(state.T, -0.33900000000000002) * std::exp(0.11 * cse18) * state.xn[15] * state.xn[7];
	const amrex::Real cse20 = 7.9513696852843427e-10 * cse19;
	const amrex::Real cse21 = M_LN10;
	const amrex::Real cse22 = 1.0 / cse21;
	const amrex::Real cse23 = 2 * state.xn[3];
	const amrex::Real cse24 = cse23 + state.xn[0] + state.xn[10] + state.xn[14] + state.xn[15] + 3 * state.xn[16] + state.xn[1] + 2 * state.xn[4];
	const amrex::Real cse25 = std::log((1.0 / 10000.0) * state.T);
	const amrex::Real cse26 = std::pow(cse21, -2);
	const amrex::Real cse27 = std::pow(cse25, 2) * cse26;
	const amrex::Real cse28 = 1.0 / cse24;
	const amrex::Real cse29 = cse28 * state.xn[3];
	const amrex::Real cse30 = cse28 * state.xn[0];
	const amrex::Real cse31 = std::pow(10, 0.41599999999999998 * cse22 * cse25 + 0.32700000000000001 * cse27 - 3) * cse30 +
				  std::pow(10, 1.3 * cse22 * cse25 - 1.6200000000000001 * cse27 - 4.8449999999999998) * cse29;
	const amrex::Real cse32 = cse24 * ((cse31 < 1.0e-50) ? (1.0e-50) : (cse31));
	const amrex::Real cse33 = cse22 / (cse32 + 1);
	const amrex::Real cse34 = cse32 * cse33;
	const amrex::Real cse35 = state.T > 700.0;
	const amrex::Real cse36 =
	    ((cse35)
		 ? (std::pow(10, cse33 * std::log(5.9959999999999997e-30 * std::pow(state.T, 4.1881000000000004) *
						  std::pow(6.7610000000000002e-6 * state.T + 1, -5.6881000000000004) * std::exp(-54657.400000000001 * cse18)) +
				     cse34 * std::log(1.3000000000000001e-9 * std::exp(-53300 * cse18))))
		 : (0.0)) *
	    std::pow(state.xn[3], 2);
	const amrex::Real cse37 =
	    (7.5e-10 * std::pow(state.T, -0.45800000000000002) + 4.9899999999999997e-11 * std::pow(state.T, 0.40500000000000003)) * state.xn[0] * state.xn[17];
	const amrex::Real cse38 = ((cse35) ? (std::pow(10, cse33 * std::log(6.6699999999999996e-12 * cse2 * std::exp(-63590 * cse18 - 1)) +
							       cse34 * std::log(3.5199999999999998e-9 * std::exp(-43900 * cse18))))
					   : (0.0)) *
				  state.xn[0];
	const amrex::Real cse39 = std::pow(state.T, -0.52000000000000002);
	const amrex::Real cse40 = cse39 * state.xn[16] * state.xn[2];
	const amrex::Real cse41 = cse24 * d2g;
	const amrex::Real cse42 = std::exp(-227 * cse18);
	const amrex::Real cse43 =
	    4.0000000000000001e-10 * std::pow(state.T, 0.0066899999999999998) + 1.1000000000000001e-11 * std::pow(state.T, 0.51700000000000002);
	const amrex::Real cse44 = cse42 * cse43 * state.xn[11] * state.xn[1];
	const amrex::Real cse45 = std::sqrt(2.0000000000000002e-15 * ncol_H2 + 1);
	const amrex::Real cse46 = 0.96499999999999997 / std::pow(1.4142135623730951e-10 * ncol_H2 / vdisp + 1, 2) +
				  0.035000000000000003 * std::exp(-0.00084999999999999995 * cse45) / cse45;
	const amrex::Real cse47 = std::exp(-4.1799999999999997 * av);
	const amrex::Real cse48 = chi * cse46 * cse47;
	const amrex::Real cse49 = std::pow(state.T, -1.3);
	const amrex::Real cse50 = std::exp(-23 * cse18);
	const amrex::Real cse51 = cse49 * cse50 * state.xn[3] * state.xn[8];
	const amrex::Real cse52 = state.xn[11] * state.xn[14];
	const amrex::Real cse53 = 7.0000000000000004e-11 * cse52;
	const amrex::Real cse54 = chi * std::exp(-2.1200000000000001 * av) * state.xn[14];
	const amrex::Real cse55 = 9.0999999999999996e-10 * cse54;
	const amrex::Real cse56 = cse53 + cse55;
	const amrex::Real cse57 = 26.218500000000002 * std::sqrt(3) * std::sqrt(cse18) + 0.62;
	const amrex::Real cse58 = cse57 * state.xn[15];
	const amrex::Real cse59 = 1.3500000000000001e-9 * cse58 * state.xn[6];
	const amrex::Real cse60 = state.xn[3] * state.xn[6];
	const amrex::Real cse61 = 1.2599999999999999e-13 * cse60 * std::exp(-22.5 * cse18);
	const amrex::Real cse62 = cse59 + cse61;
	const amrex::Real cse63 = std::pow(state.T, 0.26000000000000001) * state.xn[0] * state.xn[14];
	const amrex::Real cse64 = 2.8142984982406731e-11 * cse63;
	const amrex::Real cse65 = std::pow(state.T, 0.042000000000000003);
	const amrex::Real cse66 = std::exp(-2.145922746781116e-5 * state.T);
	const amrex::Real cse67 = 1.7599999999999999e-9 * cse65 * cse66 * state.xn[3] * state.xn[4];
	const amrex::Real cse68 = -cse64 + cse67;
	const amrex::Real cse69 = crate * state.xn[9];
	const amrex::Real cse70 = 6.5199999999999996 * cse69;
	const amrex::Real cse71 = crate * (2.2999999999999998 * cse29 + 1.5 * cse30);
	const amrex::Real cse72 = cse71 * state.xn[0];
	const amrex::Real cse73 = std::log(8.6173000000000003e-5 * state.T);
	const amrex::Real cse74 = state.xn[0] * state.xn[2];
	const amrex::Real cse75 =
	    cse74 * ((cse35) ? (5.9078447946382801e-70 * std::pow(state.T, 13.536555999999999) *
				std::exp(-2.0391498499999999e-6 * std::pow(cse73, 8) + 0.000111954395 * std::pow(cse73, 7) -
					 0.00263197617 * std::pow(cse73, 6) + 0.034825597700000002 * std::pow(cse73, 5) -
					 0.28770560000000001 * std::pow(cse73, 4) + 1.56315498 * std::pow(cse73, 3) - 5.7393287500000003 * std::pow(cse73, 2)))
			     : (0.0));
	const amrex::Real cse76 = std::pow(state.T, -0.64000000000000001);
	const amrex::Real cse77 = 1.05849862170431e-5 * cse76 * state.xn[10] * state.xn[2];
	const amrex::Real cse78 = std::pow(cse18, 1.5);
	const amrex::Real cse79 = std::pow(114.81344434488582 * std::pow(cse18, 0.40699999999999997) + 1, -2.242);
	const amrex::Real cse80 = 4.8813573300130879e-6 * cse78 * cse79 * state.xn[1] * state.xn[2];
	const amrex::Real cse81 = std::exp(-1.8700000000000001 * av);
	const amrex::Real cse82 = chi * cse81;
	const amrex::Real cse83 = cse3 * cse82;
	const amrex::Real cse84 = 1.7 * cse83;
	const amrex::Real cse85 = std::log(state.T);
	const amrex::Real cse86 = 1.0 / (1.6774414051711492e-5 * std::pow(cse83, 1.3779999999999999) *
					     (508.69999999999999 * std::pow(cse84, -1.102e-5 * cse85 - 0.4723) * std::pow(state.T, 0.015859999999999999) + 1) +
					 1);
	const amrex::Real cse87 = 1.2250000000000001e-11 * cse41 * cse86 * state.xn[1];
	const amrex::Real cse88 = -cse70 - cse72 - cse75 + cse77 + cse80 + cse87;
	const amrex::Real cse89 = cse37 - cse44;
	const amrex::Real cse90 = cse23 * cse71;
	const amrex::Real cse91 = std::pow(state.T, -0.62) * state.xn[13] * state.xn[2];
	const amrex::Real cse92 = 1.4595000290776864e-10 * cse91;
	const amrex::Real cse93 = std::exp(-2.6099999999999999 * av);
	const amrex::Real cse94 = 4.4999999999999998e-9 * chi * cse93 * state.xn[12];
	const amrex::Real cse95 = cse41 * state.xn[13] /
				  (1.534817233071468e-7 * std::pow(cse83, 1.8740000000000001) *
				       (43750.0 * std::pow(cse84, -7.538e-5 * cse85 - 0.89639999999999997) * std::pow(state.T, 1.635e-6) + 1) +
				   1);
	const amrex::Real cse96 = 2.1659999999999999e-12 * cse95;
	const amrex::Real cse97 = 16800.0 * crate * cse29 * state.xn[12];
	const amrex::Real cse98 =
	    1.0537934368240927e-9 * std::pow(cse18, 0.00231) +
	    std::pow(state.T, -1.5) * (0.00015100000000000001 * std::exp(-8110.0 * cse18) + 1.31e-7 * std::exp(-26.600000000000001 * cse18) +
				       3.4e-8 * std::exp(-7.6200000000000001 * cse18) + 6.9699999999999997e-9 * std::exp(-1.3799999999999999 * cse18));
	const amrex::Real cse99 = cse98 * state.xn[16] * state.xn[7];
	const amrex::Real cse100 = cse28 * state.xn[2];
	const amrex::Real cse101 = ((cse100 < 1.0e-50) ? (10000000000.0) : (cse4));
	const amrex::Real cse102 = 1.0 / (cse101 + 1);
	const amrex::Real cse103 = 1.99e-9 * cse101 * cse102 * cse11;
	const amrex::Real cse104 = cse103 + cse13 + 1.3007021411393064e-6 * cse40 + cse99;
	const amrex::Real cse105 = crate * state.xn[5];
	const amrex::Real cse106 = std::pow(state.T, -0.5);
	const amrex::Real cse107 = cse22 * cse85;
	const amrex::Real cse108 = cse107 * (cse107 * (0.044330000000000001 * cse107 - 0.28520000000000001) - 1.6759999999999999) + 11.19;
	const amrex::Real cse109 =
	    1.0 / (7.104740267838494e-7 * std::pow(cse83, 1.512) *
		       (5115.0 * std::pow(cse84, -5.4939999999999998e-7 * cse85 - 0.49559999999999998) * std::pow(state.T, 3.9029999999999999e-7) + 1) +
		   1);
	const amrex::Real cse110 = -1.1000000000000001 * cse105 + 9.9999999999999994e-12 * cse106 * cse108 * state.xn[2] * state.xn[6] +
				   5.5720000000000001e-12 * cse109 * cse41 * state.xn[6];
	const amrex::Real cse111 = crate * state.xn[7];
	const amrex::Real cse112 = 3.8500000000000001 * cse111;
	const amrex::Real cse113 = 0.15970000000000001 * std::exp(-49550.0 * cse18);
	const amrex::Real cse114 = (1.6339999999999999e-6 * std::exp(-15230.0 * cse18) + 9.7930000000000004e-9 * std::exp(-73.799999999999997 * cse18) +
				    6.3460000000000002e-9 * std::exp(-12.17 * cse18)) /
				       std::pow(state.T, 3.0 / 2.0) +
				   2.4460185352936309e-10 * std::pow(0.00071740367932898903 * cse2 + 1, -cse113 - 1.7848999999999999) *
				       std::pow(12.244387999457523 * cse2 + 1, cse113 - 0.21509999999999996) / cse2;
	const amrex::Real cse115 = cse114 * state.xn[2] * state.xn[8];
	const amrex::Real cse116 =
	    1.0 / (0.011078790689599145 * std::pow(cse83, 1.1279999999999999) *
		       (433.10000000000002 * std::pow(cse84, -0.00013329999999999999 * cse85 - 0.81200000000000006) * std::pow(state.T, 0.04845) + 1) +
		   1);
	const amrex::Real cse117 = 4.5579999999999993e-11 * cse116 * cse41 * state.xn[8];
	const amrex::Real cse118 = cse111 * cse29;
	const amrex::Real cse119 = 1040.0 * cse118;
	const amrex::Real cse120 = 2.808e-22 * ncol_H2;
	const amrex::Real cse121 =
	    chi * std::exp(-3.7599999999999998 * av) * std::exp(-cse120) * std::exp(-1.6000000000000001e-17 * ncol_C) * state.xn[7] / (cse120 + 1);
	const amrex::Real cse122 = 3.4999999999999998e-10 * cse121;
	const amrex::Real cse123 = -cse112 + cse115 + cse117 - cse119 - cse122;
	const amrex::Real cse124 = 3.2999999999999996e-13 * cse51;
	const amrex::Real cse125 = cse8 + cse9;
	const amrex::Real cse126 = cse124 + cse125;
	const amrex::Real cse127 = -cse99;
	const amrex::Real cse128 = -cse103;
	const amrex::Real cse129 = state.xn[16] * state.xn[9];
	const amrex::Real cse130 = 1.6999999999999999e-9 * cse129;
	const amrex::Real cse131 = cse38 * state.xn[3];
	const amrex::Real cse132 = 5.6999999999999997e-11 * cse48;
	const amrex::Real cse133 = 7.2000000000000002e-15 * cse60;
	const amrex::Real cse134 = cse1 + cse133 + cse90;
	const amrex::Real cse135 = -cse67;
	const amrex::Real cse136 = 1.6000000000000001e-9 * state.xn[6] * state.xn[9];
	const amrex::Real cse137 = cse110 + cse133 + cse136 + cse62;
	const amrex::Real cse138 = cse127 + cse64;
	const amrex::Real cse139 = 184.0 * cse29 * cse69;
	const amrex::Real cse140 = 2.4e-10 * chi * fshield_co_interp2d(ncol_CO, ncol_H2) * std::exp(-3.8799999999999999 * av) * state.xn[9];
	const amrex::Real cse141 = cse139 + cse140 - cse20;
	const amrex::Real cse142 = cse58 * state.xn[8];
	const amrex::Real cse143 = 9.1500000000000005e-10 * cse142;
	const amrex::Real cse144 = cse130 + cse70 - cse77;
	const amrex::Real cse145 = cse136 - cse53;
	const amrex::Real cse146 = cse128 + cse15 + cse17;
	const amrex::Real cse147 = -cse92 + cse94 - cse96 + cse97;
	const amrex::Real cse148 = 1.6737729999999998e-24 * state.xn[0] + 4.8185222061629005e-23 * state.xn[10] + 2.6567630000000005e-23 * state.xn[11] +
				   4.6637070000000001e-23 * state.xn[12] + 4.6636159061629004e-23 * state.xn[13] + 2.1618503000000003e-23 * state.xn[14] +
				   2.8241403000000005e-23 * state.xn[15] + 5.0204080616289998e-24 * state.xn[16] + 2.6566719061629008e-23 * state.xn[17] +
				   1.6728620616289998e-24 * state.xn[1] + 9.1093837099999994e-28 * state.xn[2] + 3.3475459999999996e-24 * state.xn[3] +
				   3.3466350616289996e-24 * state.xn[4] + 6.6464729999999996e-24 * state.xn[5] + 6.6455620616289996e-24 * state.xn[6] +
				   1.9944730000000003e-23 * state.xn[7] + 1.9943819061629003e-23 * state.xn[8] + 4.6512360000000008e-23 * state.xn[9];
	const amrex::Real cse149 = std::pow(state.T, 0.14999999999999999) / (0.00088240812266906664 * std::pow(state.T, 0.65000000000000002) + 1);
	const amrex::Real cse150 = cse149 * state.xn[2];
	const amrex::Real cse151 = std::exp(-118400.0 * cse18) / (4.4460389837719569e-9 * cse150 + 626500000.0);
	const amrex::Real cse152 = cse84 + 50;
	const amrex::Real cse153 = std::log(1.7 * cse2 * cse82 / (state.xn[2] + 1.0e-50) + 50);
	const amrex::Real cse154 = 0.001 * state.T;
	const amrex::Real cse155 = cse154 + 1;
	const amrex::Real cse156 = std::pow(cse155, -2);
	const amrex::Real cse157 = 1.0 / cse155;
	const amrex::Real cse158 = 1.0 / (cse28 * (cse132 + 1.9999999999999999e-7) /
					      (std::pow(10, -2.3900000000000001 * cse156 + 0.055500000000000001 * cse157 - 11.06) * cse30 +
					       std::pow(10, -2.0230000000000001 * cse156 - 3.6709999999999998 * cse157 - 11.08) * cse29) +
					  1);
	const amrex::Real cse159 = std::sqrt(cse18) * state.xn[2];
	const amrex::Real cse160 = (1.0 / 100.0) * state.T;
	const amrex::Real cse161 = std::log(cse160);
	const amrex::Real cse162 = state.T < 500;
	const amrex::Real cse163 = std::pow(state.T, 0.070000000000000007);
	const amrex::Real cse164 = 4.5299999999999998e-6 * cse159 +
				   7.5799999999999997e-10 * std::pow(cse160, 0.0086999999999999994 * cse161 + 0.12809999999999999) * state.xn[0] +
				   (0.75 * ((cse162) ? (1.1000000000000001e-13 * state.T + 5.3300000000000002e-10) : (3.74757785025e-10 * cse163)) +
				    0.25 * ((cse162) ? (3.3000000000000001e-13 * state.T + 4.4299999999999997e-10) : (3.8899728635600003e-10 * cse163))) *
				       state.xn[3];
	const amrex::Real cse165 = cse164 * std::exp(-91.209999999999994 * cse18) / (cse164 + 2.3e-6);
	const amrex::Real cse166 = cse22 * std::log(cse24);
	const amrex::Real cse167 = std::pow(state.T, 1.0 / 4.0);
	const amrex::Real cse168 = 1.75 * cse167 * state.xn[0] + state.xn[3] + 680.10000000000002 * state.xn[2] / cse167;
	const amrex::Real cse169 = l0_co_interp1d(state.T);
	const amrex::Real cse170 = 1.0 / cse169;
	const amrex::Real cse171 = ((ncol_H <= 0) ? (100000.0 * state.xn[9] / std::max(gradv, 5.7445531974402559e-18 * M_SQRT2 * cse2))
						  : (1756701454.4310031 * ncol_CO / std::sqrt(std::max(1.0000000000000001e-30, cse28 * ncol_H))));
	const amrex::Real cse172 = 1.0 / llte_co_interp2d(state.T, cse171);
	const amrex::Real cse173 = nhalf_co_interp2d(state.T, cse171);
	const amrex::Real cse174 = 0.042999999999999997 * cse161;
	const amrex::Real cse175 = 4.34e-10 * std::pow(cse160, 0.755 - 0.16 * cse161) * state.xn[0] +
				   1.08e-14 * std::pow(state.T, 0.92600000000000005) * state.xn[2] +
				   (2.2500000000000003e-12 * std::pow(cse160, 0.52500000000000002 * cse161 + 1.198) +
				    5.2499999999999998e-13 * std::pow(cse160, cse174 + 0.88900000000000001)) *
				       state.xn[3];
	const amrex::Real cse176 = 3.5700000000000001e-10 * std::pow(cse160, 0.41899999999999998 - 0.0030000000000000001 * cse161) * state.xn[0] +
				   5.1199999999999999e-10 * std::pow(state.T, -0.074999999999999997) * state.xn[2] +
				   (3.7250000000000001e-11 * std::pow(cse160, 0.025000000000000001 * cse161 + 0.26400000000000001) +
				    1.0274999999999999e-10 * std::pow(cse160, cse174 + 0.29599999999999999)) *
				       state.xn[3];
	const amrex::Real cse177 = (1.0 / 5.0) * std::exp(-326.60000000000002 * cse18);
	const amrex::Real cse178 = 3.1899999999999998e-10 * std::pow(cse160, 0.36899999999999999 - 0.0060000000000000001 * cse161) * state.xn[0] +
				   4.8599999999999998e-10 * std::pow(state.T, -0.025999999999999999) * state.xn[2] +
				   (4.7499999999999998e-11 * std::pow(cse160, 0.041000000000000002 * cse161 + 0.20300000000000001) +
				    1.6725000000000002e-10 * std::pow(cse160, 0.058000000000000003 * cse161 + 0.23699999999999999)) *
				       state.xn[3];
	const amrex::Real cse179 = cse176 + 8.9099999999999997e-5;
	const amrex::Real cse180 = cse177 * cse178;
	const amrex::Real cse181 = cse175 * cse176 * cse177 + (1.0 / 15.0) * cse175 * cse178 * std::exp(-425.40000000000003 * cse18) + cse179 * cse180;
	const amrex::Real cse182 = cse175 + 1.7499999999999998e-5;
	const amrex::Real cse183 = cse178 + 1.34e-10;
	const amrex::Real cse184 = (3.0 / 5.0) * cse176 * std::exp(-227.80000000000001 * cse18);
	const amrex::Real cse185 = cse180 * cse182 + cse182 * cse184 + cse183 * cse184;
	const amrex::Real cse186 =
	    1.0 / ((1.0 / 3.0) * cse175 * cse183 * std::exp(-98.800000000000011 * cse18) + cse179 * cse182 + cse179 * cse183 + cse181 + cse185);
	const amrex::Real cse187 = state.T < 100.0;
	const amrex::Real cse188 = state.T < 6000.0;
	const amrex::Real cse189 = state.T < 10;
	const amrex::Real cse190 = std::log(cse154);
	const amrex::Real cse191 = cse190 * cse22;
	const amrex::Real cse192 = std::pow(cse21, -8);
	const amrex::Real cse193 = std::pow(cse190, 8) * cse192;
	const amrex::Real cse194 = std::pow(cse21, -7);
	const amrex::Real cse195 = std::pow(cse190, 7) * cse194;
	const amrex::Real cse196 = std::pow(cse21, -6);
	const amrex::Real cse197 = std::pow(cse190, 6) * cse196;
	const amrex::Real cse198 = std::pow(cse21, -5);
	const amrex::Real cse199 = std::pow(cse190, 5) * cse198;
	const amrex::Real cse200 = std::pow(cse21, -4);
	const amrex::Real cse201 = std::pow(cse190, 4) * cse200;
	const amrex::Real cse202 = std::pow(cse21, -3);
	const amrex::Real cse203 = std::pow(cse190, 3);
	const amrex::Real cse204 = cse202 * cse203;
	const amrex::Real cse205 = std::pow(cse190, 2) * cse26;
	const amrex::Real cse206 = std::pow(10.0, 16.815729999999999 * cse191 + 70.609154000000004 * cse193 + 364.14445999999998 * cse195 +
						      801.81246999999996 * cse197 + 983.67575999999997 * cse199 + 734.71650999999997 * cse201 +
						      343.1918 * cse204 + 96.743155000000002 * cse205 - 21.928795999999998) *
				   state.xn[2];
	const amrex::Real cse207 = state.xn[5] + state.xn[6];
	const amrex::Real cse208 = std::pow(10.0, 1.5714710999999999 * cse191 + 0.32168730000000001 * cse199 - 0.51002221000000003 * cse201 -
						      0.23619984999999999 * cse204 + 0.015391166 * cse205 - 22.089523) *
				       state.xn[1] +
				   std::pow(10.0, 2.1892372 * cse191 + 0.19191374999999999 * cse199 - 0.16596184 * cse201 + 0.29036281000000003 * cse204 -
						      0.81520437999999995 * cse205 - 23.689236999999999) *
				       cse207 +
				   std::pow(10.0, 2.0943374000000001 * cse190 * cse22 - 0.033638326000000003 * cse199 - 0.14913216000000001 * cse201 +
						      0.43693353000000001 * cse202 * cse203 - 0.77151435999999995 * cse205 - 23.962112000000001) *
				       state.xn[3];
	const amrex::Real cse209 = std::pow(10, 37.383713 * cse191 + 3.8479610000000002 * cse199 + 20.159831000000001 * cse201 + 48.656103000000002 * cse204 +
						    58.145166000000003 * cse205 - 16.818342000000001) *
				       state.xn[0] +
				   cse208;
	const amrex::Real cse210 = std::pow(10.0, 1.6802758 * cse191 - 6.3701156000000001 * cse193 + 6.4380698000000001 * cse195 + 8.9167182999999994 * cse197 -
						      8.8077017000000009 * cse199 - 4.7274035999999997 * cse201 + 4.0406627000000004 * cse204 +
						      0.93310621999999999 * cse205 - 22.921188999999998) *
				   state.xn[2];
	const amrex::Real cse211 = state.T < 1000.0;
	const amrex::Real cse212 = std::pow(10.0, 3.5692468000000002 * cse190 * cse22 - 4.2519023000000002 * cse199 - 21.328264000000001 * cse201 -
						      27.850082 * cse204 - 11.33286 * cse205 - 24.311209000000002) *
				       state.xn[0] +
				   cse208;
	const amrex::Real cse213 = std::pow(10.0, 4.6450521 * cse191 + 1.5538288 * cse199 - 5.5108049000000001 * cse201 + 5.9369081000000001 * cse204 -
						      3.7209846 * cse205 - 24.311209000000002) *
				       state.xn[0] +
				   cse208;
	const amrex::Real cse214 = std::pow(10.0, -0.62120210234156448 * cse198 - 1.5370575959566981 * cse200 + 2.5133588394664677 * cse202 +
						      3.7525488682084647 * cse22 - 2.4768712409536775 * cse26 - 23.962112000000001) *
				       state.xn[3] +
				   std::pow(10.0, 3.5440891133599637 * cse198 - 1.7105157385968939 * cse200 + 1.6702447513374001 * cse202 +
						      3.9225864834863131 * cse22 - 2.6171337683481011 * cse26 - 23.689236999999999) *
				       cse207 +
				   std::pow(10.0, 5.9406293599919788 * cse198 - 5.2566362076906961 * cse200 - 1.3586848802337363 * cse202 +
						      2.8156982240432273 * cse22 + 0.049411830040524525 * cse26 - 22.089523) *
				       state.xn[1] +
				   std::pow(10.0, 28.69470131298657 * cse198 - 56.79810800956944 * cse200 + 34.150687525022562 * cse202 +
						      8.3228160852326614 * cse22 - 11.945856385319289 * cse26 - 24.311209000000002) *
				       state.xn[0];
	const amrex::Real cse215 = ((cse188) ? (cse18) : (0.00016666666666666666));
	const amrex::Real cse216 = 1.6e-18 * std::exp(-11700.0 * cse215) + 6.7e-19 * std::exp(-5860.0 * cse215) + 3.0e-24 * std::exp(-510.0 * cse215) +
				   4.985670872372847e-33 * ((cse188) ? (std::pow(state.T, 3.7599999999999998)) : (160638152785302.53)) *
				       std::exp(-2197000.0 * ((cse188) ? (std::pow(state.T, -3)) : (4.6296296296296294e-12))) /
				       (6.0142468035272636e-8 * ((cse188) ? (std::pow(state.T, 2.1000000000000001)) : (85924621.118090734)) + 1.0);
	const amrex::Real cse217 = std::log(0.001 * ((cse188) ? (state.T) : (6000.0)));
	const amrex::Real cse218 = cse198 * std::pow(cse217, 5);
	const amrex::Real cse219 = cse200 * std::pow(cse217, 4);
	const amrex::Real cse220 = std::pow(cse217, 3);
	const amrex::Real cse221 = std::pow(cse217, 2) * cse26;
	const amrex::Real cse222 = cse217 * cse22;
	const amrex::Real cse223 = cse202 * cse220;
	const amrex::Real cse224 = cse192 * std::pow(cse217, 8);
	const amrex::Real cse225 = cse194 * std::pow(cse217, 7);
	const amrex::Real cse226 = cse196 * std::pow(cse217, 6);
	const amrex::Real cse227 = 0.045999999999999999 * cse161;
	const amrex::Real cse228 = 1.7258000000000001e-6 * cse159;
	const amrex::Real cse229 =
	    8.9000000000000003e-11 * std::pow(cse160, cse227 + 0.22800000000000001) * state.xn[0] +
	    cse228 *
		std::exp(((cse211)
			      ? (cse85 * (cse85 * (cse85 * (0.0070527699999999999 * cse85 - 0.11133800000000001) + 0.69763799999999998) - 1.3074300000000001) -
				 7.6973500000000001)
			      : (cse85 * (cse85 * (cse85 * (0.093813800000000003 * cse85 - 3.0328300000000001) + 36.180300000000003) - 187.47399999999999) +
				 350.60899999999998))) +
	    (5.1749999999999996e-11 * std::pow(cse160, 0.037999999999999999 * cse161 + 0.16900000000000001) +
	     2.15e-11 * std::pow(cse160, 0.048000000000000001 * cse161 - 0.01)) *
		state.xn[3];
	const amrex::Real cse230 =
	    2.6400000000000002e-10 * std::pow(cse160, cse227 + 0.23100000000000001) * state.xn[0] +
	    cse228 *
		std::exp(((cse211)
			      ? (cse85 * (cse85 * (cse85 * (0.00235272 * cse85 - 0.041816600000000002) + 0.35826400000000003) - 0.57443) - 7.4386999999999999)
			      : (cse85 * (cse85 * (cse85 * (0.097857299999999994 * cse85 - 3.1926800000000002) + 38.504899999999999) - 202.19300000000001) +
				 386.18599999999998))) +
	    (1.11e-10 * std::pow(cse160, 0.031 * cse161 + 0.26300000000000001) +
	     4.3749999999999998e-11 * std::pow(cse160, 0.064000000000000001 * cse161 + 0.071999999999999995)) *
		state.xn[3];
	const amrex::Real cse231 =
	    2.8763333333333333e-6 * cse159 *
		std::exp(((cse211) ? (cse85 * (cse85 * (cse85 * (-0.00065632500000000003 * cse85 - 0.015089200000000001) + 0.36118400000000001) -
					       0.77378199999999997) -
				      9.2514099999999999)
				   : (cse85 * (cse85 * (cse85 * (0.10508000000000001 * cse85 - 3.4762) + 42.594999999999999) - 227.91300000000001) +
				      444.60000000000002))) +
	    1.26e-10 * std::pow(cse160, 0.057000000000000002 * cse161 + 0.115) * state.xn[0] +
	    (5.3249999999999997e-11 * std::pow(cse160, 0.049000000000000002 * cse161 - 0.0040000000000000001) +
	     1.675e-11 * std::pow(cse160, 0.10199999999999999 * cse161 - 0.085000000000000006)) *
		state.xn[3];
	const amrex::Real cse232 = 5 * std::exp(-62.460000000000001 * cse18);
	const amrex::Real cse233 = cse231 + 7.8800000000000004e-8;
	const amrex::Real cse234 = cse229 * cse232;
	const amrex::Real cse235 = (25.0 / 3.0) * cse229 * cse230 * std::exp(-101.30000000000001 * cse18) + cse230 * cse231 * cse232 + cse233 * cse234;
	const amrex::Real cse236 = cse230 + 2.65e-7;
	const amrex::Real cse237 = cse229 + 1.81e-14;
	const amrex::Real cse238 = 3 * cse231 * std::exp(-23.619999999999997 * cse18);
	const amrex::Real cse239 = cse234 * cse236 + cse236 * cse238 + cse237 * cse238;
	const amrex::Real cse240 =
	    1.0 / ((5.0 / 3.0) * cse230 * cse237 * std::exp(-38.840000000000003 * cse18) + cse233 * cse236 + cse233 * cse237 + cse235 + cse239);

	ydot(1) = cse1 + cse13 + cse15 + cse17 + cse20 + cse23 * cse38 + 2 * cse36 - cse37 + 2.9935562710997168e-6 * cse40 -
		  6.0000000000000005e-15 * cse41 * state.xn[0] + cse44 + 1.1399999999999999e-10 * cse48 * state.xn[3] + 4.2899999999999994e-13 * cse51 + cse56 +
		  cse62 + cse68 + 3.2000000000000001e-9 * cse7 + cse88 + cse9;
	ydot(2) = cse0 + cse61 + cse72 + cse75 - cse80 - cse87 + cse89;
	ydot(3) = -cse104 - cse110 - cse123 - cse126 - cse88 + cse90 - cse92 + cse94 - cse96 + cse97;
	ydot(4) = -cse126 - cse127 - cse128 + cse13 + cse130 - cse131 - cse132 * state.xn[3] - cse134 + 3.0000000000000002e-15 * cse24 * d2g * state.xn[0] -
		  cse36 + 4.5427507615910105e-7 * cse39 * state.xn[16] * state.xn[2] - cse61 - cse68;
	ydot(5) = cse134 + cse135;
	ydot(6) = cse137;
	ydot(7) = -cse137;
	ydot(8) = cse123 + cse138 + cse141 + 9.8999999999999995e-14 * cse51 + cse55;
	ydot(9) = cse112 - cse115 - cse117 + cse119 + cse122 - cse124 + cse136 - cse143;
	ydot(10) = -cse141 - cse144 - cse145;
	ydot(11) = cse143 + cse144;
	ydot(12) = cse139 + cse140 + cse145 + cse146 + cse8 + cse89;
	ydot(13) = -cse147;
	ydot(14) = cse147;
	ydot(15) = -cse138 + 2.3099999999999997e-13 * cse49 * cse50 * state.xn[3] * state.xn[8] - cse56;
	ydot(16) = -cse143 - cse146 - cse20 - cse59 + cse9;
	ydot(17) = -cse104 - cse130 - cse135;
	ydot(18) = -cse125 + cse59 - cse89;
	ydot(19) =
	    (1.7e-24 * chi * cse24 * cse81 * d2g * (2.25 * std::pow(state.T, 0.14699999999999999) + 5.2199999999999998) /
		 (0.049959999999999997 * std::pow(cse152, 0.43099999999999999) * (0.0043 * std::pow(cse152, 0.69199999999999995) + 1) + 1) +
	     5.6999999999999997e-11 * chi * cse46 * cse47 * (2.5634832e-11 * cse158 + 6.4087080000000003e-13) * state.xn[3] +
	     1.602177e-12 * crate * (26.399999999999999 * std::sqrt(cse100 / (cse100 + 0.070000000000000007)) + 6.5) * state.xn[0] +
	     3.2043540000000001e-12 * crate *
		 ((cse166 < 2) ? (10.0)
			       : ((cse166 < 4) ? ((3.0 / 2.0) * cse166 + 7)
					       : ((cse166 < 7) ? ((4.0 / 3.0) * cse166 + 23.0 / 3.0)
							       : ((cse166 < 10) ? ((1.0 / 3.0) * cse166 + 44.0 / 3.0) : (18.0))))) *
		 state.xn[3] -
	     7.1777529600000012e-12 * cse131 - 1.3659996292638161e-10 * cse149 * cse151 * cse74 / (1.3338116951315871e-8 * cse150 * cse151 + 1) -
	     5.7927337833400002e-20 * cse165 * state.xn[8] / (2 * cse165 + 1) -
	     cse168 * std::pow(1 - std::exp(-state.T), 1000) * state.xn[9] /
		 (cse168 * cse172 + cse170 * std::pow(cse168 / cse173, alpha_co_interp2d(state.T, cse171)) * (-cse169 * cse172 * cse173 + 1) + cse170) +
	     3.0000000000000002e-15 * cse24 * d2g * (6.7291434000000005e-12 * cse158 + 3.2043540000000002e-13) * state.xn[0] - 7.1777529600000012e-12 * cse36 -
	     1.8998126555754159e-29 * cse41 * std::pow(state.T, 0.45350000000000001 + 2.234 / cse153) *
		 std::exp(cse153 * (1.4419999999999999 - 0.050889999999999998 * cse153)) * state.xn[2] -
	     2.1789607199999999e-11 * cse75 - (2.3872025442750958e-19 * cse181 * cse186 + 2.8023005140019997e-18 * cse185 * cse186) * state.xn[11] -
	     ((((cse162 && cse187 && cse188)
		    ? (cse189 || cse206 + cse209 < 1.0e-100)
		    : (((cse187 && cse188)
			    ? (cse189 || cse209 + cse210 < 1.0e-100)
			    : (((cse162 && cse188 && cse211)
				    ? (cse189 || cse206 + cse212 < 1.0e-100)
				    : (((cse188 && cse211)
					    ? (cse189 || cse210 + cse212 < 1.0e-100)
					    : (((cse162 && cse188)
						    ? (cse189 || cse206 + cse213 < 1.0e-100)
						    : (((cse188) ? (cse189 || cse210 + cse213 < 1.0e-100)
								 : (((cse162) ? (cse189 ||
										 std::pow(10.0, 7500.646099484944 * cse192 + 21588.957206746247 * cse194 +
												    26530.843406310876 * cse196 + 18165.631967965237 * cse198 +
												    7572.4886742758581 * cse200 + 1974.131269262874 * cse202 +
												    30.129743459482278 * cse22 + 310.58441786958315 * cse26 -
												    21.928795999999998) *
											     state.xn[2] +
											 cse214 <
										     1.0e-100)
									      : (cse189 ||
										 std::pow(10.0, -676.68255490510751 * cse192 + 381.69251128040054 * cse194 +
												    295.04162851880631 * cse196 - 162.65264843551881 * cse198 -
												    48.723840736518795 * cse200 + 23.242975457496804 * cse202 +
												    3.0106500755647456 * cse22 + 2.9956460707652877 * cse26 -
												    22.921188999999998) *
											     state.xn[2] +
											 cse214 <
										     1.0e-100)))))))))))))))
		  ? (0.0)
		  : (cse216 /
		     (cse216 / (std::pow(10.0, 0.19191374999999999 * cse218 - 0.16596184 * cse219 - 0.81520437999999995 * cse221 + 2.1892372 * cse222 +
						   0.29036281000000003 * cse223 - 23.689236999999999) *
				    cse207 +
				std::pow(10.0, 0.32168730000000001 * cse218 - 0.51002221000000003 * cse219 + 0.015391166 * cse221 +
						   1.5714710999999999 * cse222 - 0.23619984999999999 * cse223 - 22.089523) *
				    state.xn[1] +
				std::pow(10.0, 0.43693353000000001 * cse202 * cse220 + 2.0943374000000001 * cse217 * cse22 - 0.033638326000000003 * cse218 -
						   0.14913216000000001 * cse219 - 0.77151435999999995 * cse221 - 23.962112000000001) *
				    state.xn[3] +
				((cse162) ? (std::pow(10.0, 983.67575999999997 * cse218 + 734.71650999999997 * cse219 + 96.743155000000002 * cse221 +
								16.815729999999999 * cse222 + 343.1918 * cse223 + 70.609154000000004 * cse224 +
								364.14445999999998 * cse225 + 801.81246999999996 * cse226 - 21.928795999999998))
					  : (std::pow(10.0, -8.8077017000000009 * cse218 - 4.7274035999999997 * cse219 + 0.93310621999999999 * cse221 +
								1.6802758 * cse222 + 4.0406627000000004 * cse223 - 6.3701156000000001 * cse224 +
								6.4380698000000001 * cse225 + 8.9167182999999994 * cse226 - 22.921188999999998))) *
				    state.xn[2] +
				((((cse188) ? (cse187) : (false)))
				     ? (std::pow(10, 3.8479610000000002 * cse218 + 20.159831000000001 * cse219 + 58.145166000000003 * cse221 +
							 37.383713 * cse222 + 48.656103000000002 * cse223 - 16.818342000000001))
				     : ((((cse188) ? (cse211) : (false)))
					    ? (std::pow(10.0, 3.5692468000000002 * cse217 * cse22 - 4.2519023000000002 * cse218 - 21.328264000000001 * cse219 -
								  11.33286 * cse221 - 27.850082 * cse223 - 24.311209000000002))
					    : (std::pow(10.0, 1.5538288 * cse218 - 5.5108049000000001 * cse219 - 3.7209846 * cse221 + 4.6450521 * cse222 +
								  5.9369081000000001 * cse223 - 24.311209000000002)))) *
				    state.xn[0]) +
		      1))) *
		 state.xn[3] -
	     ((state.T < 1000000.0) ? (1.4210469458259591e-21 * cse235 * cse240 + 2.5697412351439998e-22 * cse239 * cse240) : (0.0)) * state.xn[7] -
	     std::min(100.0 * cse41 * psigd_coll_interp2d(cse24, state.T),
		      3.2000000000000002e-32 * cse2 * std::pow(cse24, 2) * d2g *
			  (state.T - 2.1544346900318838 * std::pow(85973.158732959506 * cse82 + 4.1397668473788896, 1.0 / 6.0)))) /
		cse148 -
	    1.4999999999999249 * state.T *
		(1.380649e-16 * state.xn[0] + 1.380649e-16 * state.xn[10] + 1.380649e-16 * state.xn[11] + 1.380649e-16 * state.xn[12] +
		 1.380649e-16 * state.xn[13] + 1.380649e-16 * state.xn[14] + 1.380649e-16 * state.xn[15] + 1.380649e-16 * state.xn[16] +
		 1.380649e-16 * state.xn[17] + 1.380649e-16 * state.xn[1] + 1.380649e-16 * state.xn[2] + 1.380649e-16 * state.xn[3] +
		 1.380649e-16 * state.xn[4] + 1.380649e-16 * state.xn[5] + 1.380649e-16 * state.xn[6] + 1.380649e-16 * state.xn[7] +
		 1.380649e-16 * state.xn[8] + 1.380649e-16 * state.xn[9]) *
		(1.3012031544987096e-47 * chi * cse93 * state.xn[12] + 4.8578251101285161e-35 * crate * cse28 * state.xn[12] * state.xn[3] +
		 9.3874889891030969e-50 * cse10 * cse101 * cse102 * state.xn[11] * state.xn[16] - 5.1890642653333708e-41 * cse105 +
		 4.7173311503030638e-52 * cse106 * cse108 * state.xn[2] * state.xn[6] + 2.6284969169488674e-52 * cse109 * cse24 * d2g * state.xn[6] -
		 1.8161724928666798e-40 * cse111 + 4.7173311503030642e-41 * cse114 * state.xn[2] * state.xn[8] +
		 2.1501595383081364e-51 * cse116 * cse24 * d2g * state.xn[8] - 4.9060243963151868e-38 * cse118 - 6.3713565952657906e-49 * cse12 -
		 1.6510659026060724e-50 * cse121 - 4.9958509909947217e-48 * cse129 - 1.285696946211877e-50 * cse14 - 2.6889433275059828e-48 * cse142 -
		 1.3958995416014663e-49 * cse16 - 2.9208719207347922e-49 * cse19 + 5.7787306591212543e-52 * cse24 * cse86 * d2g * state.xn[1] -
		 2.9387358770557188e-39 * cse37 - 7.2737227656057311e-46 * cse40 + 2.9387358770557188e-39 * cse42 * cse43 * state.xn[11] * state.xn[1] +
		 1.0042319124598398e-52 * cse49 * cse50 * state.xn[3] * state.xn[8] - 4.0387529294731574e-48 * cse5 * cse7 - 2.5713938924237541e-50 * cse52 -
		 3.34281206015088e-49 * cse54 + 3.4713817547720679e-48 * cse57 * state.xn[15] * state.xn[6] - 1.0338099956904871e-50 * cse63 +
		 6.465218929522581e-49 * cse65 * cse66 * state.xn[3] * state.xn[4] - 1.7073058169602635e-38 * cse69 - 4.626500104884301e-48 * cse7 -
		 9.4346623006061284e-41 * cse71 * state.xn[3] - 4.7173311503030642e-41 * cse72 - 4.7173311503030642e-41 * cse75 +
		 2.7717497761966105e-44 * cse76 * state.xn[10] * state.xn[2] + 2.3026978988630935e-46 * cse78 * cse79 * state.xn[1] * state.xn[2] -
		 4.220235648504098e-49 * cse91 - 6.2631245169871218e-51 * cse95 + 4.7173311503030642e-41 * cse98 * state.xn[16] * state.xn[7]) /
		std::pow(cse148, 2);
	// $JAFF END
}

template <class MatrixType> AMREX_GPU_HOST_DEVICE AMREX_INLINE void actual_jac(const burn_t &state, MatrixType &jac)
{
	jac.zero();
	const amrex::Real ncol_H2 = network_rp::ncol_H2;
	const amrex::Real ncol_CO = network_rp::ncol_CO;
	const amrex::Real ncol_C = network_rp::ncol_C;
	const amrex::Real ncol_H = network_rp::ncol_H;
	const amrex::Real d2g = network_rp::dust2gas_ratio;
	const amrex::Real gradv = 3.0 * 3.0e-14;
	const amrex::Real vdisp = 3.0 * 1.0e5 / pow(2, 0.5);
	const amrex::Real av = network_rp::Av;

	// $JAFF REPEAT idx, expr, cse IN jacobian $[REPLACE radeden\[\s*(\d+)\s*\] state.rn[2*\1] REPLACE photden\[\s*(\d+)\s*\] state.rn[2*\1] REPLACE
	// rflux\[\s*(\d+)\s*\] state.rn[2*\1+1] REPLACE nden\[\s*(\d+)\s*\] state.xn[\1] REPLACE tgas state.T REPLACE c_hat state.c_hat THERMAL dedt DEDT_TYPE
	// specific]$
	const amrex::Real cse0 = std::pow(state.T, 0.26000000000000001);
	const amrex::Real cse1 = 2.8142984982406731e-11 * cse0;
	const amrex::Real cse2 = cse1 * state.xn[14];
	const amrex::Real cse3 = std::sqrt(state.T);
	const amrex::Real cse4 = 1.0 / state.T;
	const amrex::Real cse5 = 63590 * cse4 + 1;
	const amrex::Real cse6 = std::exp(-cse5);
	const amrex::Real cse7 = std::log(6.6699999999999996e-12 * cse3 * cse6);
	const amrex::Real cse8 = M_LN10;
	const amrex::Real cse9 = 1.0 / cse8;
	const amrex::Real cse10 = 2 * state.xn[3];
	const amrex::Real cse11 = cse10 + state.xn[0] + state.xn[1] + state.xn[10] + state.xn[14] + state.xn[15] + 3 * state.xn[16] + 2 * state.xn[4];
	const amrex::Real cse12 = 1.0 / cse11;
	const amrex::Real cse13 = std::log((1.0 / 10000.0) * state.T);
	const amrex::Real cse14 = std::pow(cse8, -2);
	const amrex::Real cse15 = std::pow(cse13, 2) * cse14;
	const amrex::Real cse16 = std::pow(10, 1.3 * cse13 * cse9 - 1.6200000000000001 * cse15 - 4.8449999999999998);
	const amrex::Real cse17 = cse12 * cse16 * state.xn[3];
	const amrex::Real cse18 = std::pow(10, 0.41599999999999998 * cse13 * cse9 + 0.32700000000000001 * cse15 - 3);
	const amrex::Real cse19 = cse12 * cse18 * state.xn[0];
	const amrex::Real cse20 = cse17 + cse19;
	const amrex::Real cse21 = cse20 < 1.0e-50;
	const amrex::Real cse22 = ((cse21) ? (1.0e-50) : (cse20));
	const amrex::Real cse23 = cse11 * cse22;
	const amrex::Real cse24 = cse23 + 1;
	const amrex::Real cse25 = cse9 / cse24;
	const amrex::Real cse26 = std::log(3.5199999999999998e-9 * std::exp(-43900 * cse4));
	const amrex::Real cse27 = cse23 * cse25;
	const amrex::Real cse28 = std::pow(10, cse25 * cse7 + cse26 * cse27);
	const amrex::Real cse29 = state.T > 700.0;
	const amrex::Real cse30 = ((cse29) ? (cse28) : (0.0));
	const amrex::Real cse31 = std::pow(state.xn[3], 2);
	const amrex::Real cse32 = std::log(1.3000000000000001e-9 * std::exp(-53300 * cse4));
	const amrex::Real cse33 = cse22 * cse25;
	const amrex::Real cse34 = cse32 * cse33;
	const amrex::Real cse35 = std::pow(cse11, 2);
	const amrex::Real cse36 = 1.0 / cse35;
	const amrex::Real cse37 = cse36 * state.xn[3];
	const amrex::Real cse38 = cse16 * cse37;
	const amrex::Real cse39 = cse36 * state.xn[0];
	const amrex::Real cse40 = cse18 * cse39;
	const amrex::Real cse41 = cse38 + cse40;
	const amrex::Real cse42 = cse11 * ((cse21) ? (0) : (cse12 * cse18 - cse41));
	const amrex::Real cse43 = cse25 * cse42;
	const amrex::Real cse44 = std::pow(state.T, 4.1881000000000004);
	const amrex::Real cse45 = 6.7610000000000002e-6 * state.T + 1;
	const amrex::Real cse46 = std::pow(cse45, 5.6881000000000004);
	const amrex::Real cse47 = 54657.400000000001 * cse4;
	const amrex::Real cse48 = std::exp(-cse47);
	const amrex::Real cse49 = cse48 / cse46;
	const amrex::Real cse50 = std::log(5.9959999999999997e-30 * cse44 * cse49);
	const amrex::Real cse51 = cse9 / std::pow(cse24, 2);
	const amrex::Real cse52 = cse51 * (-cse22 - cse42);
	const amrex::Real cse53 = cse23 * cse52;
	const amrex::Real cse54 = std::pow(10, cse25 * cse50 + cse27 * cse32);
	const amrex::Real cse55 = cse54 * cse8;
	const amrex::Real cse56 = ((cse29) ? (cse55 * (cse32 * cse43 + cse32 * cse53 + cse34 + cse50 * cse52)) : (0));
	const amrex::Real cse57 = 6.0000000000000005e-15 * d2g;
	const amrex::Real cse58 = cse57 * state.xn[0];
	const amrex::Real cse59 = cse26 * cse33;
	const amrex::Real cse60 = cse28 * cse8;
	const amrex::Real cse61 = ((cse29) ? (cse60 * (cse26 * cse43 + cse26 * cse53 + cse52 * cse7 + cse59)) : (0));
	const amrex::Real cse62 = 1.6737729999999998e-24 * state.xn[0] + 4.8185222061629005e-23 * state.xn[10] + 2.6567630000000005e-23 * state.xn[11] +
				  4.6637070000000001e-23 * state.xn[12] + 4.6636159061629004e-23 * state.xn[13] + 2.1618503000000003e-23 * state.xn[14] +
				  2.8241403000000005e-23 * state.xn[15] + 5.0204080616289998e-24 * state.xn[16] + 2.6566719061629008e-23 * state.xn[17] +
				  1.6728620616289998e-24 * state.xn[1] + 9.1093837099999994e-28 * state.xn[2] + 3.3475459999999996e-24 * state.xn[3] +
				  3.3466350616289996e-24 * state.xn[4] + 6.6464729999999996e-24 * state.xn[5] + 6.6455620616289996e-24 * state.xn[6] +
				  1.9944730000000003e-23 * state.xn[7] + 1.9943819061629003e-23 * state.xn[8] + 4.6512360000000008e-23 * state.xn[9];
	const amrex::Real cse63 = 2.0709734999998963e-16 * state.T;
	const amrex::Real cse64 = cse63 / cse62;
	const amrex::Real cse65 = 1.380649e-16 * state.xn[0] + 1.380649e-16 * state.xn[10] + 1.380649e-16 * state.xn[11] + 1.380649e-16 * state.xn[12] +
				  1.380649e-16 * state.xn[13] + 1.380649e-16 * state.xn[14] + 1.380649e-16 * state.xn[15] + 1.380649e-16 * state.xn[16] +
				  1.380649e-16 * state.xn[17] + 1.380649e-16 * state.xn[1] + 1.380649e-16 * state.xn[2] + 1.380649e-16 * state.xn[3] +
				  1.380649e-16 * state.xn[4] + 1.380649e-16 * state.xn[5] + 1.380649e-16 * state.xn[6] + 1.380649e-16 * state.xn[7] +
				  1.380649e-16 * state.xn[8] + 1.380649e-16 * state.xn[9];
	const amrex::Real cse66 = cse65 * state.T / std::pow(cse62, 2);
	const amrex::Real cse67 = cse64 - 2.5106594999998743e-24 * cse66;
	const amrex::Real cse68 = std::pow(state.T, -2);
	const amrex::Real cse69 = std::exp(-22.5 * cse4);
	const amrex::Real cse70 = 2.8349999999999999e-12 * cse68 * cse69 * state.xn[3] * state.xn[6];
	const amrex::Real cse71 = -cse70;
	const amrex::Real cse72 = std::pow(state.T, -0.73999999999999999) * state.xn[0] * state.xn[14];
	const amrex::Real cse73 = 7.3171760954257497e-12 * cse72;
	const amrex::Real cse74 = std::sqrt(cse4);
	const amrex::Real cse75 = std::sqrt(3) * cse4 * cse74 * state.xn[15];
	const amrex::Real cse76 = cse75 * state.xn[6];
	const amrex::Real cse77 = 1.7697487500000004e-8 * cse76;
	const amrex::Real cse78 = cse27 * cse68;
	const amrex::Real cse79 = cse14 * cse4;
	const amrex::Real cse80 = cse13 * cse79;
	const amrex::Real cse81 = cse4 * cse9;
	const amrex::Real cse82 = ((cse21) ? (0)
					   : (cse17 * cse8 * (1.3 * cse4 * cse9 - 3.2400000000000002 * cse80) +
					      cse19 * cse8 * (0.65400000000000003 * cse80 + 0.41599999999999998 * cse81)));
	const amrex::Real cse83 = cse11 * cse82;
	const amrex::Real cse84 = cse25 * cse32;
	const amrex::Real cse85 = cse51 * cse83;
	const amrex::Real cse86 = cse22 * cse35 * cse51 * cse82;
	const amrex::Real cse87 = ((cse29) ? (cse55 * (1.6677785190126752e+29 * cse25 * cse46 *
							   (-2.3058963562359999e-34 * cse44 * std::pow(cse45, -6.6881000000000004) * cse48 +
							    3.2772577039999999e-25 * cse49 * std::pow(state.T, 2.1881000000000004) +
							    2.51118476e-29 * cse49 * std::pow(state.T, 3.1881000000000004)) *
							   std::exp(cse47) / cse44 -
						       cse32 * cse86 - cse50 * cse85 + 53300.0 * cse78 + cse83 * cse84))
					   : (0));
	const amrex::Real cse88 =
	    state.xn[0] * state.xn[17] * (-3.435e-10 * std::pow(state.T, -1.458) + 2.0209499999999999e-11 * std::pow(state.T, -0.59499999999999997));
	const amrex::Real cse89 = 1.0 / cse3;
	const amrex::Real cse90 = std::pow(state.T, -3.0 / 2.0);
	const amrex::Real cse91 = cse25 * cse26;
	const amrex::Real cse92 = ((cse29) ? (cse60 * (149925037481.25937 * cse25 * cse89 *
							   (3.3349999999999998e-12 * cse6 * cse89 + 4.2414529999999999e-7 * cse6 * cse90) * std::exp(cse5) -
						       cse26 * cse86 - cse7 * cse85 + 43900.0 * cse78 + cse83 * cse91))
					   : (0));
	const amrex::Real cse93 = std::pow(state.T, -1.52) * state.xn[16] * state.xn[2];
	const amrex::Real cse94 = std::exp(-227 * cse4);
	const amrex::Real cse95 =
	    cse94 * state.xn[1] * state.xn[11] *
	    (2.6759999999999999e-12 * std::pow(state.T, -0.99331000000000003) + 5.6870000000000003e-12 * std::pow(state.T, -0.48299999999999998));
	const amrex::Real cse96 = std::pow(state.T, -2.2999999999999998);
	const amrex::Real cse97 = std::exp(-23 * cse4);
	const amrex::Real cse98 = cse97 * state.xn[3] * state.xn[8];
	const amrex::Real cse99 = cse96 * cse98;
	const amrex::Real cse100 = std::pow(state.T, 0.042000000000000003);
	const amrex::Real cse101 = std::exp(-2.145922746781116e-5 * state.T);
	const amrex::Real cse102 = cse100 * cse101;
	const amrex::Real cse103 = state.xn[3] * state.xn[4];
	const amrex::Real cse104 = cse102 * cse103;
	const amrex::Real cse105 = 3.7768240343347639e-14 * cse104;
	const amrex::Real cse106 = cse101 * cse103 * std::pow(state.T, -0.95799999999999996);
	const amrex::Real cse107 = 7.3920000000000002e-11 * cse106;
	const amrex::Real cse108 = std::pow(state.T, -3.2999999999999998);
	const amrex::Real cse109 =
	    4.0000000000000001e-10 * std::pow(state.T, 0.0066899999999999998) + 1.1000000000000001e-11 * std::pow(state.T, 0.51700000000000002);
	const amrex::Real cse110 = cse109 * cse94;
	const amrex::Real cse111 = cse110 * state.xn[11];
	const amrex::Real cse112 = cse111 * cse68 * state.xn[1];
	const amrex::Real cse113 = 227 * cse112;
	const amrex::Real cse114 = 1.0 / state.xn[2];
	const amrex::Real cse115 = cse114 * cse89;
	const amrex::Real cse116 = cse114 * cse3;
	const amrex::Real cse117 = 0.00011320754716981131 * cse116;
	const amrex::Real cse118 = cse117 * state.xn[3];
	const amrex::Real cse119 = (((cse118 - 10000000000.0 > 0) ? (0) : ((cse118 - 10000000000.0 == 0) ? (1.0 / 2.0) : (1))));
	const amrex::Real cse120 = cse119 * cse31;
	const amrex::Real cse121 = cse115 * cse120;
	const amrex::Real cse122 = std::min(10000000000.0, cse118);
	const amrex::Real cse123 = cse122 + 1;
	const amrex::Real cse124 = std::pow(cse123, -2);
	const amrex::Real cse125 = cse124 * state.xn[17];
	const amrex::Real cse126 = 1.811320754716981e-13 * cse125;
	const amrex::Real cse127 = std::pow(state.T, -1.6400000000000001) * state.xn[10] * state.xn[2];
	const amrex::Real cse128 = 6.7743911789075847e-6 * cse127;
	const amrex::Real cse129 = std::exp(0.11 * cse4);
	const amrex::Real cse130 = cse129 * state.xn[15] * state.xn[7];
	const amrex::Real cse131 = cse130 * std::pow(state.T, -1.339);
	const amrex::Real cse132 = cse130 * std::pow(state.T, -2.339);
	const amrex::Real cse133 = 2.6955143233113923e-10 * cse131 + 8.7465066538127767e-11 * cse132;
	const amrex::Real cse134 = cse128 + cse133;
	const amrex::Real cse135 = std::log(8.6173000000000003e-5 * state.T);
	const amrex::Real cse136 = std::pow(cse135, 2);
	const amrex::Real cse137 = std::pow(cse135, 3);
	const amrex::Real cse138 = std::pow(cse135, 4);
	const amrex::Real cse139 = std::pow(cse135, 5);
	const amrex::Real cse140 = std::pow(cse135, 6);
	const amrex::Real cse141 = std::pow(cse135, 7);
	const amrex::Real cse142 = std::exp(-2.0391498499999999e-6 * std::pow(cse135, 8) - 5.7393287500000003 * cse136 + 1.56315498 * cse137 -
					    0.28770560000000001 * cse138 + 0.034825597700000002 * cse139 - 0.00263197617 * cse140 + 0.000111954395 * cse141);
	const amrex::Real cse143 = 5.9078447946382801e-70 * cse142 * std::pow(state.T, 13.536555999999999);
	const amrex::Real cse144 = state.xn[0] * state.xn[2];
	const amrex::Real cse145 =
	    cse144 * ((cse29) ? (7.9971871901929577e-69 * cse142 * std::pow(state.T, 12.536555999999999) +
				 cse143 * (-11.478657500000001 * cse135 * cse4 + 4.6894649399999997 * cse136 * cse4 - 1.1508224 * cse137 * cse4 +
					   0.1741279885 * cse138 * cse4 - 0.015791857020000001 * cse139 * cse4 + 0.00078368076500000001 * cse140 * cse4 -
					   1.6313198799999999e-5 * cse141 * cse4))
			      : (0));
	const amrex::Real cse146 = std::pow(cse4, 1.5);
	const amrex::Real cse147 = 114.81344434488582 * std::pow(cse4, 0.40699999999999997) + 1;
	const amrex::Real cse148 = std::pow(cse147, -2.242);
	const amrex::Real cse149 = cse146 * cse148;
	const amrex::Real cse150 = cse4 * state.xn[1] * state.xn[2];
	const amrex::Real cse151 = cse149 * cse150;
	const amrex::Real cse152 = 7.3220359950196314e-6 * cse151;
	const amrex::Real cse153 = std::pow(cse147, -3.242) * cse150 * std::pow(cse4, 1.907);
	const amrex::Real cse154 = 0.0005114031087522854 * cse153;
	const amrex::Real cse155 = std::exp(-1.8700000000000001 * av);
	const amrex::Real cse156 = chi * cse155;
	const amrex::Real cse157 = cse116 * cse156;
	const amrex::Real cse158 = std::pow(cse157, 1.3779999999999999);
	const amrex::Real cse159 = 1.7 * cse157;
	const amrex::Real cse160 = std::log(state.T);
	const amrex::Real cse161 = 1.102e-5 * cse160 + 0.4723;
	const amrex::Real cse162 = std::pow(cse159, -cse161);
	const amrex::Real cse163 = cse162 * std::pow(state.T, 0.015859999999999999);
	const amrex::Real cse164 = 508.69999999999999 * cse163;
	const amrex::Real cse165 = cse164 + 1;
	const amrex::Real cse166 = cse158 * cse165;
	const amrex::Real cse167 = -cse161;
	const amrex::Real cse168 = 0.5 * cse4;
	const amrex::Real cse169 = cse4 * std::log(cse159);
	const amrex::Real cse170 = 1.6774414051711492e-5 * cse158;
	const amrex::Real cse171 = cse165 * cse170 + 1;
	const amrex::Real cse172 = state.xn[1] / std::pow(cse171, 2);
	const amrex::Real cse173 =
	    cse172 * (-1.1557571281629217e-5 * cse166 * cse4 -
		      cse170 * (8.0679819999999989 * cse162 * std::pow(state.T, -0.98414000000000001) + cse164 * (cse167 * cse168 - 1.102e-5 * cse169)));
	const amrex::Real cse174 = cse11 * d2g;
	const amrex::Real cse175 = 1.2250000000000001e-11 * cse174;
	const amrex::Real cse176 = cse173 * cse175;
	const amrex::Real cse177 = 1.0 / cse123;
	const amrex::Real cse178 = cse177 * state.xn[17];
	const amrex::Real cse179 = 9.0566037735849048e-14 * cse121;
	const amrex::Real cse180 = cse178 * cse179;
	const amrex::Real cse181 = cse125 * cse179;
	const amrex::Real cse182 = cse122 * cse181;
	const amrex::Real cse183 = -cse180 + cse182;
	const amrex::Real cse184 = state.xn[11] * state.xn[16];
	const amrex::Real cse185 = cse184 * std::pow(state.T, -1.1899999999999999);
	const amrex::Real cse186 = cse177 * cse185;
	const amrex::Real cse187 = cse119 * state.xn[3];
	const amrex::Real cse188 = cse114 * cse124 * cse184;
	const amrex::Real cse189 = cse187 * cse188 * std::pow(state.T, -0.68999999999999995);
	const amrex::Real cse190 = 3.7810000000000002e-10 * cse186 + 1.1264150943396225e-13 * cse189;
	const amrex::Real cse191 = cse145 + cse152 - cse154 - cse176 + cse183 + cse190;
	const amrex::Real cse192 = 0.66666666666670005 * cse62 / cse65;
	const amrex::Real cse193 = cse192 * (-cse105 + cse107 + 9.8669999999999991e-12 * cse108 * cse97 * state.xn[3] * state.xn[8] + cse113 - cse121 * cse126 -
					     cse134 - cse191 + 2 * cse31 * cse87 - cse71 - cse73 - cse77 - cse88 + 2 * cse92 * state.xn[0] * state.xn[3] -
					     1.5566492609718528e-6 * cse93 + cse95 - 5.5769999999999994e-13 * cse99);
	const amrex::Real cse194 = 7.5e-10 * std::pow(state.T, -0.45800000000000002) + 4.9899999999999997e-11 * std::pow(state.T, 0.40500000000000003);
	const amrex::Real cse195 = cse194 * state.xn[17];
	const amrex::Real cse196 = 6.3999999999999996e-10 * state.xn[4];
	const amrex::Real cse197 = crate * (1.5 * cse12 * state.xn[0] + 2.2999999999999998 * cse12 * state.xn[3]);
	const amrex::Real cse198 = ((cse29) ? (cse143) : (0.0));
	const amrex::Real cse199 = cse198 * state.xn[2];
	const amrex::Real cse200 = 2.2999999999999998 * cse37 + 1.5 * cse39;
	const amrex::Real cse201 = 1.5 * cse12 - cse200;
	const amrex::Real cse202 = crate * state.xn[0];
	const amrex::Real cse203 = cse201 * cse202;
	const amrex::Real cse204 = 1.0 / cse171;
	const amrex::Real cse205 = 1.2250000000000001e-11 * cse204;
	const amrex::Real cse206 = d2g * state.xn[1];
	const amrex::Real cse207 = cse205 * cse206;
	const amrex::Real cse208 = cse195 + cse196 + cse197 + cse199 + cse203 - cse207;
	const amrex::Real cse209 = cse64 - 2.5092930924433742e-24 * cse66;
	const amrex::Real cse210 = 4.8813573300130879e-6 * cse149;
	const amrex::Real cse211 = -cse200;
	const amrex::Real cse212 = cse202 * cse211;
	const amrex::Real cse213 = cse207 - cse212;
	const amrex::Real cse214 = cse174 * cse205 + cse210 * state.xn[2] + cse213;
	const amrex::Real cse215 = cse111 + cse214;
	const amrex::Real cse216 = cse11 * ((cse21) ? (0) : (-cse41));
	const amrex::Real cse217 = cse51 * (-cse216 - cse22);
	const amrex::Real cse218 = cse217 * cse23;
	const amrex::Real cse219 = cse31 * ((cse29) ? (cse55 * (cse216 * cse84 + cse217 * cse50 + cse218 * cse32 + cse34)) : (0));
	const amrex::Real cse220 = -cse58;
	const amrex::Real cse221 = ((cse29) ? (cse60 * (cse216 * cse91 + cse217 * cse7 + cse218 * cse26 + cse59)) : (0));
	const amrex::Real cse222 = cse10 * state.xn[0];
	const amrex::Real cse223 = 2 * cse219 + cse220 + cse221 * cse222;
	const amrex::Real cse224 = std::pow(state.T, 0.31);
	const amrex::Real cse225 = 2.2528301886792451e-13 * cse224;
	const amrex::Real cse226 = std::pow(state.xn[2], 2);
	const amrex::Real cse227 = 1.0 / cse226;
	const amrex::Real cse228 = cse124 * cse184 * cse187 * cse227;
	const amrex::Real cse229 = cse225 * cse228;
	const amrex::Real cse230 = std::pow(state.T, -0.52000000000000002);
	const amrex::Real cse231 = 2.9935562710997168e-6 * cse230;
	const amrex::Real cse232 = cse64 - 1.3664075564999315e-27 * cse66;
	const amrex::Real cse233 = cse227 * cse3;
	const amrex::Real cse234 = cse120 * cse233;
	const amrex::Real cse235 = 3.6226415094339619e-13 * cse125;
	const amrex::Real cse236 = 1.811320754716981e-13 * cse178;
	const amrex::Real cse237 = cse234 * cse236;
	const amrex::Real cse238 = cse126 * cse234;
	const amrex::Real cse239 = cse122 * cse238;
	const amrex::Real cse240 = -cse237 + cse239;
	const amrex::Real cse241 = std::pow(state.T, -0.64000000000000001);
	const amrex::Real cse242 = 1.05849862170431e-5 * cse241;
	const amrex::Real cse243 = cse242 * state.xn[10];
	const amrex::Real cse244 = cse198 * state.xn[0];
	const amrex::Real cse245 = cse172 * (0.0085331444281056355 * cse114 * cse158 * cse163 * cse167 + 2.3115142563258434e-5 * cse114 * cse166);
	const amrex::Real cse246 = cse175 * cse245 + cse210 * state.xn[1] - cse244;
	const amrex::Real cse247 = cse243 + cse246;
	const amrex::Real cse248 = 1.2599999999999999e-13 * cse69;
	const amrex::Real cse249 = cse248 * state.xn[6];
	const amrex::Real cse250 = 1.7599999999999999e-9 * cse102;
	const amrex::Real cse251 = cse250 * state.xn[4];
	const amrex::Real cse252 = cse204 * cse206;
	const amrex::Real cse253 = 2.4500000000000003e-11 * cse252;
	const amrex::Real cse254 = cse30 * state.xn[0];
	const amrex::Real cse255 = 2 * cse22;
	const amrex::Real cse256 = cse255 * cse84;
	const amrex::Real cse257 = cse10 * cse36;
	const amrex::Real cse258 = 2 * state.xn[0];
	const amrex::Real cse259 = cse258 * cse36;
	const amrex::Real cse260 = cse16 * cse257 + cse18 * cse259;
	const amrex::Real cse261 = cse11 * ((cse21) ? (0) : (cse12 * cse16 - cse260));
	const amrex::Real cse262 = cse51 * (-cse255 - cse261);
	const amrex::Real cse263 = cse23 * cse262;
	const amrex::Real cse264 = cse31 * ((cse29) ? (cse55 * (cse256 + cse261 * cse84 + cse262 * cse50 + cse263 * cse32)) : (0));
	const amrex::Real cse265 = ((cse29) ? (cse54) : (0.0));
	const amrex::Real cse266 = 4 * state.xn[3];
	const amrex::Real cse267 = d2g * state.xn[0];
	const amrex::Real cse268 = 1.2000000000000001e-14 * cse267;
	const amrex::Real cse269 = cse255 * cse91;
	const amrex::Real cse270 = ((cse29) ? (cse60 * (cse26 * cse263 + cse261 * cse91 + cse262 * cse7 + cse269)) : (0));
	const amrex::Real cse271 = std::sqrt(2.0000000000000002e-15 * ncol_H2 + 1);
	const amrex::Real cse272 = 0.96499999999999997 / std::pow(1.4142135623730951e-10 * ncol_H2 / vdisp + 1, 2) +
				   0.035000000000000003 * std::exp(-0.00084999999999999995 * cse271) / cse271;
	const amrex::Real cse273 = std::exp(-4.1799999999999997 * av);
	const amrex::Real cse274 = chi * cse272 * cse273;
	const amrex::Real cse275 = std::pow(state.T, -1.3);
	const amrex::Real cse276 = cse275 * cse97;
	const amrex::Real cse277 = 4.2899999999999994e-13 * cse276;
	const amrex::Real cse278 = cse64 - 5.0213189999997486e-24 * cse66;
	const amrex::Real cse279 = cse116 * cse187;
	const amrex::Real cse280 = 1.6000000000000001e-9 * cse177;
	const amrex::Real cse281 = cse280 * state.xn[17];
	const amrex::Real cse282 = cse126 * cse279;
	const amrex::Real cse283 = cse122 * cse281 - cse122 * cse282 + cse236 * cse279;
	const amrex::Real cse284 = 4.5999999999999996 * cse37 + 3.0 * cse39;
	const amrex::Real cse285 = 2.2999999999999998 * cse12 - cse284;
	const amrex::Real cse286 = cse202 * cse285;
	const amrex::Real cse287 = cse119 * cse188;
	const amrex::Real cse288 = cse225 * cse287;
	const amrex::Real cse289 = -cse286 - cse288;
	const amrex::Real cse290 = cse11 * ((cse21) ? (0) : (-cse260));
	const amrex::Real cse291 = cse51 * (-cse255 - cse290);
	const amrex::Real cse292 = cse23 * cse291;
	const amrex::Real cse293 = ((cse29) ? (cse55 * (cse256 + cse290 * cse84 + cse291 * cse50 + cse292 * cse32)) : (0));
	const amrex::Real cse294 = ((cse29) ? (cse60 * (cse26 * cse292 + cse269 + cse290 * cse91 + cse291 * cse7)) : (0));
	const amrex::Real cse295 = cse250 * state.xn[3];
	const amrex::Real cse296 = cse64 - 5.0199525924432485e-24 * cse66;
	const amrex::Real cse297 = -cse253;
	const amrex::Real cse298 = 6.3999999999999996e-10 * state.xn[0];
	const amrex::Real cse299 = -cse284;
	const amrex::Real cse300 = cse202 * cse299;
	const amrex::Real cse301 = cse297 + cse298 + cse300;
	const amrex::Real cse302 = cse64 - 9.9697094999995006e-24 * cse66;
	const amrex::Real cse303 = cse64 - 9.9683430924430005e-24 * cse66;
	const amrex::Real cse304 = cse248 * state.xn[3];
	const amrex::Real cse305 = 26.218500000000002 * std::sqrt(3) * std::sqrt(cse4) + 0.62;
	const amrex::Real cse306 = 1.3500000000000001e-9 * cse305;
	const amrex::Real cse307 = cse306 * state.xn[15];
	const amrex::Real cse308 = cse304 + cse307;
	const amrex::Real cse309 = cse129 * std::pow(state.T, -0.33900000000000002);
	const amrex::Real cse310 = 7.9513696852843427e-10 * cse309;
	const amrex::Real cse311 = cse310 * state.xn[15];
	const amrex::Real cse312 = cse64 - 2.9917094999998509e-23 * cse66;
	const amrex::Real cse313 = cse64 - 2.991572859244201e-23 * cse66;
	const amrex::Real cse314 = 6.5199999999999996 * crate;
	const amrex::Real cse315 = cse64 - 6.976853999999652e-23 * cse66;
	const amrex::Real cse316 = cse64 - 7.2277833092439893e-23 * cse66;
	const amrex::Real cse317 = cse242 * state.xn[2];
	const amrex::Real cse318 = cse213 + cse317;
	const amrex::Real cse319 = std::pow(state.T, -0.19);
	const amrex::Real cse320 = cse177 * cse319;
	const amrex::Real cse321 = 1.99e-9 * cse320;
	const amrex::Real cse322 = cse321 * state.xn[16];
	const amrex::Real cse323 = 3.5000000000000002e-11 * state.xn[15];
	const amrex::Real cse324 = cse64 - 3.9851444999998012e-23 * cse66;
	const amrex::Real cse325 = cse110 * state.xn[1];
	const amrex::Real cse326 = 7.0000000000000004e-11 * state.xn[14];
	const amrex::Real cse327 = cse325 + cse326;
	const amrex::Real cse328 = cse64 - 6.9955604999996505e-23 * cse66;
	const amrex::Real cse329 = cse64 - 6.9954238592440006e-23 * cse66;
	const amrex::Real cse330 = 7.0000000000000004e-11 * state.xn[11];
	const amrex::Real cse331 = chi * std::exp(-2.1200000000000001 * av);
	const amrex::Real cse332 = 9.0999999999999996e-10 * cse331;
	const amrex::Real cse333 = cse1 * state.xn[0];
	const amrex::Real cse334 = -cse333;
	const amrex::Real cse335 = cse64 - 3.242775449999838e-23 * cse66;
	const amrex::Real cse336 = cse213 + cse223;
	const amrex::Real cse337 = cse64 - 4.2362104499997889e-23 * cse66;
	const amrex::Real cse338 = cse306 * state.xn[6];
	const amrex::Real cse339 = cse310 * state.xn[7];
	const amrex::Real cse340 = chi * std::exp(-2.6600000000000001 * av);
	const amrex::Real cse341 = 3.7999999999999998e-10 * cse340 + 3.5000000000000002e-11 * state.xn[11];
	const amrex::Real cse342 = cse338 + cse339 + cse341;
	const amrex::Real cse343 = 3 * cse22;
	const amrex::Real cse344 = cse11 * ((cse21) ? (0) : (-3 * cse38 - 3 * cse40));
	const amrex::Real cse345 = cse51 * (-cse343 - cse344);
	const amrex::Real cse346 = cse23 * cse345;
	const amrex::Real cse347 = cse31 * ((cse29) ? (cse55 * (cse32 * cse346 + cse343 * cse84 + cse344 * cse84 + cse345 * cse50)) : (0));
	const amrex::Real cse348 = ((cse29) ? (cse60 * (cse26 * cse346 + cse343 * cse91 + cse344 * cse91 + cse345 * cse7)) : (0));
	const amrex::Real cse349 = cse64 - 7.5306120924431232e-24 * cse66;
	const amrex::Real cse350 = cse321 * state.xn[11];
	const amrex::Real cse351 = -6.8999999999999995 * cse37 - 4.5 * cse39;
	const amrex::Real cse352 = cse202 * cse351;
	const amrex::Real cse353 = 3.6750000000000002e-11 * cse252 - cse352;
	const amrex::Real cse354 = cse350 + cse353;
	const amrex::Real cse355 = cse194 * state.xn[0];
	const amrex::Real cse356 = cse280 * state.xn[3];
	const amrex::Real cse357 = cse122 * cse356;
	const amrex::Real cse358 = cse64 - 3.9850078592441519e-23 * cse66;
	const amrex::Real cse359 = cse113 - cse88 + cse95;
	const amrex::Real cse360 = cse192 * (cse145 + cse152 - cse154 - cse176 - cse359 - cse71);
	const amrex::Real cse361 = crate * cse10;
	const amrex::Real cse362 = cse201 * cse361;
	const amrex::Real cse363 = -cse362;
	const amrex::Real cse364 = std::pow(cse4, 0.00231);
	const amrex::Real cse365 = std::exp(-8110.0 * cse4);
	const amrex::Real cse366 = std::exp(-26.600000000000001 * cse4);
	const amrex::Real cse367 = std::exp(-7.6200000000000001 * cse4);
	const amrex::Real cse368 = std::exp(-1.3799999999999999 * cse4);
	const amrex::Real cse369 = 0.00015100000000000001 * cse365 + 1.31e-7 * cse366 + 3.4e-8 * cse367 + 6.9699999999999997e-9 * cse368;
	const amrex::Real cse370 = std::pow(state.T, -1.5);
	const amrex::Real cse371 = state.xn[16] * state.xn[7] *
				   (-2.4342628390636543e-12 * cse364 * cse4 - 1.5 * cse369 * std::pow(state.T, -2.5) +
				    cse370 * (1.2246100000000002 * cse365 * cse68 + 3.4846000000000001e-6 * cse366 * cse68 +
					      2.5908000000000001e-7 * cse367 * cse68 + 9.6185999999999982e-9 * cse368 * cse68));
	const amrex::Real cse372 = -cse371;
	const amrex::Real cse373 = 6.7636511339243933e-7 * cse93;
	const amrex::Real cse374 = cse12 * state.xn[2];
	const amrex::Real cse375 = cse374 < 1.0e-50;
	const amrex::Real cse376 = ((cse375) ? (10000000000.0) : (cse118));
	const amrex::Real cse377 = cse376 + 1;
	const amrex::Real cse378 = 1.0 / cse377;
	const amrex::Real cse379 = cse376 * cse378;
	const amrex::Real cse380 = cse185 * cse379;
	const amrex::Real cse381 = 3.7810000000000002e-10 * cse380;
	const amrex::Real cse382 = ((cse375) ? (0) : (5.6603773584905654e-5 * cse115 * state.xn[3]));
	const amrex::Real cse383 = cse184 * cse319;
	const amrex::Real cse384 = cse378 * cse383;
	const amrex::Real cse385 = 1.99e-9 * cse384;
	const amrex::Real cse386 = cse382 * cse385;
	const amrex::Real cse387 = -cse386;
	const amrex::Real cse388 = cse376 * cse383 / std::pow(cse377, 2);
	const amrex::Real cse389 = 1.99e-9 * cse388;
	const amrex::Real cse390 = cse382 * cse389;
	const amrex::Real cse391 = std::exp(-15230.0 * cse4);
	const amrex::Real cse392 = std::exp(-73.799999999999997 * cse4);
	const amrex::Real cse393 = std::exp(-12.17 * cse4);
	const amrex::Real cse394 = 1.6339999999999999e-6 * cse391 + 9.7930000000000004e-9 * cse392 + 6.3460000000000002e-9 * cse393;
	const amrex::Real cse395 = 0.00071740367932898903 * cse3 + 1;
	const amrex::Real cse396 = std::exp(-49550.0 * cse4);
	const amrex::Real cse397 = 0.15970000000000001 * cse396;
	const amrex::Real cse398 = cse397 + 1.7848999999999999;
	const amrex::Real cse399 = 12.244387999457523 * cse3 + 1;
	const amrex::Real cse400 = 0.21509999999999996 - cse397;
	const amrex::Real cse401 = std::pow(cse395, -cse398) * std::pow(cse399, -cse400);
	const amrex::Real cse402 = 7913.1350000000002 * cse396 * cse68;
	const amrex::Real cse403 = 2.4460185352936309e-10 * cse401 * cse89;
	const amrex::Real cse404 =
	    state.xn[2] * state.xn[8] *
	    (-3.0 / 2.0 * cse394 / std::pow(state.T, 5.0 / 2.0) - 1.2230092676468155e-10 * cse401 * cse90 +
	     cse403 * (-cse402 * std::log(cse395) - 0.00035870183966449452 * cse398 * cse89 / cse395) +
	     cse403 * (cse402 * std::log(cse399) - 6.1221939997287613 * cse400 * cse89 / cse399) +
	     cse90 * (0.024885819999999999 * cse391 * cse68 + 7.2272340000000005e-7 * cse392 * cse68 + 7.7230820000000007e-8 * cse393 * cse68));
	const amrex::Real cse405 = std::pow(state.T, -1.6200000000000001) * state.xn[13] * state.xn[2];
	const amrex::Real cse406 = 9.0489001802816564e-11 * cse405;
	const amrex::Real cse407 = cse160 * cse9;
	const amrex::Real cse408 = 0.044330000000000001 * cse407 - 0.28520000000000001;
	const amrex::Real cse409 = cse407 * cse408 - 1.6759999999999999;
	const amrex::Real cse410 = cse407 * cse409 + 11.19;
	const amrex::Real cse411 = state.xn[2] * state.xn[6];
	const amrex::Real cse412 = cse370 * cse410 * cse411;
	const amrex::Real cse413 = 4.9999999999999997e-12 * cse412;
	const amrex::Real cse414 = cse108 * cse98;
	const amrex::Real cse415 = 7.5899999999999998e-12 * cse414;
	const amrex::Real cse416 = std::pow(state.T, -0.5);
	const amrex::Real cse417 = cse411 * cse416 * (cse407 * (0.044330000000000001 * cse160 * cse79 + cse408 * cse81) + cse409 * cse81);
	const amrex::Real cse418 = 9.9999999999999994e-12 * cse417;
	const amrex::Real cse419 = std::pow(cse157, 1.8740000000000001);
	const amrex::Real cse420 = 7.538e-5 * cse160 + 0.89639999999999997;
	const amrex::Real cse421 = std::pow(cse159, -cse420);
	const amrex::Real cse422 = cse421 * std::pow(state.T, 1.635e-6);
	const amrex::Real cse423 = 43750.0 * cse422;
	const amrex::Real cse424 = cse423 + 1;
	const amrex::Real cse425 = cse419 * cse424;
	const amrex::Real cse426 = -cse420;
	const amrex::Real cse427 = 1.534817233071468e-7 * cse419;
	const amrex::Real cse428 = cse424 * cse427 + 1;
	const amrex::Real cse429 = state.xn[13] / std::pow(cse428, 2);
	const amrex::Real cse430 =
	    cse429 * (-1.4381237473879656e-7 * cse4 * cse425 -
		      cse427 * (0.071531250000000005 * cse421 * std::pow(state.T, -0.99999836499999994) + cse423 * (cse168 * cse426 - 7.538e-5 * cse169)));
	const amrex::Real cse431 = 2.1659999999999999e-12 * cse174;
	const amrex::Real cse432 = cse430 * cse431;
	const amrex::Real cse433 = std::pow(cse157, 1.512);
	const amrex::Real cse434 = 5.4939999999999998e-7 * cse160 + 0.49559999999999998;
	const amrex::Real cse435 = std::pow(cse159, -cse434);
	const amrex::Real cse436 = cse435 * std::pow(state.T, 3.9029999999999999e-7);
	const amrex::Real cse437 = 5115.0 * cse436;
	const amrex::Real cse438 = cse437 + 1;
	const amrex::Real cse439 = cse433 * cse438;
	const amrex::Real cse440 = -cse434;
	const amrex::Real cse441 = 7.104740267838494e-7 * cse433;
	const amrex::Real cse442 = cse438 * cse441 + 1;
	const amrex::Real cse443 = state.xn[6] / std::pow(cse442, 2);
	const amrex::Real cse444 =
	    cse443 * (-5.3711836424859012e-7 * cse4 * cse439 - cse441 * (0.0019963845000000001 * cse435 * std::pow(state.T, -0.99999960970000001) +
									 cse437 * (cse168 * cse440 - 5.4939999999999998e-7 * cse169)));
	const amrex::Real cse445 = 5.5720000000000001e-12 * cse174;
	const amrex::Real cse446 = cse444 * cse445;
	const amrex::Real cse447 = std::pow(cse157, 1.1279999999999999);
	const amrex::Real cse448 = 0.00013329999999999999 * cse160 + 0.81200000000000006;
	const amrex::Real cse449 = std::pow(cse159, -cse448);
	const amrex::Real cse450 = cse449 * std::pow(state.T, 0.04845);
	const amrex::Real cse451 = 433.10000000000002 * cse450;
	const amrex::Real cse452 = cse451 + 1;
	const amrex::Real cse453 = cse447 * cse452;
	const amrex::Real cse454 = -cse448;
	const amrex::Real cse455 = 0.011078790689599145 * cse447;
	const amrex::Real cse456 = cse452 * cse455 + 1;
	const amrex::Real cse457 = state.xn[8] / std::pow(cse456, 2);
	const amrex::Real cse458 =
	    cse457 * (-0.0062484379489339175 * cse4 * cse453 - cse455 * (20.983695000000001 * cse449 * std::pow(state.T, -0.95155000000000001) +
									 cse451 * (cse168 * cse454 - 0.00013329999999999999 * cse169)));
	const amrex::Real cse459 = 4.5579999999999993e-11 * cse174;
	const amrex::Real cse460 = cse458 * cse459;
	const amrex::Real cse461 = cse192 * (cse128 + cse181 + cse191 + cse372 + cse373 + cse381 + cse387 + cse390 - cse404 + cse406 + cse413 - cse415 -
					     cse418 - cse432 - cse446 - cse460 + 4.2899999999999994e-13 * cse99);
	const amrex::Real cse462 = 1.0 / cse442;
	const amrex::Real cse463 = 5.5720000000000001e-12 * cse462;
	const amrex::Real cse464 = d2g * state.xn[6];
	const amrex::Real cse465 = cse463 * cse464;
	const amrex::Real cse466 = 1.0 / cse456;
	const amrex::Real cse467 = 4.5579999999999993e-11 * cse466;
	const amrex::Real cse468 = d2g * state.xn[8];
	const amrex::Real cse469 = cse467 * cse468;
	const amrex::Real cse470 = crate * cse37;
	const amrex::Real cse471 = cse470 * state.xn[7];
	const amrex::Real cse472 = 1040.0 * cse471;
	const amrex::Real cse473 = cse469 + cse472;
	const amrex::Real cse474 = 1.0 / cse428;
	const amrex::Real cse475 = 2.1659999999999999e-12 * cse474;
	const amrex::Real cse476 = d2g * state.xn[13];
	const amrex::Real cse477 = cse470 * state.xn[12];
	const amrex::Real cse478 = cse475 * cse476 + 16800.0 * cse477;
	const amrex::Real cse479 = cse465 + cse473 + cse478;
	const amrex::Real cse480 = cse211 * cse361;
	const amrex::Real cse481 = cse479 - cse480;
	const amrex::Real cse482 = cse410 * cse416;
	const amrex::Real cse483 = 9.9999999999999994e-12 * cse482;
	const amrex::Real cse484 = cse443 * (0.0036340746469993896 * cse114 * cse433 * cse436 * cse440 + 1.0742367284971802e-6 * cse114 * cse439);
	const amrex::Real cse485 = cse445 * cse484 + cse483 * state.xn[6];
	const amrex::Real cse486 = cse394 * cse90 + cse403;
	const amrex::Real cse487 = cse486 * state.xn[8];
	const amrex::Real cse488 = cse457 * (4.7982242476653898 * cse114 * cse447 * cse450 * cse454 + 0.012496875897867835 * cse114 * cse453);
	const amrex::Real cse489 = cse459 * cse488 + cse487;
	const amrex::Real cse490 = std::pow(state.T, -0.62);
	const amrex::Real cse491 = 1.4595000290776864e-10 * cse490;
	const amrex::Real cse492 = cse429 * (0.0067148253946876721 * cse114 * cse419 * cse422 * cse426 + 2.8762474947759312e-7 * cse114 * cse425);
	const amrex::Real cse493 = cse431 * cse492 + cse491 * state.xn[13];
	const amrex::Real cse494 = cse238 + cse240;
	const amrex::Real cse495 = 1.3007021411393064e-6 * cse230;
	const amrex::Real cse496 = ((cse375) ? (0) : (-0.00011320754716981131 * cse233 * state.xn[3]));
	const amrex::Real cse497 = cse385 * cse496;
	const amrex::Real cse498 = cse389 * cse496;
	const amrex::Real cse499 = cse497 - cse498;
	const amrex::Real cse500 = cse229 + cse495 * state.xn[16] + cse499;
	const amrex::Real cse501 = 2 * cse197;
	const amrex::Real cse502 = cse285 * cse361;
	const amrex::Real cse503 = crate * cse12;
	const amrex::Real cse504 = 16800.0 * cse503;
	const amrex::Real cse505 = cse474 * cse476;
	const amrex::Real cse506 = 33600.0 * cse477 + 4.3319999999999997e-12 * cse505;
	const amrex::Real cse507 = -cse504 * state.xn[12] + cse506;
	const amrex::Real cse508 = 3.2999999999999996e-13 * cse276;
	const amrex::Real cse509 = cse508 * state.xn[8];
	const amrex::Real cse510 = 1040.0 * cse503;
	const amrex::Real cse511 = cse466 * cse468;
	const amrex::Real cse512 = 2080.0 * cse471 + 9.1159999999999986e-11 * cse511;
	const amrex::Real cse513 = -cse510 * state.xn[7] + cse512;
	const amrex::Real cse514 = cse509 + cse513;
	const amrex::Real cse515 = ((cse375) ? (0) : (cse117));
	const amrex::Real cse516 = cse385 * cse515;
	const amrex::Real cse517 = cse389 * cse515;
	const amrex::Real cse518 = cse516 - cse517;
	const amrex::Real cse519 = cse283 + cse518;
	const amrex::Real cse520 = cse281 - cse282;
	const amrex::Real cse521 = cse462 * cse464;
	const amrex::Real cse522 = 1.1144e-11 * cse521;
	const amrex::Real cse523 = cse253 + cse522;
	const amrex::Real cse524 = cse299 * cse361;
	const amrex::Real cse525 = -cse524;
	const amrex::Real cse526 = 1.1000000000000001 * crate;
	const amrex::Real cse527 = cse174 * cse463 + cse483 * state.xn[2];
	const amrex::Real cse528 = 1.0537934368240927e-9 * cse364 + cse369 * cse370;
	const amrex::Real cse529 = cse528 * state.xn[16];
	const amrex::Real cse530 = 2.808e-22 * ncol_H2;
	const amrex::Real cse531 = chi * std::exp(-3.7599999999999998 * av) * std::exp(-cse530) * std::exp(-1.6000000000000001e-17 * ncol_C) / (cse530 + 1);
	const amrex::Real cse532 = 3.8500000000000001 * crate + cse510 * state.xn[3] + 3.4999999999999998e-10 * cse531;
	const amrex::Real cse533 = cse508 * state.xn[3];
	const amrex::Real cse534 = cse486 * state.xn[2];
	const amrex::Real cse535 = cse174 * cse467 + cse534;
	const amrex::Real cse536 = cse533 + cse535;
	const amrex::Real cse537 = 1.99e-9 * cse319 * cse379;
	const amrex::Real cse538 = cse537 * state.xn[16];
	const amrex::Real cse539 = cse322 + cse538;
	const amrex::Real cse540 = std::exp(-2.6099999999999999 * av);
	const amrex::Real cse541 = chi * cse540;
	const amrex::Real cse542 = cse504 * state.xn[3] + 4.4999999999999998e-9 * cse541;
	const amrex::Real cse543 = cse174 * cse475 + cse491 * state.xn[2];
	const amrex::Real cse544 = cse213 + cse481;
	const amrex::Real cse545 = 1.6715999999999999e-11 * cse521;
	const amrex::Real cse546 = cse528 * state.xn[7];
	const amrex::Real cse547 = cse537 * state.xn[11];
	const amrex::Real cse548 = cse495 * state.xn[2];
	const amrex::Real cse549 = cse351 * cse361;
	const amrex::Real cse550 = -cse549;
	const amrex::Real cse551 = 1.3673999999999998e-10 * cse511;
	const amrex::Real cse552 = 3120.0 * cse471;
	const amrex::Real cse553 = cse551 + cse552;
	const amrex::Real cse554 = 50400.0 * cse477 + 6.4979999999999992e-12 * cse505;
	const amrex::Real cse555 = cse356 + cse357;
	const amrex::Real cse556 = -3.0000000000000002e-15 * d2g * state.xn[0];
	const amrex::Real cse557 = cse30 * state.xn[3];
	const amrex::Real cse558 = state.xn[0] * state.xn[3];
	const amrex::Real cse559 = cse558 * cse61;
	const amrex::Real cse560 = cse31 * cse56;
	const amrex::Real cse561 = -cse73;
	const amrex::Real cse562 = cse31 * cse87;
	const amrex::Real cse563 = cse558 * cse92;
	const amrex::Real cse564 = cse415 - 4.2899999999999994e-13 * cse96 * cse97 * state.xn[3] * state.xn[8];
	const amrex::Real cse565 = cse180 - cse182;
	const amrex::Real cse566 = -cse105 + cse107;
	const amrex::Real cse567 = cse190 + cse372 + cse381 + cse387 + cse390 + cse566;
	const amrex::Real cse568 = cse192 * (cse181 - cse561 - cse562 - cse563 - cse564 - cse565 - cse567 - cse70 - 2.3622303960273254e-7 * cse93);
	const amrex::Real cse569 = cse221 * cse558;
	const amrex::Real cse570 = cse219 + cse480 + cse556 + cse569;
	const amrex::Real cse571 = 4.5427507615910105e-7 * cse230;
	const amrex::Real cse572 = -cse238 + cse499;
	const amrex::Real cse573 = cse237 - cse239;
	const amrex::Real cse574 = 5.6999999999999997e-11 * cse274;
	const amrex::Real cse575 = cse270 * cse558;
	const amrex::Real cse576 = 7.2000000000000002e-15 * state.xn[6];
	const amrex::Real cse577 = cse249 + cse576;
	const amrex::Real cse578 = cse501 + cse502;
	const amrex::Real cse579 = cse251 + cse288 - cse516 + cse517;
	const amrex::Real cse580 = cse283 + cse520;
	const amrex::Real cse581 = cse294 * cse558;
	const amrex::Real cse582 = cse293 * cse31;
	const amrex::Real cse583 = 7.2000000000000002e-15 * state.xn[3];
	const amrex::Real cse584 = 1.6999999999999999e-9 * state.xn[16];
	const amrex::Real cse585 = 9.0000000000000011e-15 * cse267;
	const amrex::Real cse586 = cse348 * cse558;
	const amrex::Real cse587 = 1.6999999999999999e-9 * state.xn[9];
	const amrex::Real cse588 = cse350 + cse546 + cse547 + cse587;
	const amrex::Real cse589 = -cse192 * cse566;
	const amrex::Real cse590 = -cse77;
	const amrex::Real cse591 = -cse413 + cse418 + cse446 + cse590 + cse70;
	const amrex::Real cse592 = cse192 * cse591;
	const amrex::Real cse593 = cse522 + cse577;
	const amrex::Real cse594 = 1.6000000000000001e-9 * state.xn[9];
	const amrex::Real cse595 = cse308 + cse527 + cse583 + cse594;
	const amrex::Real cse596 = 1.6000000000000001e-9 * state.xn[6];
	const amrex::Real cse597 = cse338 + cse465;
	const amrex::Real cse598 = -cse192 * cse591;
	const amrex::Real cse599 = cse404 + cse460;
	const amrex::Real cse600 = cse192 * (cse133 + cse372 + 2.2769999999999997e-12 * cse414 + cse599 + cse73 - 1.287e-13 * cse99);
	const amrex::Real cse601 = cse470 * state.xn[9];
	const amrex::Real cse602 = 184.0 * cse601;
	const amrex::Real cse603 = -cse602;
	const amrex::Real cse604 = cse473 + cse603;
	const amrex::Real cse605 = 9.8999999999999995e-14 * cse276;
	const amrex::Real cse606 = 184.0 * cse503;
	const amrex::Real cse607 = cse606 * state.xn[9];
	const amrex::Real cse608 = 368.0 * cse601;
	const amrex::Real cse609 = -cse608;
	const amrex::Real cse610 = cse607 + cse609;
	const amrex::Real cse611 = 2.4e-10 * chi * fshield_co_interp2d(ncol_CO, ncol_H2) * std::exp(-3.8799999999999999 * av) + cse606 * state.xn[3];
	const amrex::Real cse612 = cse332 + cse333;
	const amrex::Real cse613 = cse339 + cse602;
	const amrex::Real cse614 = 552.0 * cse601;
	const amrex::Real cse615 = cse75 * state.xn[8];
	const amrex::Real cse616 = 1.1994963750000002e-8 * cse615;
	const amrex::Real cse617 = -cse616;
	const amrex::Real cse618 = cse192 * (-cse564 - cse599 - cse617);
	const amrex::Real cse619 = 9.1500000000000005e-10 * cse305;
	const amrex::Real cse620 = cse619 * state.xn[15];
	const amrex::Real cse621 = cse619 * state.xn[8];
	const amrex::Real cse622 = -cse134 * cse192;
	const amrex::Real cse623 = cse314 + cse584;
	const amrex::Real cse624 = cse596 + cse611;
	const amrex::Real cse625 = cse330 + cse602;
	const amrex::Real cse626 = cse192 * (cse128 + cse617);
	const amrex::Real cse627 = -cse381 + cse386 - cse390;
	const amrex::Real cse628 = cse181 + cse359;
	const amrex::Real cse629 = cse192 * (-cse627 - cse628);
	const amrex::Real cse630 = -cse406 + cse432;
	const amrex::Real cse631 = cse192 * cse630;
	const amrex::Real cse632 = -cse192 * cse630;
	const amrex::Real cse633 = cse192 * (cse371 + 5.3129999999999997e-12 * cse414 + cse561 - 3.0029999999999999e-13 * cse99);
	const amrex::Real cse634 = 2.3099999999999997e-13 * cse276;
	const amrex::Real cse635 = cse192 * (cse133 + cse565 + cse616 + cse627 + cse77);
	const amrex::Real cse636 = cse192 * (cse373 + cse567);
	const amrex::Real cse637 = cse192 * (cse183 + cse590 + cse628);
	const amrex::Real cse638 = 2.9387358770557188e-39 * cse195;
	const amrex::Real cse639 = 4.7173311503030642e-41 * cse197;
	const amrex::Real cse640 = 4.7173311503030642e-41 * cse199;
	const amrex::Real cse641 = 1.0338099956904871e-50 * cse0;
	const amrex::Real cse642 = cse641 * state.xn[14];
	const amrex::Real cse643 = 9.4346623006061284e-41 * crate * state.xn[3];
	const amrex::Real cse644 = 6.2631245169871218e-51 * cse505;
	const amrex::Real cse645 = 4.8578251101285161e-35 * cse477;
	const amrex::Real cse646 = -4.9060243963151868e-38 * crate * cse36 * state.xn[3] * state.xn[7] - 5.7787306591212543e-52 * cse204 * d2g * state.xn[1] -
				   2.6284969169488674e-52 * cse462 * d2g * state.xn[6] - 2.1501595383081364e-51 * cse466 * d2g * state.xn[8] + cse644 + cse645;
	const amrex::Real cse647 = 1.6737729999999998e-24 * state.xn[0] + 1.6728620616289998e-24 * state.xn[1] + 4.8185222061629005e-23 * state.xn[10] +
				   2.6567630000000005e-23 * state.xn[11] + 4.6637070000000001e-23 * state.xn[12] + 4.6636159061629004e-23 * state.xn[13] +
				   2.1618503000000003e-23 * state.xn[14] + 2.8241403000000005e-23 * state.xn[15] + 5.0204080616289998e-24 * state.xn[16] +
				   2.6566719061629008e-23 * state.xn[17] + 9.1093837099999994e-28 * state.xn[2] + 3.3475459999999996e-24 * state.xn[3] +
				   3.3466350616289996e-24 * state.xn[4] + 6.6464729999999996e-24 * state.xn[5] + 6.6455620616289996e-24 * state.xn[6] +
				   1.9944730000000003e-23 * state.xn[7] + 1.9943819061629003e-23 * state.xn[8] + 4.6512360000000008e-23 * state.xn[9];
	const amrex::Real cse648 = std::pow(cse647, -2);
	const amrex::Real cse649 = 1.380649e-16 * state.xn[0] + 1.380649e-16 * state.xn[1] + 1.380649e-16 * state.xn[10] + 1.380649e-16 * state.xn[11] +
				   1.380649e-16 * state.xn[12] + 1.380649e-16 * state.xn[13] + 1.380649e-16 * state.xn[14] + 1.380649e-16 * state.xn[15] +
				   1.380649e-16 * state.xn[16] + 1.380649e-16 * state.xn[17] + 1.380649e-16 * state.xn[2] + 1.380649e-16 * state.xn[3] +
				   1.380649e-16 * state.xn[4] + 1.380649e-16 * state.xn[5] + 1.380649e-16 * state.xn[6] + 1.380649e-16 * state.xn[7] +
				   1.380649e-16 * state.xn[8] + 1.380649e-16 * state.xn[9];
	const amrex::Real cse650 = 1.4999999999999249 * cse649;
	const amrex::Real cse651 = cse648 * cse650 * state.T;
	const amrex::Real cse652 = 1.8161724928666798e-40 * crate;
	const amrex::Real cse653 = 1.7073058169602635e-38 * crate;
	const amrex::Real cse654 = 2.5713938924237541e-50 * state.xn[14];
	const amrex::Real cse655 = 1.285696946211877e-50 * state.xn[15];
	const amrex::Real cse656 = 4.9958509909947217e-48 * state.xn[16];
	const amrex::Real cse657 = 4.220235648504098e-49 * cse490;
	const amrex::Real cse658 = cse657 * state.xn[13];
	const amrex::Real cse659 = 7.2737227656057311e-46 * cse230;
	const amrex::Real cse660 = cse659 * state.xn[16];
	const amrex::Real cse661 = 3.34281206015088e-49 * cse331;
	const amrex::Real cse662 = 1.3958995416014663e-49 * cse340;
	const amrex::Real cse663 = 2.9208719207347922e-49 * cse309;
	const amrex::Real cse664 = cse663 * state.xn[15];
	const amrex::Real cse665 = cse305 * state.xn[15];
	const amrex::Real cse666 = 2.6889433275059828e-48 * cse665;
	const amrex::Real cse667 = 4.626500104884301e-48 * cse177;
	const amrex::Real cse668 = cse667 * state.xn[17];
	const amrex::Real cse669 = 6.3713565952657906e-49 * cse320;
	const amrex::Real cse670 = cse669 * state.xn[16];
	const amrex::Real cse671 = 1.6510659026060724e-50 * cse531;
	const amrex::Real cse672 = 4.9060243963151868e-38 * cse503;
	const amrex::Real cse673 = cse672 * state.xn[7];
	const amrex::Real cse674 = 4.0387529294731574e-48 * cse122;
	const amrex::Real cse675 = cse178 * cse674;
	const amrex::Real cse676 = 9.4346623006061284e-41 * cse197;
	const amrex::Real cse677 = 6.2631245169871218e-51 * cse174;
	const amrex::Real cse678 = cse474 * cse677;
	const amrex::Real cse679 =
	    1.3012031544987096e-47 * chi * cse540 * state.xn[12] + 4.8578251101285161e-35 * crate * cse12 * state.xn[12] * state.xn[3] -
	    5.1890642653333708e-41 * crate * state.xn[5] + 6.465218929522581e-49 * cse100 * cse101 * state.xn[3] * state.xn[4] +
	    2.9387358770557188e-39 * cse109 * cse94 * state.xn[1] * state.xn[11] + 5.7787306591212543e-52 * cse11 * cse204 * d2g * state.xn[1] +
	    2.6284969169488674e-52 * cse11 * cse462 * d2g * state.xn[6] + 2.1501595383081364e-51 * cse11 * cse466 * d2g * state.xn[8] +
	    2.3026978988630935e-46 * cse146 * cse148 * state.xn[1] * state.xn[2] + 2.7717497761966105e-44 * cse241 * state.xn[10] * state.xn[2] +
	    1.0042319124598398e-52 * cse275 * cse97 * state.xn[3] * state.xn[8] + 3.4713817547720679e-48 * cse305 * state.xn[15] * state.xn[6] +
	    9.3874889891030969e-50 * cse319 * cse376 * cse378 * state.xn[11] * state.xn[16] +
	    4.7173311503030638e-52 * cse410 * cse416 * state.xn[2] * state.xn[6] + 4.7173311503030642e-41 * cse486 * state.xn[2] * state.xn[8] +
	    4.7173311503030642e-41 * cse528 * state.xn[16] * state.xn[7] - cse638 * state.xn[0] - cse639 * state.xn[0] - cse640 * state.xn[0] -
	    cse642 * state.xn[0] - cse652 * state.xn[7] - cse653 * state.xn[9] - cse654 * state.xn[11] - cse655 * state.xn[11] - cse656 * state.xn[9] -
	    cse658 * state.xn[2] - cse660 * state.xn[2] - cse661 * state.xn[14] - cse662 * state.xn[15] - cse664 * state.xn[7] - cse666 * state.xn[8] -
	    cse668 * state.xn[3] - cse670 * state.xn[11] - cse671 * state.xn[7] - cse673 * state.xn[3] - cse675 * state.xn[3] - cse676 * state.xn[3] -
	    cse678 * state.xn[13];
	const amrex::Real cse680 = cse648 * cse679;
	const amrex::Real cse681 = cse63 * cse680;
	const amrex::Real cse682 = std::pow(cse647, -3);
	const amrex::Real cse683 = cse374 + 0.070000000000000007;
	const amrex::Real cse684 = 1.0 / cse683;
	const amrex::Real cse685 = std::sqrt(cse374 * cse684);
	const amrex::Real cse686 = 26.399999999999999 * cse685 + 6.5;
	const amrex::Real cse687 = 2.1789607199999999e-11 * cse199;
	const amrex::Real cse688 = 2.25 * std::pow(state.T, 0.14699999999999999) + 5.2199999999999998;
	const amrex::Real cse689 = cse159 + 50;
	const amrex::Real cse690 = 0.0043 * std::pow(cse689, 0.69199999999999995) + 1;
	const amrex::Real cse691 = 0.049959999999999997 * std::pow(cse689, 0.43099999999999999) * cse690 + 1;
	const amrex::Real cse692 = 1.0 / cse691;
	const amrex::Real cse693 = psigd_coll_interp2d(cse11, state.T);
	const amrex::Real cse694 = 100.0 * cse174;
	const amrex::Real cse695 = cse693 * cse694;
	const amrex::Real cse696 = state.T - 2.1544346900318838 * std::pow(85973.158732959506 * cse156 + 4.1397668473788896, 1.0 / 6.0);
	const amrex::Real cse697 = cse35 * d2g;
	const amrex::Real cse698 = 3.2000000000000002e-32 * cse3;
	const amrex::Real cse699 = cse697 * cse698;
	const amrex::Real cse700 = cse696 * cse699;
	const amrex::Real cse701 = std::pow(state.T, 0.14999999999999999);
	const amrex::Real cse702 = 0.00088240812266906664 * std::pow(state.T, 0.65000000000000002) + 1;
	const amrex::Real cse703 = 1.0 / cse702;
	const amrex::Real cse704 = cse701 * cse703;
	const amrex::Real cse705 = cse703 * state.xn[2];
	const amrex::Real cse706 = cse701 * cse705;
	const amrex::Real cse707 = 4.4460389837719569e-9 * cse706 + 626500000.0;
	const amrex::Real cse708 = 1.0 / cse707;
	const amrex::Real cse709 = std::exp(-118400.0 * cse4);
	const amrex::Real cse710 = cse708 * cse709;
	const amrex::Real cse711 = 1.3338116951315871e-8 * cse710;
	const amrex::Real cse712 = cse706 * cse711 + 1;
	const amrex::Real cse713 = 1.0 / cse712;
	const amrex::Real cse714 = cse710 * cse713;
	const amrex::Real cse715 = 1.3659996292638161e-10 * cse714;
	const amrex::Real cse716 = cse704 * cse715;
	const amrex::Real cse717 = state.xn[2] + 1.0e-50;
	const amrex::Real cse718 = cse156 / cse717;
	const amrex::Real cse719 = 1.7 * cse3;
	const amrex::Real cse720 = cse718 * cse719 + 50;
	const amrex::Real cse721 = std::log(cse720);
	const amrex::Real cse722 = 0.45350000000000001 + 2.234 / cse721;
	const amrex::Real cse723 = std::pow(state.T, cse722);
	const amrex::Real cse724 = 1.4419999999999999 - 0.050889999999999998 * cse721;
	const amrex::Real cse725 = std::exp(cse721 * cse724);
	const amrex::Real cse726 = 1.8998126555754159e-29 * cse723 * cse725;
	const amrex::Real cse727 = cse174 * cse726;
	const amrex::Real cse728 = cse727 * state.xn[2];
	const amrex::Real cse729 = 0.001 * state.T;
	const amrex::Real cse730 = cse729 + 1;
	const amrex::Real cse731 = std::pow(cse730, -2);
	const amrex::Real cse732 = 1.0 / cse730;
	const amrex::Real cse733 = std::pow(10, -2.0230000000000001 * cse731 - 3.6709999999999998 * cse732 - 11.08);
	const amrex::Real cse734 = cse12 * cse733;
	const amrex::Real cse735 = cse734 * state.xn[3];
	const amrex::Real cse736 = std::pow(10, -2.3900000000000001 * cse731 + 0.055500000000000001 * cse732 - 11.06);
	const amrex::Real cse737 = cse12 * cse736;
	const amrex::Real cse738 = cse737 * state.xn[0];
	const amrex::Real cse739 = cse735 + cse738;
	const amrex::Real cse740 = 1.0 / cse739;
	const amrex::Real cse741 = cse574 + 1.9999999999999999e-7;
	const amrex::Real cse742 = cse740 * cse741;
	const amrex::Real cse743 = cse12 * cse742 + 1;
	const amrex::Real cse744 = 1.0 / cse743;
	const amrex::Real cse745 = 6.7291434000000005e-12 * cse744 + 3.2043540000000002e-13;
	const amrex::Real cse746 = cse9 * std::log(cse11);
	const amrex::Real cse747 = cse746 < 2;
	const amrex::Real cse748 = cse746 < 4;
	const amrex::Real cse749 = cse746 < 7;
	const amrex::Real cse750 = cse746 < 10;
	const amrex::Real cse751 =
	    ((cse747) ? (10.0)
		      : ((cse748) ? ((3.0 / 2.0) * cse746 + 7)
				  : ((cse749) ? ((4.0 / 3.0) * cse746 + 23.0 / 3.0) : ((cse750) ? ((1.0 / 3.0) * cse746 + 44.0 / 3.0) : (18.0)))));
	const amrex::Real cse752 = 2.5634832e-11 * cse744 + 6.4087080000000003e-13;
	const amrex::Real cse753 = cse74 * state.xn[2];
	const amrex::Real cse754 = (1.0 / 100.0) * state.T;
	const amrex::Real cse755 = std::log(cse754);
	const amrex::Real cse756 = 0.0086999999999999994 * cse755;
	const amrex::Real cse757 = std::pow(cse754, cse756 + 0.12809999999999999);
	const amrex::Real cse758 = 7.5799999999999997e-10 * cse757 * state.xn[0];
	const amrex::Real cse759 = state.T < 500;
	const amrex::Real cse760 = std::pow(state.T, 0.070000000000000007);
	const amrex::Real cse761 = 0.75 * ((cse759) ? (1.1000000000000001e-13 * state.T + 5.3300000000000002e-10) : (3.74757785025e-10 * cse760)) +
				   0.25 * ((cse759) ? (3.3000000000000001e-13 * state.T + 4.4299999999999997e-10) : (3.8899728635600003e-10 * cse760));
	const amrex::Real cse762 = 4.5299999999999998e-6 * cse753 + cse758 + cse761 * state.xn[3];
	const amrex::Real cse763 = cse762 + 2.3e-6;
	const amrex::Real cse764 = std::exp(-91.209999999999994 * cse4);
	const amrex::Real cse765 = cse764 / cse763;
	const amrex::Real cse766 = 2 * cse765;
	const amrex::Real cse767 = cse762 * cse766 + 1;
	const amrex::Real cse768 = 1.0 / cse767;
	const amrex::Real cse769 = cse768 * state.xn[8];
	const amrex::Real cse770 = cse762 * cse765;
	const amrex::Real cse771 = 5.7927337833400002e-20 * cse770;
	const amrex::Real cse772 = std::pow(state.T, 1.0 / 4.0);
	const amrex::Real cse773 = 680.10000000000002 / cse772;
	const amrex::Real cse774 = 1.75 * cse772;
	const amrex::Real cse775 = cse773 * state.xn[2] + cse774 * state.xn[0] + state.xn[3];
	const amrex::Real cse776 = std::exp(-state.T);
	const amrex::Real cse777 = 1 - cse776;
	const amrex::Real cse778 = std::pow(cse777, 1000);
	const amrex::Real cse779 = l0_co_interp1d(state.T);
	const amrex::Real cse780 = 1.0 / cse779;
	const amrex::Real cse781 = M_SQRT2;
	const amrex::Real cse782 = std::max(gradv, 5.7445531974402559e-18 * cse3 * cse781);
	const amrex::Real cse783 = 100000.0 / cse782;
	const amrex::Real cse784 = ncol_H <= 0;
	const amrex::Real cse785 = cse12 * ncol_H;
	const amrex::Real cse786 = std::max(1.0000000000000001e-30, cse785);
	const amrex::Real cse787 = 1756701454.4310031 * ncol_CO;
	const amrex::Real cse788 = ((cse784) ? (cse783 * state.xn[9]) : (cse787 / std::sqrt(cse786)));
	const amrex::Real cse789 = llte_co_interp2d(state.T, cse788);
	const amrex::Real cse790 = 1.0 / cse789;
	const amrex::Real cse791 = nhalf_co_interp2d(state.T, cse788);
	const amrex::Real cse792 = cse779 * cse790;
	const amrex::Real cse793 = -cse791 * cse792 + 1;
	const amrex::Real cse794 = 1.0 / cse791;
	const amrex::Real cse795 = cse775 * cse794;
	const amrex::Real cse796 = alpha_co_interp2d(state.T, cse788);
	const amrex::Real cse797 = std::pow(cse795, cse796);
	const amrex::Real cse798 = cse780 * cse797;
	const amrex::Real cse799 = cse793 * cse798;
	const amrex::Real cse800 = cse775 * cse790 + cse780 + cse799;
	const amrex::Real cse801 = 1.0 / cse800;
	const amrex::Real cse802 = cse778 * cse801;
	const amrex::Real cse803 = cse802 * state.xn[9];
	const amrex::Real cse804 = 7.1777529600000012e-12 * cse557;
	const amrex::Real cse805 = std::pow(state.T, 0.92600000000000005);
	const amrex::Real cse806 = 1.08e-14 * cse805;
	const amrex::Real cse807 = 0.16 * cse755;
	const amrex::Real cse808 = std::pow(cse754, 0.755 - cse807);
	const amrex::Real cse809 = 4.34e-10 * cse808;
	const amrex::Real cse810 = cse809 * state.xn[0];
	const amrex::Real cse811 = 0.042999999999999997 * cse755;
	const amrex::Real cse812 = 5.2499999999999998e-13 * std::pow(cse754, cse811 + 0.88900000000000001);
	const amrex::Real cse813 = 0.52500000000000002 * cse755;
	const amrex::Real cse814 = 2.2500000000000003e-12 * std::pow(cse754, cse813 + 1.198);
	const amrex::Real cse815 = cse812 + cse814;
	const amrex::Real cse816 = cse806 * state.xn[2] + cse810 + cse815 * state.xn[3];
	const amrex::Real cse817 = std::pow(state.T, -0.074999999999999997);
	const amrex::Real cse818 = 5.1199999999999999e-10 * cse817;
	const amrex::Real cse819 = 0.0030000000000000001 * cse755;
	const amrex::Real cse820 = std::pow(cse754, 0.41899999999999998 - cse819);
	const amrex::Real cse821 = 3.5700000000000001e-10 * cse820;
	const amrex::Real cse822 = cse821 * state.xn[0];
	const amrex::Real cse823 = 0.025000000000000001 * cse755;
	const amrex::Real cse824 = 3.7250000000000001e-11 * std::pow(cse754, cse823 + 0.26400000000000001);
	const amrex::Real cse825 = 1.0274999999999999e-10 * std::pow(cse754, cse811 + 0.29599999999999999);
	const amrex::Real cse826 = cse824 + cse825;
	const amrex::Real cse827 = cse818 * state.xn[2] + cse822 + cse826 * state.xn[3];
	const amrex::Real cse828 = std::exp(-326.60000000000002 * cse4);
	const amrex::Real cse829 = (1.0 / 5.0) * cse828;
	const amrex::Real cse830 = cse827 * cse829;
	const amrex::Real cse831 = std::exp(-425.40000000000003 * cse4);
	const amrex::Real cse832 = cse816 * cse831;
	const amrex::Real cse833 = std::pow(state.T, -0.025999999999999999);
	const amrex::Real cse834 = 4.8599999999999998e-10 * cse833;
	const amrex::Real cse835 = 0.0060000000000000001 * cse755;
	const amrex::Real cse836 = std::pow(cse754, 0.36899999999999999 - cse835);
	const amrex::Real cse837 = 3.1899999999999998e-10 * cse836;
	const amrex::Real cse838 = cse837 * state.xn[0];
	const amrex::Real cse839 = 0.041000000000000002 * cse755;
	const amrex::Real cse840 = 4.7499999999999998e-11 * std::pow(cse754, cse839 + 0.20300000000000001);
	const amrex::Real cse841 = 0.058000000000000003 * cse755;
	const amrex::Real cse842 = 1.6725000000000002e-10 * std::pow(cse754, cse841 + 0.23699999999999999);
	const amrex::Real cse843 = cse840 + cse842;
	const amrex::Real cse844 = cse834 * state.xn[2] + cse838 + cse843 * state.xn[3];
	const amrex::Real cse845 = (1.0 / 15.0) * cse844;
	const amrex::Real cse846 = cse827 + 8.9099999999999997e-5;
	const amrex::Real cse847 = cse829 * cse844;
	const amrex::Real cse848 = cse816 * cse830 + cse832 * cse845 + cse846 * cse847;
	const amrex::Real cse849 = cse816 + 1.7499999999999998e-5;
	const amrex::Real cse850 = cse844 + 1.34e-10;
	const amrex::Real cse851 = std::exp(-98.800000000000011 * cse4);
	const amrex::Real cse852 = cse816 * cse851;
	const amrex::Real cse853 = (1.0 / 3.0) * cse852;
	const amrex::Real cse854 = std::exp(-227.80000000000001 * cse4);
	const amrex::Real cse855 = cse827 * cse854;
	const amrex::Real cse856 = (3.0 / 5.0) * cse855;
	const amrex::Real cse857 = cse847 * cse849 + cse849 * cse856 + cse850 * cse856;
	const amrex::Real cse858 = cse846 * cse849 + cse846 * cse850 + cse848 + cse850 * cse853 + cse857;
	const amrex::Real cse859 = 1.0 / cse858;
	const amrex::Real cse860 = 2.3872025442750958e-19 * cse859;
	const amrex::Real cse861 = 2.8023005140019997e-18 * cse859;
	const amrex::Real cse862 = cse848 * cse860 + cse857 * cse861;
	const amrex::Real cse863 = state.T < 100.0;
	const amrex::Real cse864 = state.T < 6000.0;
	const amrex::Real cse865 = state.T < 10;
	const amrex::Real cse866 = std::log(cse729);
	const amrex::Real cse867 = cse866 * cse9;
	const amrex::Real cse868 = std::pow(cse8, -8);
	const amrex::Real cse869 = std::pow(cse866, 8) * cse868;
	const amrex::Real cse870 = std::pow(cse8, -7);
	const amrex::Real cse871 = std::pow(cse866, 7) * cse870;
	const amrex::Real cse872 = std::pow(cse8, -6);
	const amrex::Real cse873 = std::pow(cse866, 6) * cse872;
	const amrex::Real cse874 = std::pow(cse8, -5);
	const amrex::Real cse875 = std::pow(cse866, 5) * cse874;
	const amrex::Real cse876 = std::pow(cse8, -4);
	const amrex::Real cse877 = std::pow(cse866, 4) * cse876;
	const amrex::Real cse878 = std::pow(cse8, -3);
	const amrex::Real cse879 = std::pow(cse866, 3);
	const amrex::Real cse880 = cse878 * cse879;
	const amrex::Real cse881 = cse14 * std::pow(cse866, 2);
	const amrex::Real cse882 = std::pow(10.0, 16.815729999999999 * cse867 + 70.609154000000004 * cse869 + 364.14445999999998 * cse871 +
						      801.81246999999996 * cse873 + 983.67575999999997 * cse875 + 734.71650999999997 * cse877 +
						      343.1918 * cse880 + 96.743155000000002 * cse881 - 21.928795999999998) *
				   state.xn[2];
	const amrex::Real cse883 = state.xn[5] + state.xn[6];
	const amrex::Real cse884 = std::pow(10.0, 1.5714710999999999 * cse867 + 0.32168730000000001 * cse875 - 0.51002221000000003 * cse877 -
						      0.23619984999999999 * cse880 + 0.015391166 * cse881 - 22.089523) *
				       state.xn[1] +
				   std::pow(10.0, 2.1892372 * cse867 + 0.19191374999999999 * cse875 - 0.16596184 * cse877 + 0.29036281000000003 * cse880 -
						      0.81520437999999995 * cse881 - 23.689236999999999) *
				       cse883 +
				   std::pow(10.0, 2.0943374000000001 * cse866 * cse9 - 0.033638326000000003 * cse875 - 0.14913216000000001 * cse877 +
						      0.43693353000000001 * cse878 * cse879 - 0.77151435999999995 * cse881 - 23.962112000000001) *
				       state.xn[3];
	const amrex::Real cse885 = std::pow(10, 37.383713 * cse867 + 3.8479610000000002 * cse875 + 20.159831000000001 * cse877 + 48.656103000000002 * cse880 +
						    58.145166000000003 * cse881 - 16.818342000000001) *
				       state.xn[0] +
				   cse884;
	const amrex::Real cse886 = std::pow(10.0, 1.6802758 * cse867 - 6.3701156000000001 * cse869 + 6.4380698000000001 * cse871 + 8.9167182999999994 * cse873 -
						      8.8077017000000009 * cse875 - 4.7274035999999997 * cse877 + 4.0406627000000004 * cse880 +
						      0.93310621999999999 * cse881 - 22.921188999999998) *
				   state.xn[2];
	const amrex::Real cse887 = state.T < 1000.0;
	const amrex::Real cse888 = std::pow(10.0, 3.5692468000000002 * cse866 * cse9 - 4.2519023000000002 * cse875 - 21.328264000000001 * cse877 -
						      27.850082 * cse880 - 11.33286 * cse881 - 24.311209000000002) *
				       state.xn[0] +
				   cse884;
	const amrex::Real cse889 = std::pow(10.0, 4.6450521 * cse867 + 1.5538288 * cse875 - 5.5108049000000001 * cse877 + 5.9369081000000001 * cse880 -
						      3.7209846 * cse881 - 24.311209000000002) *
				       state.xn[0] +
				   cse884;
	const amrex::Real cse890 = std::pow(10.0, -11.945856385319289 * cse14 + 28.69470131298657 * cse874 - 56.79810800956944 * cse876 +
						      34.150687525022562 * cse878 + 8.3228160852326614 * cse9 - 24.311209000000002) *
				       state.xn[0] +
				   std::pow(10.0, -2.6171337683481011 * cse14 + 3.5440891133599637 * cse874 - 1.7105157385968939 * cse876 +
						      1.6702447513374001 * cse878 + 3.9225864834863131 * cse9 - 23.689236999999999) *
				       cse883 +
				   std::pow(10.0, -2.4768712409536775 * cse14 - 0.62120210234156448 * cse874 - 1.5370575959566981 * cse876 +
						      2.5133588394664677 * cse878 + 3.7525488682084647 * cse9 - 23.962112000000001) *
				       state.xn[3] +
				   std::pow(10.0, 0.049411830040524525 * cse14 + 5.9406293599919788 * cse874 - 5.2566362076906961 * cse876 -
						      1.3586848802337363 * cse878 + 2.8156982240432273 * cse9 - 22.089523) *
				       state.xn[1];
	const amrex::Real cse891 =
	    ((cse759 && cse863 && cse864)
		 ? (cse865 || cse882 + cse885 < 1.0e-100)
		 : (((cse863 && cse864)
			 ? (cse865 || cse885 + cse886 < 1.0e-100)
			 : (((cse759 && cse864 && cse887)
				 ? (cse865 || cse882 + cse888 < 1.0e-100)
				 : (((cse864 && cse887)
					 ? (cse865 || cse886 + cse888 < 1.0e-100)
					 : (((cse759 && cse864)
						 ? (cse865 || cse882 + cse889 < 1.0e-100)
						 : (((cse864) ? (cse865 || cse886 + cse889 < 1.0e-100)
							      : (((cse759) ? (cse865 ||
									      std::pow(10.0, 310.58441786958315 * cse14 + 7500.646099484944 * cse868 +
												 21588.957206746247 * cse870 + 26530.843406310876 * cse872 +
												 18165.631967965237 * cse874 + 7572.4886742758581 * cse876 +
												 1974.131269262874 * cse878 + 30.129743459482278 * cse9 -
												 21.928795999999998) *
											  state.xn[2] +
										      cse890 <
										  1.0e-100)
									   : (cse865 ||
									      std::pow(10.0, 2.9956460707652877 * cse14 - 676.68255490510751 * cse868 +
												 381.69251128040054 * cse870 + 295.04162851880631 * cse872 -
												 162.65264843551881 * cse874 - 48.723840736518795 * cse876 +
												 23.242975457496804 * cse878 + 3.0106500755647456 * cse9 -
												 22.921188999999998) *
											  state.xn[2] +
										      cse890 <
										  1.0e-100))))))))))))));
	const amrex::Real cse892 = ((cse864) ? (cse4) : (0.00016666666666666666));
	const amrex::Real cse893 = std::exp(-11700.0 * cse892);
	const amrex::Real cse894 = std::exp(-5860.0 * cse892);
	const amrex::Real cse895 = std::exp(-510.0 * cse892);
	const amrex::Real cse896 = std::exp(-2197000.0 * ((cse864) ? (std::pow(state.T, -3)) : (4.6296296296296294e-12)));
	const amrex::Real cse897 = cse896 * ((cse864) ? (std::pow(state.T, 3.7599999999999998)) : (160638152785302.53));
	const amrex::Real cse898 = 6.0142468035272636e-8 * ((cse864) ? (std::pow(state.T, 2.1000000000000001)) : (85924621.118090734)) + 1.0;
	const amrex::Real cse899 = 1.0 / cse898;
	const amrex::Real cse900 = 1.6e-18 * cse893 + 6.7e-19 * cse894 + 3.0e-24 * cse895 + 4.985670872372847e-33 * cse897 * cse899;
	const amrex::Real cse901 = std::log(0.001 * ((cse864) ? (state.T) : (6000.0)));
	const amrex::Real cse902 = std::pow(cse901, 5);
	const amrex::Real cse903 = cse874 * cse902;
	const amrex::Real cse904 = std::pow(cse901, 4);
	const amrex::Real cse905 = cse876 * cse904;
	const amrex::Real cse906 = std::pow(cse901, 3);
	const amrex::Real cse907 = std::pow(cse901, 2);
	const amrex::Real cse908 = cse14 * cse907;
	const amrex::Real cse909 = std::pow(10.0, 0.43693353000000001 * cse878 * cse906 + 2.0943374000000001 * cse9 * cse901 - 0.033638326000000003 * cse903 -
						      0.14913216000000001 * cse905 - 0.77151435999999995 * cse908 - 23.962112000000001);
	const amrex::Real cse910 = cse909 * state.xn[3];
	const amrex::Real cse911 = cse9 * cse901;
	const amrex::Real cse912 = cse878 * cse906;
	const amrex::Real cse913 = std::pow(10.0, 0.32168730000000001 * cse903 - 0.51002221000000003 * cse905 + 0.015391166 * cse908 +
						      1.5714710999999999 * cse911 - 0.23619984999999999 * cse912 - 22.089523);
	const amrex::Real cse914 = cse913 * state.xn[1];
	const amrex::Real cse915 = std::pow(10.0, 0.19191374999999999 * cse903 - 0.16596184 * cse905 - 0.81520437999999995 * cse908 + 2.1892372 * cse911 +
						      0.29036281000000003 * cse912 - 23.689236999999999);
	const amrex::Real cse916 = cse883 * cse915;
	const amrex::Real cse917 = std::pow(10, 3.8479610000000002 * cse903 + 20.159831000000001 * cse905 + 58.145166000000003 * cse908 + 37.383713 * cse911 +
						    48.656103000000002 * cse912 - 16.818342000000001);
	const amrex::Real cse918 = ((cse864) ? (cse863) : (false));
	const amrex::Real cse919 = std::pow(10.0, 3.5692468000000002 * cse9 * cse901 - 4.2519023000000002 * cse903 - 21.328264000000001 * cse905 -
						      11.33286 * cse908 - 27.850082 * cse912 - 24.311209000000002);
	const amrex::Real cse920 = ((cse864) ? (cse887) : (false));
	const amrex::Real cse921 = std::pow(10.0, 1.5538288 * cse903 - 5.5108049000000001 * cse905 - 3.7209846 * cse908 + 4.6450521 * cse911 +
						      5.9369081000000001 * cse912 - 24.311209000000002);
	const amrex::Real cse922 = ((cse918) ? (cse917) : ((cse920) ? (cse919) : (cse921)));
	const amrex::Real cse923 = cse868 * std::pow(cse901, 8);
	const amrex::Real cse924 = std::pow(cse901, 7);
	const amrex::Real cse925 = cse870 * cse924;
	const amrex::Real cse926 = std::pow(cse901, 6);
	const amrex::Real cse927 = cse872 * cse926;
	const amrex::Real cse928 = std::pow(10.0, 983.67575999999997 * cse903 + 734.71650999999997 * cse905 + 96.743155000000002 * cse908 +
						      16.815729999999999 * cse911 + 343.1918 * cse912 + 70.609154000000004 * cse923 +
						      364.14445999999998 * cse925 + 801.81246999999996 * cse927 - 21.928795999999998);
	const amrex::Real cse929 = std::pow(10.0, -8.8077017000000009 * cse903 - 4.7274035999999997 * cse905 + 0.93310621999999999 * cse908 +
						      1.6802758 * cse911 + 4.0406627000000004 * cse912 - 6.3701156000000001 * cse923 +
						      6.4380698000000001 * cse925 + 8.9167182999999994 * cse927 - 22.921188999999998);
	const amrex::Real cse930 = ((cse759) ? (cse928) : (cse929));
	const amrex::Real cse931 = cse910 + cse914 + cse916 + cse922 * state.xn[0] + cse930 * state.xn[2];
	const amrex::Real cse932 = 1.0 / cse931;
	const amrex::Real cse933 = cse900 * cse932 + 1;
	const amrex::Real cse934 = 1.0 / cse933;
	const amrex::Real cse935 = ((cse891) ? (0.0) : (cse900 * cse934));
	const amrex::Real cse936 = 0.045999999999999999 * cse755;
	const amrex::Real cse937 = std::pow(cse754, cse936 + 0.22800000000000001);
	const amrex::Real cse938 = 8.9000000000000003e-11 * cse937;
	const amrex::Real cse939 = cse938 * state.xn[0];
	const amrex::Real cse940 = 0.037999999999999999 * cse755;
	const amrex::Real cse941 = 5.1749999999999996e-11 * std::pow(cse754, cse940 + 0.16900000000000001);
	const amrex::Real cse942 = 0.048000000000000001 * cse755;
	const amrex::Real cse943 = 2.15e-11 * std::pow(cse754, cse942 - 0.01);
	const amrex::Real cse944 = cse941 + cse943;
	const amrex::Real cse945 = 0.0070527699999999999 * cse160;
	const amrex::Real cse946 = cse945 - 0.11133800000000001;
	const amrex::Real cse947 = cse160 * cse946 + 0.69763799999999998;
	const amrex::Real cse948 = cse160 * cse947 - 1.3074300000000001;
	const amrex::Real cse949 = 0.093813800000000003 * cse160;
	const amrex::Real cse950 = cse949 - 3.0328300000000001;
	const amrex::Real cse951 = cse160 * cse950 + 36.180300000000003;
	const amrex::Real cse952 = cse160 * cse951 - 187.47399999999999;
	const amrex::Real cse953 = std::exp(((cse887) ? (cse160 * cse948 - 7.6973500000000001) : (cse160 * cse952 + 350.60899999999998)));
	const amrex::Real cse954 = 1.7258000000000001e-6 * cse753;
	const amrex::Real cse955 = cse953 * cse954;
	const amrex::Real cse956 = cse939 + cse944 * state.xn[3] + cse955;
	const amrex::Real cse957 = std::exp(-101.30000000000001 * cse4);
	const amrex::Real cse958 = cse956 * cse957;
	const amrex::Real cse959 = std::pow(cse754, cse936 + 0.23100000000000001);
	const amrex::Real cse960 = 2.6400000000000002e-10 * cse959;
	const amrex::Real cse961 = cse960 * state.xn[0];
	const amrex::Real cse962 = 0.031 * cse755;
	const amrex::Real cse963 = 1.11e-10 * std::pow(cse754, cse962 + 0.26300000000000001);
	const amrex::Real cse964 = 0.064000000000000001 * cse755;
	const amrex::Real cse965 = 4.3749999999999998e-11 * std::pow(cse754, cse964 + 0.071999999999999995);
	const amrex::Real cse966 = cse963 + cse965;
	const amrex::Real cse967 = 0.00235272 * cse160;
	const amrex::Real cse968 = cse967 - 0.041816600000000002;
	const amrex::Real cse969 = cse160 * cse968 + 0.35826400000000003;
	const amrex::Real cse970 = cse160 * cse969 - 0.57443;
	const amrex::Real cse971 = 0.097857299999999994 * cse160;
	const amrex::Real cse972 = cse971 - 3.1926800000000002;
	const amrex::Real cse973 = cse160 * cse972 + 38.504899999999999;
	const amrex::Real cse974 = cse160 * cse973 - 202.19300000000001;
	const amrex::Real cse975 = std::exp(((cse887) ? (cse160 * cse970 - 7.4386999999999999) : (cse160 * cse974 + 386.18599999999998)));
	const amrex::Real cse976 = cse954 * cse975;
	const amrex::Real cse977 = cse961 + cse966 * state.xn[3] + cse976;
	const amrex::Real cse978 = (25.0 / 3.0) * cse977;
	const amrex::Real cse979 = 0.057000000000000002 * cse755;
	const amrex::Real cse980 = std::pow(cse754, cse979 + 0.115);
	const amrex::Real cse981 = 1.26e-10 * cse980;
	const amrex::Real cse982 = cse981 * state.xn[0];
	const amrex::Real cse983 = 0.049000000000000002 * cse755;
	const amrex::Real cse984 = 5.3249999999999997e-11 * std::pow(cse754, cse983 - 0.0040000000000000001);
	const amrex::Real cse985 = 0.10199999999999999 * cse755;
	const amrex::Real cse986 = 1.675e-11 * std::pow(cse754, cse985 - 0.085000000000000006);
	const amrex::Real cse987 = cse984 + cse986;
	const amrex::Real cse988 = 0.00065632500000000003 * cse160;
	const amrex::Real cse989 = -cse988 - 0.015089200000000001;
	const amrex::Real cse990 = cse160 * cse989 + 0.36118400000000001;
	const amrex::Real cse991 = cse160 * cse990 - 0.77378199999999997;
	const amrex::Real cse992 = 0.10508000000000001 * cse160;
	const amrex::Real cse993 = cse992 - 3.4762;
	const amrex::Real cse994 = cse160 * cse993 + 42.594999999999999;
	const amrex::Real cse995 = cse160 * cse994 - 227.91300000000001;
	const amrex::Real cse996 = std::exp(((cse887) ? (cse160 * cse991 - 9.2514099999999999) : (cse160 * cse995 + 444.60000000000002)));
	const amrex::Real cse997 = 2.8763333333333333e-6 * cse996;
	const amrex::Real cse998 = cse753 * cse997;
	const amrex::Real cse999 = cse982 + cse987 * state.xn[3] + cse998;
	const amrex::Real cse1000 = std::exp(-62.460000000000001 * cse4);
	const amrex::Real cse1001 = 5 * cse1000;
	const amrex::Real cse1002 = cse1001 * cse999;
	const amrex::Real cse1003 = cse999 + 7.8800000000000004e-8;
	const amrex::Real cse1004 = cse1001 * cse956;
	const amrex::Real cse1005 = cse1002 * cse977 + cse1003 * cse1004 + cse958 * cse978;
	const amrex::Real cse1006 = cse977 + 2.65e-7;
	const amrex::Real cse1007 = cse956 + 1.81e-14;
	const amrex::Real cse1008 = std::exp(-38.840000000000003 * cse4);
	const amrex::Real cse1009 = cse1008 * cse977;
	const amrex::Real cse1010 = (5.0 / 3.0) * cse1009;
	const amrex::Real cse1011 = std::exp(-23.619999999999997 * cse4);
	const amrex::Real cse1012 = cse1011 * cse999;
	const amrex::Real cse1013 = 3 * cse1012;
	const amrex::Real cse1014 = cse1004 * cse1006 + cse1006 * cse1013 + cse1007 * cse1013;
	const amrex::Real cse1015 = cse1003 * cse1006 + cse1003 * cse1007 + cse1005 + cse1007 * cse1010 + cse1014;
	const amrex::Real cse1016 = 1.0 / cse1015;
	const amrex::Real cse1017 = 1.4210469458259591e-21 * cse1016;
	const amrex::Real cse1018 = 2.5697412351439998e-22 * cse1016;
	const amrex::Real cse1019 = state.T < 1000000.0;
	const amrex::Real cse1020 = ((cse1019) ? (cse1005 * cse1017 + cse1014 * cse1018) : (0.0));
	const amrex::Real cse1021 =
	    cse648 *
	    (1.7e-24 * chi * cse11 * cse155 * cse688 * cse692 * d2g + 5.6999999999999997e-11 * chi * cse272 * cse273 * cse752 * state.xn[3] +
	     1.602177e-12 * crate * cse686 * state.xn[0] + 3.2043540000000001e-12 * crate * cse751 * state.xn[3] - cse1020 * state.xn[7] +
	     3.0000000000000002e-15 * cse11 * cse745 * d2g * state.xn[0] - cse144 * cse716 - 7.1777529600000012e-12 * cse265 * cse31 - cse687 * state.xn[0] -
	     cse728 - cse769 * cse771 - cse775 * cse803 - cse804 * state.xn[0] - cse862 * state.xn[11] - cse935 * state.xn[3] - std::min(cse695, cse700));
	const amrex::Real cse1022 = 1.0 / cse647;
	const amrex::Real cse1023 = cse828 * cse844;
	const amrex::Real cse1024 = 7.1400000000000007e-11 * cse820;
	const amrex::Real cse1025 = cse816 * cse828;
	const amrex::Real cse1026 = 8.68e-11 * cse808;
	const amrex::Real cse1027 = cse827 * cse828;
	const amrex::Real cse1028 = cse831 * cse844;
	const amrex::Real cse1029 = 6.3800000000000002e-11 * cse836;
	const amrex::Real cse1030 = cse828 * cse846;
	const amrex::Real cse1031 = cse1023 * cse1024 + cse1024 * cse1025 + cse1026 * cse1027 + 2.8933333333333332e-11 * cse1028 * cse808 + cse1029 * cse1030 +
				    2.1266666666666666e-11 * cse832 * cse836;
	const amrex::Real cse1032 = cse828 * cse849;
	const amrex::Real cse1033 = 2.1419999999999999e-10 * cse820;
	const amrex::Real cse1034 = cse850 * cse854;
	const amrex::Real cse1035 = cse849 * cse854;
	const amrex::Real cse1036 = cse1023 * cse1026 + cse1029 * cse1032 + cse1033 * cse1034 + cse1033 * cse1035 + 2.604e-10 * cse808 * cse855 +
				    1.9139999999999998e-10 * cse836 * cse855;
	const amrex::Real cse1037 = std::pow(cse858, -2);
	const amrex::Real cse1038 = cse850 * cse851;
	const amrex::Real cse1039 = cse1037 * (-cse1031 - cse1036 - 1.4466666666666666e-10 * cse1038 * cse808 - cse809 * cse846 - cse821 * cse849 -
					       cse821 * cse850 - 1.0633333333333333e-10 * cse836 * cse852 - cse837 * cse846);
	const amrex::Real cse1040 = 2.3872025442750958e-19 * cse848;
	const amrex::Real cse1041 = 2.8023005140019997e-18 * cse857;
	const amrex::Real cse1042 = std::pow(cse931, -2);
	const amrex::Real cse1043 = std::pow(cse933, -2);
	const amrex::Real cse1044 = cse1042 * cse1043 * std::pow(cse900, 2);
	const amrex::Real cse1045 = cse957 * cse977;
	const amrex::Real cse1046 = cse1000 * cse999;
	const amrex::Real cse1047 = 1.32e-9 * cse959;
	const amrex::Real cse1048 = 6.3e-10 * cse980;
	const amrex::Real cse1049 = cse1000 * cse956;
	const amrex::Real cse1050 = cse1000 * cse977;
	const amrex::Real cse1051 = cse1000 * cse1003;
	const amrex::Real cse1052 = 4.4500000000000001e-10 * cse937;
	const amrex::Real cse1053 = 7.4166666666666679e-10 * cse1045 * cse937 + cse1046 * cse1047 + cse1048 * cse1049 + cse1048 * cse1050 + cse1051 * cse1052 +
				    2.2000000000000003e-9 * cse958 * cse959;
	const amrex::Real cse1054 = cse1000 * cse1006;
	const amrex::Real cse1055 = 3.7800000000000004e-10 * cse980;
	const amrex::Real cse1056 = cse1007 * cse1011;
	const amrex::Real cse1057 = cse1006 * cse1011;
	const amrex::Real cse1058 = 2.6700000000000001e-10 * cse1012 * cse937 + 7.9200000000000006e-10 * cse1012 * cse959 + cse1047 * cse1049 +
				    cse1052 * cse1054 + cse1055 * cse1056 + cse1055 * cse1057;
	const amrex::Real cse1059 = std::pow(cse1015, -2);
	const amrex::Real cse1060 = cse1007 * cse1008;
	const amrex::Real cse1061 = cse1059 * (-cse1003 * cse938 - cse1003 * cse960 - cse1006 * cse981 - cse1007 * cse981 -
					       1.4833333333333334e-10 * cse1009 * cse937 - cse1053 - cse1058 - 4.4000000000000003e-10 * cse1060 * cse959);
	const amrex::Real cse1062 = 1.4210469458259591e-21 * cse1005;
	const amrex::Real cse1063 = 2.5697412351439998e-22 * cse1014;
	const amrex::Real cse1064 = cse36 * ncol_H *
				    (((cse785 - 1.0000000000000001e-30 < 0) ? (0) : ((cse785 - 1.0000000000000001e-30 == 0) ? (1.0 / 2.0) : (1)))) /
				    std::pow(cse786, 3.0 / 2.0);
	const amrex::Real cse1065 = cse1064 * ncol_CO;
	const amrex::Real cse1066 = ((cse784) ? (0) : (878350727.21550155 * cse1065));
	const amrex::Real cse1067 = alpha_co_interp2d_partial_1(state.T, cse788);
	const amrex::Real cse1068 = std::log(cse795);
	const amrex::Real cse1069 = cse1067 * cse1068;
	const amrex::Real cse1070 = cse775 / std::pow(cse791, 2);
	const amrex::Real cse1071 = nhalf_co_interp2d_partial_1(state.T, cse788);
	const amrex::Real cse1072 = cse1066 * cse1071;
	const amrex::Real cse1073 = cse796 / cse775;
	const amrex::Real cse1074 = cse1073 * cse791;
	const amrex::Real cse1075 = std::pow(cse789, -2);
	const amrex::Real cse1076 = llte_co_interp2d_partial_1(state.T, cse788);
	const amrex::Real cse1077 = -cse1066 * cse1075 * cse1076 * cse775 + cse798 * (cse1066 * cse1075 * cse1076 * cse779 * cse791 - cse1072 * cse792);
	const amrex::Real cse1078 = cse775 * state.xn[9];
	const amrex::Real cse1079 = cse1078 * cse778 / std::pow(cse800, 2);
	const amrex::Real cse1080 = cse765 * cse769;
	const amrex::Real cse1081 = std::pow(cse743, -2);
	const amrex::Real cse1082 = cse36 * cse742;
	const amrex::Real cse1083 = cse37 * cse733;
	const amrex::Real cse1084 = cse39 * cse736;
	const amrex::Real cse1085 = cse1083 + cse1084;
	const amrex::Real cse1086 = cse741 / std::pow(cse739, 2);
	const amrex::Real cse1087 = cse1086 * cse12;
	const amrex::Real cse1088 = cse1082 - cse1087 * (cse1085 - cse737);
	const amrex::Real cse1089 = 1.5159999999999999e-9 * cse757;
	const amrex::Real cse1090 = std::pow(cse763, -2);
	const amrex::Real cse1091 = cse1090 * cse762 * cse764;
	const amrex::Real cse1092 = cse771 * state.xn[8] / std::pow(cse767, 2);
	const amrex::Real cse1093 = cse693 * d2g;
	const amrex::Real cse1094 = psigd_coll_interp2d_partial_0(cse11, state.T);
	const amrex::Real cse1095 = -cse695 + cse700;
	const amrex::Real cse1096 = (((cse1095 < 0) ? (0) : ((cse1095 == 0) ? (1.0 / 2.0) : (1))));
	const amrex::Real cse1097 = cse1096 * (100.0 * cse1093 + cse1094 * cse694);
	const amrex::Real cse1098 = cse12 * cse9;
	const amrex::Real cse1099 =
	    ((cse747) ? (0) : ((cse748) ? ((3.0 / 2.0) * cse1098) : ((cse749) ? ((4.0 / 3.0) * cse1098) : ((cse750) ? ((1.0 / 3.0) * cse1098) : (0)))));
	const amrex::Real cse1100 = d2g * state.xn[2];
	const amrex::Real cse1101 = cse1100 * cse726;
	const amrex::Real cse1102 = (((cse1095 > 0) ? (0) : ((cse1095 == 0) ? (1.0 / 2.0) : (1))));
	const amrex::Real cse1103 = cse1102 * cse696 * cse698 * d2g;
	const amrex::Real cse1104 =
	    cse1103 * (cse258 + cse266 + 2 * state.xn[1] + 2 * state.xn[10] + 2 * state.xn[14] + 2 * state.xn[15] + 6 * state.xn[16] + 4 * state.xn[4]);
	const amrex::Real cse1105 = cse36 * state.xn[2];
	const amrex::Real cse1106 = std::pow(cse683, -2);
	const amrex::Real cse1107 = 0.5 * cse1106;
	const amrex::Real cse1108 = cse226 / std::pow(cse11, 3);
	const amrex::Real cse1109 = -0.5 * cse1105 * cse684 + cse1107 * cse1108;
	const amrex::Real cse1110 = -1.7e-24 * chi * cse155 * cse688 * cse692 * d2g - 3.2043540000000001e-12 * crate * cse1099 * state.xn[3] -
				    4.22974728e-11 * crate * cse11 * cse1109 * cse114 * cse683 * cse685 * state.xn[0] + cse1097 + cse1101 + cse1104 -
				    3.0000000000000002e-15 * cse745 * d2g * state.xn[0];
	const amrex::Real cse1111 = cse121 * cse125;
	const amrex::Real cse1112 = 9.3874889891030969e-50 * cse384;
	const amrex::Real cse1113 = 9.3874889891030969e-50 * cse388;
	const amrex::Real cse1114 = 5.7787306591212543e-52 * cse174;
	const amrex::Real cse1115 = 2.6284969169488674e-52 * cse174;
	const amrex::Real cse1116 = 2.1501595383081364e-51 * cse174;
	const amrex::Real cse1117 = cse710 * std::pow(state.T, -1.8500000000000001);
	const amrex::Real cse1118 = cse144 * cse713;
	const amrex::Real cse1119 = std::pow(state.T, -0.84999999999999998);
	const amrex::Real cse1120 = std::pow(state.T, -0.19999999999999998);
	const amrex::Real cse1121 = std::pow(cse702, -2);
	const amrex::Real cse1122 = cse1119 * cse705;
	const amrex::Real cse1123 = std::pow(cse707, -2);
	const amrex::Real cse1124 = cse1123 * cse709 * (2.5500935934394033e-12 * cse1120 * cse1121 * state.xn[2] - 6.6690584756579352e-10 * cse1122);
	const amrex::Real cse1125 = 1.3659996292638161e-10 * cse704;
	const amrex::Real cse1126 = std::pow(cse691, -2);
	const amrex::Real cse1127 = cse156 * std::pow(cse689, 0.12299999999999994);
	const amrex::Real cse1128 = cse156 * std::pow(cse689, -0.56899999999999995) * cse690;
	const amrex::Real cse1129 = cse718 * cse89;
	const amrex::Real cse1130 = 1.0 / cse720;
	const amrex::Real cse1131 = cse1130 * cse160 / std::pow(cse721, 2);
	const amrex::Real cse1132 = cse1130 * cse724;
	const amrex::Real cse1133 = cse1125 * cse144 * cse710 / std::pow(cse712, 2);
	const amrex::Real cse1134 = cse68 * cse770;
	const amrex::Real cse1135 = cse4 * cse753;
	const amrex::Real cse1136 = std::pow(state.T, -0.92999999999999994);
	const amrex::Real cse1137 = 100 * cse4;
	const amrex::Real cse1138 = -2.2649999999999999e-6 * cse1135 + cse758 * (cse1137 * (8.7000000000000001e-5 * cse755 + 0.001281) + cse4 * cse756) +
				    state.xn[3] * (0.75 * ((cse759) ? (1.1000000000000001e-13) : (2.6233044951750002e-11 * cse1136)) +
						   0.25 * ((cse759) ? (3.3000000000000001e-13) : (2.7229810044920004e-11 * cse1136)));
	const amrex::Real cse1139 = 5.7927337833400002e-20 * cse1080;
	const amrex::Real cse1140 = 0.4375 * state.xn[0] / std::pow(state.T, 3.0 / 4.0) - 170.02500000000001 * state.xn[2] / std::pow(state.T, 5.0 / 4.0);
	const amrex::Real cse1141 = -cse1138;
	const amrex::Real cse1142 = 5.7927337833400002e-20 * cse1091 * cse769;
	const amrex::Real cse1143 = 2.0187430200000004e-26 * cse267;
	const amrex::Real cse1144 = std::pow(cse730, -3);
	const amrex::Real cse1145 = cse1081 * (-cse735 * cse8 * (0.0040460000000000001 * cse1144 + 0.0036709999999999998 * cse731) -
					       cse738 * cse8 * (0.0047800000000000004 * cse1144 - 5.5500000000000001e-5 * cse731));
	const amrex::Real cse1146 = 1.461185424e-21 * cse274 * state.xn[3];
	const amrex::Real cse1147 = 2 * cse1091;
	const amrex::Real cse1148 = l0_co_interp1d_partial_0(state.T);
	const amrex::Real cse1149 = std::pow(cse779, -2);
	const amrex::Real cse1150 =
	    ((cse784)
		 ? (-2.872276598720128e-13 * cse781 * cse89 * state.xn[9] *
		    (((5.7445531974402559e-18 * cse3 * cse781 - gradv < 0) ? (0)
									   : ((5.7445531974402559e-18 * cse3 * cse781 - gradv == 0) ? (1.0 / 2.0) : (1)))) /
		    std::pow(cse782, 2))
		 : (0));
	const amrex::Real cse1151 = -cse1076 * cse1150 - llte_co_interp2d_partial_0(state.T, cse788);
	const amrex::Real cse1152 = cse1071 * cse1150 + nhalf_co_interp2d_partial_0(state.T, cse788);
	const amrex::Real cse1153 = ((cse864) ? (-cse68) : (0));
	const amrex::Real cse1154 =
	    -1.8719999999999998e-14 * cse1153 * cse893 - 3.9261999999999998e-15 * cse1153 * cse894 - 1.53e-21 * cse1153 * cse895 +
	    4.985670872372847e-33 * cse896 * cse899 * ((cse864) ? (3.7599999999999998 * std::pow(state.T, 2.7599999999999998)) : (0)) -
	    1.0953518906603145e-26 * cse897 * cse899 * ((cse864) ? (-3 / std::pow(state.T, 4)) : (0)) -
	    2.9985055107607381e-40 * cse897 * ((cse864) ? (2.1000000000000001 * std::pow(state.T, 1.1000000000000001)) : (0)) / std::pow(cse898, 2);
	const amrex::Real cse1155 = ((cse864) ? (1) : (0));
	const amrex::Real cse1156 = cse1155 * cse892;
	const amrex::Real cse1157 = cse1156 * cse14 * cse901;
	const amrex::Real cse1158 = cse1156 * cse874 * cse904;
	const amrex::Real cse1159 = cse1156 * cse876 * cse906;
	const amrex::Real cse1160 = cse1156 * cse9;
	const amrex::Real cse1161 = cse1156 * cse878 * cse907;
	const amrex::Real cse1162 = cse1156 * cse868 * cse924;
	const amrex::Real cse1163 = cse1156 * cse870 * cse926;
	const amrex::Real cse1164 = cse1156 * cse872 * cse902;
	const amrex::Real cse1165 = 65.320000000000007 * cse1023;
	const amrex::Real cse1166 = cse4 * cse811;
	const amrex::Real cse1167 = 0.00042999999999999999 * cse755;
	const amrex::Real cse1168 = cse822 * (cse1137 * (0.0041900000000000001 - 3.0000000000000001e-5 * cse755) - cse4 * cse819) -
				    3.8399999999999998e-11 * std::pow(state.T, -1.075) * state.xn[2] +
				    state.xn[3] * (cse824 * (cse1137 * (0.00025000000000000001 * cse755 + 0.00264) + cse4 * cse823) +
						   cse825 * (cse1137 * (cse1167 + 0.00296) + cse1166));
	const amrex::Real cse1169 = cse816 * cse829;
	const amrex::Real cse1170 = cse838 * (cse1137 * (0.0036900000000000001 - 6.0000000000000002e-5 * cse755) - cse4 * cse835) -
				    1.2635999999999999e-11 * std::pow(state.T, -1.026) * state.xn[2] +
				    state.xn[3] * (cse840 * (cse1137 * (0.00041000000000000005 * cse755 + 0.0020300000000000001) + cse4 * cse839) +
						   cse842 * (cse1137 * (0.00058 * cse755 + 0.0023700000000000001) + cse4 * cse841));
	const amrex::Real cse1171 = (1.0 / 15.0) * cse832;
	const amrex::Real cse1172 = cse810 * (cse1137 * (0.0075500000000000003 - 0.0016000000000000001 * cse755) - cse4 * cse807) +
				    1.0000800000000001e-14 * std::pow(state.T, -0.073999999999999955) * state.xn[2] +
				    state.xn[3] * (cse812 * (cse1137 * (cse1167 + 0.0088900000000000003) + cse1166) +
						   cse814 * (cse1137 * (0.0052500000000000003 * cse755 + 0.011979999999999999) + cse4 * cse813));
	const amrex::Real cse1173 = cse831 * cse845;
	const amrex::Real cse1174 = cse1170 * cse846;
	const amrex::Real cse1175 = 65.320000000000007 * cse1027 * cse68 * cse816 + cse1165 * cse68 * cse846 + cse1168 * cse1169 + cse1168 * cse847 +
				    cse1170 * cse1171 + cse1172 * cse1173 + cse1172 * cse830 + cse1174 * cse829 + 28.360000000000003 * cse68 * cse832 * cse844;
	const amrex::Real cse1176 = 136.68000000000001 * cse855;
	const amrex::Real cse1177 = cse68 * cse850;
	const amrex::Real cse1178 = cse68 * cse849;
	const amrex::Real cse1179 = cse1168 * cse850;
	const amrex::Real cse1180 = (3.0 / 5.0) * cse854;
	const amrex::Real cse1181 = cse1168 * cse849;
	const amrex::Real cse1182 = cse829 * cse849;
	const amrex::Real cse1183 = cse1165 * cse1178 + cse1170 * cse1182 + cse1170 * cse856 + cse1172 * cse847 + cse1172 * cse856 + cse1176 * cse1177 +
				    cse1176 * cse1178 + cse1179 * cse1180 + cse1180 * cse1181;
	const amrex::Real cse1184 = (1.0 / 3.0) * cse1038;
	const amrex::Real cse1185 = cse1037 * (-cse1170 * cse853 - cse1172 * cse1184 - cse1172 * cse846 - cse1174 - cse1175 -
					       32.933333333333337 * cse1177 * cse852 - cse1179 - cse1181 - cse1183);
	const amrex::Real cse1186 = 312.30000000000001 * cse68;
	const amrex::Real cse1187 = cse4 * cse936;
	const amrex::Real cse1188 = 0.00046000000000000001 * cse755;
	const amrex::Real cse1189 = 8.6290000000000003e-7 * cse1135;
	const amrex::Real cse1190 = -cse1189 * cse975 + cse961 * (cse1137 * (cse1188 + 0.00231) + cse1187) +
				    cse976 * ((cse887) ? (cse160 * (cse160 * (cse4 * cse967 + cse4 * cse968) + cse4 * cse969) + cse4 * cse970)
						       : (cse160 * (cse160 * (cse4 * cse971 + cse4 * cse972) + cse4 * cse973) + cse4 * cse974)) +
				    state.xn[3] * (cse963 * (cse1137 * (0.00031 * cse755 + 0.00263) + cse4 * cse962) +
						   cse965 * (cse1137 * (0.00064000000000000005 * cse755 + 0.00071999999999999994) + cse4 * cse964));
	const amrex::Real cse1191 = (25.0 / 3.0) * cse958;
	const amrex::Real cse1192 = -1.4381666666666666e-6 * cse1135 * cse996 +
				    cse982 * (cse1137 * (0.00057000000000000009 * cse755 + 0.00115) + cse4 * cse979) +
				    cse998 * ((cse887) ? (cse160 * (cse160 * (-cse4 * cse988 + cse4 * cse989) + cse4 * cse990) + cse4 * cse991)
						       : (cse160 * (cse160 * (cse4 * cse992 + cse4 * cse993) + cse4 * cse994) + cse4 * cse995)) +
				    state.xn[3] * (cse984 * (cse1137 * (0.00048999999999999998 * cse755 - 4.0000000000000003e-5) + cse4 * cse983) +
						   cse986 * (cse1137 * (0.0010199999999999999 * cse755 - 0.00085000000000000006) + cse4 * cse985));
	const amrex::Real cse1193 = -cse1189 * cse953 + cse939 * (cse1137 * (cse1188 + 0.0022800000000000003) + cse1187) +
				    cse955 * ((cse887) ? (cse160 * (cse160 * (cse4 * cse945 + cse4 * cse946) + cse4 * cse947) + cse4 * cse948)
						       : (cse160 * (cse160 * (cse4 * cse949 + cse4 * cse950) + cse4 * cse951) + cse4 * cse952)) +
				    state.xn[3] * (cse941 * (cse1137 * (0.00038000000000000002 * cse755 + 0.0016900000000000001) + cse4 * cse940) +
						   cse943 * (cse1137 * (0.00048000000000000001 * cse755 - 0.0001) + cse4 * cse942));
	const amrex::Real cse1194 = cse957 * cse978;
	const amrex::Real cse1195 = cse1001 * cse977;
	const amrex::Real cse1196 = cse1003 * cse1193;
	const amrex::Real cse1197 = cse1001 * cse1196 + cse1002 * cse1190 + cse1004 * cse1192 + cse1046 * cse1186 * cse977 + cse1051 * cse1186 * cse956 +
				    cse1190 * cse1191 + cse1192 * cse1195 + cse1193 * cse1194 + 844.16666666666686 * cse68 * cse958 * cse977;
	const amrex::Real cse1198 = 70.859999999999985 * cse1012 * cse68;
	const amrex::Real cse1199 = cse1007 * cse1192;
	const amrex::Real cse1200 = 3 * cse1011;
	const amrex::Real cse1201 = cse1001 * cse1006;
	const amrex::Real cse1202 = cse1006 * cse1192;
	const amrex::Real cse1203 = cse1004 * cse1190 + cse1006 * cse1049 * cse1186 + cse1006 * cse1198 + cse1007 * cse1198 + cse1013 * cse1190 +
				    cse1013 * cse1193 + cse1193 * cse1201 + cse1199 * cse1200 + cse1200 * cse1202;
	const amrex::Real cse1204 = (5.0 / 3.0) * cse1060;
	const amrex::Real cse1205 = cse1059 * (-cse1003 * cse1190 - 64.733333333333348 * cse1007 * cse1009 * cse68 - cse1010 * cse1193 - cse1190 * cse1204 -
					       cse1196 - cse1197 - cse1199 - cse1202 - cse1203);
	const amrex::Real cse1206 =
	    cse192 *
	    (cse1022 *
		 (1.7e-24 * chi * cse11 * cse1126 * cse155 * cse688 * d2g *
		      (-0.00012636182959999999 * cse1127 * cse115 - 0.018302846000000001 * cse1128 * cse115) +
		  5.6227499999999995e-25 * chi * cse11 * cse155 * cse692 * d2g * std::pow(state.T, -0.85299999999999998) -
		  1000 * cse1078 * cse776 * std::pow(cse777, 999) * cse801 -
		  cse1079 * (-cse1075 * cse1151 * cse775 - cse1140 * cse790 + cse1148 * cse1149 * cse793 * cse797 + cse1148 * cse1149 -
			     cse798 * (-cse1075 * cse1151 * cse779 * cse791 - cse1148 * cse790 * cse791 - cse1152 * cse792) -
			     cse799 * (cse1068 * (cse1067 * cse1150 + alpha_co_interp2d_partial_0(state.T, cse788)) +
				       cse1074 * (-cse1070 * cse1152 + cse1140 * cse794))) -
		  cse1086 * cse1143 * cse1145 - cse1087 * cse1145 * cse1146 - cse1092 * (-182.41999999999999 * cse1134 - cse1138 * cse766 - cse1141 * cse1147) -
		  cse1096 * cse694 * psigd_coll_interp2d_partial_1(cse11, state.T) - cse1102 * (1.6000000000000001e-32 * cse696 * cse697 * cse89 + cse699) -
		  1.6173435610483583e-5 * cse1117 * cse1118 * cse703 - cse1118 * cse1124 * cse1125 -
		  2.0489994438957239e-11 * cse1119 * cse144 * cse703 * cse714 +
		  7.8348995947646122e-14 * cse1120 * cse1121 * cse708 * cse709 * cse713 * state.xn[0] * state.xn[2] -
		  cse1133 * (-0.0015792330470357991 * cse1117 * cse705 + 7.6502807803182096e-12 * cse1120 * cse1121 * cse708 * cse709 * state.xn[2] -
			     2.0007175426973804e-9 * cse1122 * cse710 - 1.3338116951315871e-8 * cse1124 * cse706) -
		  5.2835524837844135e-18 * cse1134 * cse769 - cse1138 * cse1139 - cse1140 * cse803 - cse1141 * cse1142 - 2.1789607199999999e-11 * cse145 -
		  7.1777529600000012e-12 * cse562 - 7.1777529600000012e-12 * cse563 - cse728 * (-1.8989 * cse1129 * cse1131 + cse4 * cse722) -
		  cse728 * (-0.043256499999999996 * cse1129 * cse1130 * cse721 + 0.84999999999999998 * cse1129 * cse1132) -
		  state.xn[11] * (cse1040 * cse1185 + cse1041 * cse1185 + cse1175 * cse860 + cse1183 * cse861) -
		  state.xn[3] *
		      ((cse891)
			   ? (0)
			   : (cse1043 * cse900 *
				  (-cse1042 * cse900 *
				       (-cse910 * (3.0182298984217999 * cse1155 * cse878 * cse892 * cse907 + 4.8223900769399082 * cse1155 * cse892 * cse9 -
						   3.5529549287336835 * cse1157 - 0.3872755400043702 * cse1158 - 1.3735579540080118 * cse1159) -
					cse914 * (0.0708789387907936 * cse1157 + 3.7035619079275177 * cse1158 - 4.6974781513675152 * cse1159 +
						  3.6184459289309552 * cse1160 - 1.6316107607322889 * cse1161) -
					cse916 * (-3.7541549062629067 * cse1157 + 2.2094886994529301 * cse1158 - 1.528565035159452 * cse1159 +
						  5.0409049417480247 * cse1160 + 2.0057552335975877 * cse1161) -
					state.xn[0] * ((cse918) ? (cse8 * cse917 *
								   (116.29033200000001 * cse1157 + 19.239805 * cse1158 + 80.639324000000002 * cse1159 +
								    37.383713 * cse1160 + 145.968309 * cse1161))
								: ((cse920) ? (cse919 * (8.2184944748967013 * cse1155 * cse892 * cse9 -
											 52.189748993977005 * cse1157 - 48.951834264235494 * cse1158 -
											 196.44057098336626 * cse1159 - 192.38155095558542 * cse1161))
									    : (cse921 * (-17.135767342440825 * cse1157 + 17.889115159724135 * cse1158 -
											 50.756388852554174 * cse1159 + 10.695627721640689 * cse1160 +
											 41.010708268606813 * cse1161)))) -
					state.xn[2] *
					    ((cse759) ? (cse928 * (445.51869310442481 * cse1157 + 11324.985706577943 * cse1158 + 6766.9891340104432 * cse1159 +
								   38.719649225812766 * cse1160 + 2370.6849681533818 * cse1161 + 1300.6686834345674 * cse1162 +
								   5869.3152370465659 * cse1163 + 11077.448644792414 * cse1164))
						      : (cse929 * (4.297112944704045 * cse1157 - 101.40241318979159 * cse1158 - 43.540996231705549 * cse1159 +
								   3.8689780091986448 * cse1160 + 27.91190909651122 * cse1161 - 117.34186576967058 * cse1162 +
								   103.76942484394611 * cse1163 + 123.18901581604325 * cse1164)))) -
				   cse1154 * cse932) +
			      cse1154 * cse934)) -
		  state.xn[7] * ((cse1019) ? (cse1017 * cse1197 + cse1018 * cse1203 + cse1062 * cse1205 + cse1063 * cse1205) : (0))) -
	     cse650 * cse680 -
	     cse651 * (-1.3873860363782363e-53 * cse104 + 2.7153919503994842e-50 * cse106 + 2.2860865638527305e-52 * cse1111 * cse122 +
		       2.6187736442741324e-52 * cse1111 + cse1112 * cse382 - cse1113 * cse382 + cse1114 * cse173 + cse1115 * cse444 + cse1116 * cse458 +
		       6.6709304409164816e-37 * cse112 - 2.2860865638527305e-52 * cse121 * cse178 - 1.7739198567658306e-44 * cse127 +
		       9.901755811290946e-50 * cse131 + 3.2129591128082713e-50 * cse132 - 4.7173311503030642e-41 * cse145 - 3.4540468482946404e-46 * cse151 +
		       2.4124578152789816e-44 * cse153 + 1.2105577531005002e-49 * cse186 + 3.6064282614712017e-53 * cse189 + 4.7173311503030642e-41 * cse371 -
		       1.7836229079295885e-50 * cse380 + 4.7173311503030642e-41 * cse404 + 2.6165461020725406e-49 * cse405 - 2.3586655751515319e-52 * cse412 +
		       2.3097333986576313e-51 * cse414 + 4.7173311503030638e-52 * cse417 - cse430 * cse677 + 3.5250030316107808e-47 * cse615 -
		       2.6879059887952663e-51 * cse72 - 4.5507211268745734e-47 * cse76 - 2.9387358770557188e-39 * cse88 + 3.7823358381149801e-46 * cse93 +
		       2.9387358770557188e-39 * cse95 - 1.3055014861977917e-52 * cse99));
	const amrex::Real cse1207 = 2.3026978988630935e-46 * cse149;
	const amrex::Real cse1208 = 4.7173311503030642e-41 * cse212;
	const amrex::Real cse1209 = cse211 * cse643;
	const amrex::Real cse1210 = -cse1208 - cse1209 + 5.7787306591212543e-52 * cse252 + 4.9060243963151868e-38 * cse471 + 2.1501595383081364e-51 * cse511 +
				    2.6284969169488674e-52 * cse521 - cse644 - cse645;
	const amrex::Real cse1211 = 7.1777529600000012e-12 * cse219;
	const amrex::Real cse1212 = 7.1777529600000012e-12 * cse569;
	const amrex::Real cse1213 = cse794 * cse796;
	const amrex::Real cse1214 = cse1079 * (-cse1077 - cse799 * (cse1066 * cse1067 * cse1068 - cse1072 * cse1213));
	const amrex::Real cse1215 = -cse1085 * cse1087 + cse36 * cse740 * cse741;
	const amrex::Real cse1216 = 2.7717497761966105e-44 * cse241;
	const amrex::Real cse1217 = 4.7173311503030638e-52 * cse482;
	const amrex::Real cse1218 = 7.2128565229424034e-53 * cse224;
	const amrex::Real cse1219 = 5.2375472885482647e-52 * cse125;
	const amrex::Real cse1220 = 4.572173127705461e-52 * cse178;
	const amrex::Real cse1221 = 4.572173127705461e-52 * cse122 * cse125;
	const amrex::Real cse1222 = std::pow(state.T, 0.29999999999999999);
	const amrex::Real cse1223 = cse723 * cse725;
	const amrex::Real cse1224 = std::pow(cse717, -2);
	const amrex::Real cse1225 = cse1224 * cse156;
	const amrex::Real cse1226 = 9.0599999999999997e-6 * cse74;
	const amrex::Real cse1227 = 1.024e-10 * cse817;
	const amrex::Real cse1228 = 2.1600000000000004e-15 * cse805;
	const amrex::Real cse1229 = 9.7199999999999998e-11 * cse833;
	const amrex::Real cse1230 = cse1023 * cse1227 + cse1025 * cse1227 + cse1027 * cse1228 + 7.1999999999999997e-16 * cse1028 * cse805 + cse1030 * cse1229 +
				    3.2399999999999999e-11 * cse832 * cse833;
	const amrex::Real cse1231 = 3.0719999999999998e-10 * cse817;
	const amrex::Real cse1232 = cse1023 * cse1228 + cse1032 * cse1229 + cse1034 * cse1231 + cse1035 * cse1231 + 6.4799999999999996e-15 * cse805 * cse855 +
				    2.9159999999999995e-10 * cse833 * cse855;
	const amrex::Real cse1233 = cse1037 * (-3.6000000000000001e-15 * cse1038 * cse805 - cse1230 - cse1232 - cse806 * cse846 - cse818 * cse849 -
					       cse818 * cse850 - 1.6199999999999998e-10 * cse833 * cse852 - cse834 * cse846);
	const amrex::Real cse1234 = cse74 * cse975;
	const amrex::Real cse1235 = cse74 * cse996;
	const amrex::Real cse1236 = 1.4381666666666666e-5 * cse1235;
	const amrex::Real cse1237 = cse74 * cse953;
	const amrex::Real cse1238 = 8.6289999999999999e-6 * cse1234;
	const amrex::Real cse1239 = 8.6289999999999999e-6 * cse1237;
	const amrex::Real cse1240 = 1.4381666666666668e-5 * cse1045 * cse1237 + cse1046 * cse1238 + cse1049 * cse1236 + cse1050 * cse1236 + cse1051 * cse1239 +
				    1.4381666666666668e-5 * cse1234 * cse958;
	const amrex::Real cse1241 = 5.1774000000000006e-6 * cse1012;
	const amrex::Real cse1242 = 8.6289999999999999e-6 * cse1235;
	const amrex::Real cse1243 = cse1049 * cse1238 + cse1054 * cse1239 + cse1056 * cse1242 + cse1057 * cse1242 + cse1234 * cse1241 + cse1237 * cse1241;
	const amrex::Real cse1244 = cse74 * cse997;
	const amrex::Real cse1245 = 1.7258000000000001e-6 * cse1003;
	const amrex::Real cse1246 = cse1059 * (-cse1006 * cse1244 - cse1007 * cse1244 - 2.8763333333333337e-6 * cse1009 * cse1237 -
					       2.8763333333333337e-6 * cse1060 * cse1234 - cse1234 * cse1245 - cse1237 * cse1245 - cse1240 - cse1243);
	const amrex::Real cse1247 = 4.8578251101285161e-35 * cse503;
	const amrex::Real cse1248 = 1.0042319124598398e-52 * cse276;
	const amrex::Real cse1249 = 6.465218929522581e-49 * cse102;
	const amrex::Real cse1250 = 1.1557461318242509e-51 * cse252 + 9.8120487926303735e-38 * cse471 - 9.7156502202570321e-35 * cse477 -
				    1.2526249033974244e-50 * cse505 + 4.3003190766162728e-51 * cse511 + 5.2569938338977348e-52 * cse521;
	const amrex::Real cse1251 = cse1100 * cse1223;
	const amrex::Real cse1252 = 3.7996253111508318e-29 * cse1251;
	const amrex::Real cse1253 = cse1103 * (4 * state.xn[0] + 4 * state.xn[1] + 4 * state.xn[10] + 4 * state.xn[14] + 4 * state.xn[15] + 12 * state.xn[16] +
					       8 * state.xn[3] + 8 * state.xn[4]);
	const amrex::Real cse1254 = cse1094 * cse174;
	const amrex::Real cse1255 = cse1096 * (200.0 * cse1093 + 200.0 * cse1254);
	const amrex::Real cse1256 = cse1105 * cse684;
	const amrex::Real cse1257 = cse1106 * cse1108;
	const amrex::Real cse1258 = -1.0 * cse1256 + 1.0 * cse1257;
	const amrex::Real cse1259 =
	    ((cse747) ? (0) : ((cse748) ? (3 * cse1098) : ((cse749) ? ((8.0 / 3.0) * cse1098) : ((cse750) ? ((2.0 / 3.0) * cse1098) : (0)))));
	const amrex::Real cse1260 = -cse761;
	const amrex::Real cse1261 = cse257 * cse733 + cse259 * cse736;
	const amrex::Real cse1262 = 2 * cse1082 - cse1087 * (cse1261 - cse734);
	const amrex::Real cse1263 = ((cse784) ? (0) : (cse1064 * cse787));
	const amrex::Real cse1264 = cse1071 * cse1263;
	const amrex::Real cse1265 = -cse1075 * cse1076 * cse1263 * cse775 + cse798 * (cse1075 * cse1076 * cse1263 * cse779 * cse791 - cse1264 * cse792);
	const amrex::Real cse1266 = cse843 * cse846;
	const amrex::Real cse1267 = cse1169 * cse826 + cse1171 * cse843 + cse1173 * cse815 + cse1266 * cse829 + cse815 * cse830 + cse826 * cse847;
	const amrex::Real cse1268 = cse826 * cse850;
	const amrex::Real cse1269 = cse826 * cse849;
	const amrex::Real cse1270 = cse1180 * cse1268 + cse1180 * cse1269 + cse1182 * cse843 + cse815 * cse847 + cse815 * cse856 + cse843 * cse856;
	const amrex::Real cse1271 = cse1037 * (-cse1184 * cse815 - cse1266 - cse1267 - cse1268 - cse1269 - cse1270 - cse815 * cse846 - cse843 * cse853);
	const amrex::Real cse1272 = cse1003 * cse944;
	const amrex::Real cse1273 = cse1001 * cse1272 + cse1002 * cse966 + cse1004 * cse987 + cse1191 * cse966 + cse1194 * cse944 + cse1195 * cse987;
	const amrex::Real cse1274 = cse1007 * cse987;
	const amrex::Real cse1275 = cse1006 * cse987;
	const amrex::Real cse1276 = cse1004 * cse966 + cse1013 * cse944 + cse1013 * cse966 + cse1200 * cse1274 + cse1200 * cse1275 + cse1201 * cse944;
	const amrex::Real cse1277 = cse1059 * (-cse1003 * cse966 - cse1010 * cse944 - cse1204 * cse966 - cse1272 - cse1273 - cse1274 - cse1275 - cse1276);
	const amrex::Real cse1278 = cse156 * cse692 * d2g;
	const amrex::Real cse1279 = cse1278 * cse688;
	const amrex::Real cse1280 = 4.22974728e-11 * cse11 * cse114 * cse202 * cse683 * cse685;
	const amrex::Real cse1281 = 3.2043540000000001e-12 * crate * state.xn[3];
	const amrex::Real cse1282 = -cse1087 * cse1261 + 2 * cse36 * cse740 * cse741;
	const amrex::Real cse1283 = cse1081 * cse11 * cse1143;
	const amrex::Real cse1284 = cse1081 * cse1146;
	const amrex::Real cse1285 = cse1022 * state.xn[3] * ((cse891) ? (0) : (cse1044 * cse915)) + cse681;
	const amrex::Real cse1286 = ((cse784) ? (cse783) : (0));
	const amrex::Real cse1287 = cse1071 * cse1286;
	const amrex::Real cse1288 = -cse1022 * (-cse1097 + cse1099 * cse1281 - cse1101 - cse1104 + cse1109 * cse1280 - cse1211 - cse1212 - cse1214 +
						cse1215 * cse1283 + cse1215 * cse1284 + 1.7e-24 * cse1278 * cse688 + 3.0000000000000002e-15 * cse267 * cse745) +
				    cse681;
	const amrex::Real cse1289 = cse1208 + cse1209 + cse646;
	const amrex::Real cse1290 = -cse1087 * (3 * cse1083 + 3 * cse1084) + 3 * cse36 * cse740 * cse741;
	const amrex::Real cse1291 = ((cse784) ? (0) : (2635052181.6465044 * cse1065));
	const amrex::Real cse1292 = cse1071 * cse1291;

	jac(1, 1) =
	    -cse11 * cse57 - cse193 * cse67 - cse2 - cse208 + 2 * cse30 * state.xn[3] + 2 * cse31 * cse56 - cse58 + 2 * cse61 * state.xn[0] * state.xn[3];
	jac(1, 2) = -cse193 * cse209 + cse215 + cse223;
	jac(1, 3) = -cse193 * cse232 + cse229 + cse231 * state.xn[16] + cse234 * cse235 + cse240 + cse247;
	jac(1, 4) = 3.2000000000000001e-9 * cse177 * state.xn[17] - cse193 * cse278 + cse222 * cse270 - cse235 * cse279 + cse249 + cse251 + cse253 +
		    2 * cse254 + 2 * cse264 + cse265 * cse266 - cse268 + 1.1399999999999999e-10 * cse274 + cse277 * state.xn[8] + cse283 + cse289;
	jac(1, 5) = -cse193 * cse296 - cse268 + 2 * cse293 * cse31 + 2 * cse294 * state.xn[0] * state.xn[3] + cse295 - cse301;
	jac(1, 6) = -cse193 * cse302;
	jac(1, 7) = -cse193 * cse303 + cse308;
	jac(1, 8) = -cse193 * cse312 + cse311;
	jac(1, 9) = -cse193 * cse313 + cse277 * state.xn[3];
	jac(1, 10) = -cse193 * cse315 - cse314;
	jac(1, 11) = -cse193 * cse316 + cse223 + cse318;
	jac(1, 12) = -cse193 * cse324 + cse322 + cse323 + cse327;
	jac(1, 13) = -cse193 * cse328;
	jac(1, 14) = -cse193 * cse329;
	jac(1, 15) = -cse193 * cse335 + cse330 + cse332 + cse334 + cse336;
	jac(1, 16) = -cse193 * cse337 + cse336 + cse342;
	jac(1, 17) = -cse193 * cse349 + cse222 * cse348 + cse231 * state.xn[2] - 1.8000000000000002e-14 * cse267 + 2 * cse347 + cse354;
	jac(1, 18) = 3.2000000000000001e-9 * cse177 * state.xn[3] - cse193 * cse358 - cse355 + cse357;
	jac(1, 19) = cse193;
	jac(2, 1) = cse208 - cse360 * cse67;
	jac(2, 2) = -cse209 * cse360 - cse215;
	jac(2, 3) = -cse232 * cse360 - cse246;
	jac(2, 4) = cse249 - cse278 * cse360 + cse286 + cse297;
	jac(2, 5) = -cse296 * cse360 + cse301;
	jac(2, 6) = -cse302 * cse360;
	jac(2, 7) = -cse303 * cse360 + cse304;
	jac(2, 8) = -cse312 * cse360;
	jac(2, 9) = -cse313 * cse360;
	jac(2, 10) = -cse315 * cse360;
	jac(2, 11) = -cse213 - cse316 * cse360;
	jac(2, 12) = -cse324 * cse360 - cse325;
	jac(2, 13) = -cse328 * cse360;
	jac(2, 14) = -cse329 * cse360;
	jac(2, 15) = -cse213 - cse335 * cse360;
	jac(2, 16) = -cse213 - cse337 * cse360;
	jac(2, 17) = -cse349 * cse360 - cse353;
	jac(2, 18) = cse355 - cse358 * cse360;
	jac(2, 19) = cse360;
	jac(3, 1) = cse197 + cse199 + cse203 - cse207 - cse363 - cse461 * cse67 - cse479;
	jac(3, 2) = -cse209 * cse461 - cse214 - cse481;
	jac(3, 3) = -cse232 * cse461 - cse247 - cse485 - cse489 - cse493 - cse494 - cse500;
	jac(3, 4) = -cse278 * cse461 - cse289 + cse501 + cse502 - cse507 - cse514 - cse519 - cse520 - cse523;
	jac(3, 5) = -cse296 * cse461 + cse300 - cse506 - cse512 - cse523 - cse525;
	jac(3, 6) = -cse302 * cse461 + cse526;
	jac(3, 7) = -cse303 * cse461 - cse527;
	jac(3, 8) = -cse312 * cse461 - cse529 + cse532;
	jac(3, 9) = -cse313 * cse461 - cse536;
	jac(3, 10) = cse314 - cse315 * cse461;
	jac(3, 11) = -cse316 * cse461 - cse318 - cse481;
	jac(3, 12) = -cse324 * cse461 - cse539;
	jac(3, 13) = -cse328 * cse461 + cse542;
	jac(3, 14) = -cse329 * cse461 - cse543;
	jac(3, 15) = -cse335 * cse461 - cse544;
	jac(3, 16) = -cse337 * cse461 - cse544;
	jac(3, 17) = -cse349 * cse461 - cse354 - cse545 - cse546 - cse547 - cse548 - cse550 - cse553 - cse554;
	jac(3, 18) = -cse358 * cse461 - cse555;
	jac(3, 19) = cse461;
	jac(4, 1) = 3.0000000000000002e-15 * cse11 * d2g + cse196 + cse2 - cse362 - cse556 - cse557 - cse559 - cse560 - cse568 * cse67;
	jac(4, 2) = -cse209 * cse568 - cse570;
	jac(4, 3) = cse229 - cse232 * cse568 + cse571 * state.xn[16] + cse572 + cse573;
	jac(4, 4) = -cse10 * cse265 - cse220 - cse254 - cse264 - cse278 * cse568 - cse509 - cse574 - cse575 - cse577 - cse578 - cse579 - cse580;
	jac(4, 5) = -cse220 - cse295 - cse296 * cse568 + cse298 - cse524 - cse581 - cse582;
	jac(4, 6) = -cse302 * cse568;
	jac(4, 7) = -cse303 * cse568 - cse304 - cse583;
	jac(4, 8) = -cse312 * cse568 + cse529;
	jac(4, 9) = -cse313 * cse568 - cse533;
	jac(4, 10) = -cse315 * cse568 + cse584;
	jac(4, 11) = -cse316 * cse568 - cse570;
	jac(4, 12) = -cse324 * cse568 + cse539;
	jac(4, 13) = -cse328 * cse568;
	jac(4, 14) = -cse329 * cse568;
	jac(4, 15) = -cse334 - cse335 * cse568 - cse570;
	jac(4, 16) = -cse337 * cse568 - cse570;
	jac(4, 17) = -cse347 - cse349 * cse568 + cse550 + cse571 * state.xn[2] + cse585 - cse586 + cse588;
	jac(4, 18) = -cse358 * cse568 - cse555;
	jac(4, 19) = cse568;
	jac(5, 1) = -cse196 - cse363 - cse589 * cse67;
	jac(5, 2) = -cse209 * cse589 + cse480;
	jac(5, 3) = -cse232 * cse589;
	jac(5, 4) = -cse251 - cse278 * cse589 + cse576 + cse578;
	jac(5, 5) = -cse295 - cse296 * cse589 - cse298 - cse525;
	jac(5, 6) = -cse302 * cse589;
	jac(5, 7) = -cse303 * cse589 + cse583;
	jac(5, 8) = -cse312 * cse589;
	jac(5, 9) = -cse313 * cse589;
	jac(5, 10) = -cse315 * cse589;
	jac(5, 11) = -cse316 * cse589 + cse480;
	jac(5, 12) = -cse324 * cse589;
	jac(5, 13) = -cse328 * cse589;
	jac(5, 14) = -cse329 * cse589;
	jac(5, 15) = -cse335 * cse589 + cse480;
	jac(5, 16) = -cse337 * cse589 + cse480;
	jac(5, 17) = -cse349 * cse589 + cse549;
	jac(5, 18) = -cse358 * cse589;
	jac(5, 19) = cse589;
	jac(6, 1) = cse465 - cse592 * cse67;
	jac(6, 2) = -cse209 * cse592 + cse465;
	jac(6, 3) = -cse232 * cse592 + cse485;
	jac(6, 4) = -cse278 * cse592 + cse593;
	jac(6, 5) = -cse296 * cse592 + cse522;
	jac(6, 6) = -cse302 * cse592 - cse526;
	jac(6, 7) = -cse303 * cse592 + cse595;
	jac(6, 8) = -cse312 * cse592;
	jac(6, 9) = -cse313 * cse592;
	jac(6, 10) = -cse315 * cse592 + cse596;
	jac(6, 11) = -cse316 * cse592 + cse465;
	jac(6, 12) = -cse324 * cse592;
	jac(6, 13) = -cse328 * cse592;
	jac(6, 14) = -cse329 * cse592;
	jac(6, 15) = -cse335 * cse592 + cse465;
	jac(6, 16) = -cse337 * cse592 + cse597;
	jac(6, 17) = -cse349 * cse592 + cse545;
	jac(6, 18) = -cse358 * cse592;
	jac(6, 19) = cse592;
	jac(7, 1) = -cse465 - cse598 * cse67;
	jac(7, 2) = -cse209 * cse598 - cse465;
	jac(7, 3) = -cse232 * cse598 - cse485;
	jac(7, 4) = -cse278 * cse598 - cse593;
	jac(7, 5) = -cse296 * cse598 - cse522;
	jac(7, 6) = -cse302 * cse598 + cse526;
	jac(7, 7) = -cse303 * cse598 - cse595;
	jac(7, 8) = -cse312 * cse598;
	jac(7, 9) = -cse313 * cse598;
	jac(7, 10) = -cse315 * cse598 - cse596;
	jac(7, 11) = -cse316 * cse598 - cse465;
	jac(7, 12) = -cse324 * cse598;
	jac(7, 13) = -cse328 * cse598;
	jac(7, 14) = -cse329 * cse598;
	jac(7, 15) = -cse335 * cse598 - cse465;
	jac(7, 16) = -cse337 * cse598 - cse597;
	jac(7, 17) = -cse349 * cse598 - cse545;
	jac(7, 18) = -cse358 * cse598;
	jac(7, 19) = cse598;
	jac(8, 1) = cse2 - cse600 * cse67 + cse604;
	jac(8, 2) = -cse209 * cse600 + cse604;
	jac(8, 3) = -cse232 * cse600 + cse489;
	jac(8, 4) = -cse278 * cse600 + cse513 + cse605 * state.xn[8] + cse610;
	jac(8, 5) = -cse296 * cse600 + cse512 + cse609;
	jac(8, 6) = -cse302 * cse600;
	jac(8, 7) = -cse303 * cse600;
	jac(8, 8) = -cse311 - cse312 * cse600 - cse529 - cse532;
	jac(8, 9) = -cse313 * cse600 + cse535 + cse605 * state.xn[3];
	jac(8, 10) = -cse315 * cse600 + cse611;
	jac(8, 11) = -cse316 * cse600 + cse604;
	jac(8, 12) = -cse324 * cse600;
	jac(8, 13) = -cse328 * cse600;
	jac(8, 14) = -cse329 * cse600;
	jac(8, 15) = -cse335 * cse600 + cse604 + cse612;
	jac(8, 16) = -cse337 * cse600 + cse469 + cse472 - cse613;
	jac(8, 17) = -cse349 * cse600 - cse546 + cse551 + cse552 - cse614;
	jac(8, 18) = -cse358 * cse600;
	jac(8, 19) = cse600;
	jac(9, 1) = -cse473 - cse618 * cse67;
	jac(9, 2) = -cse209 * cse618 - cse473;
	jac(9, 3) = -cse232 * cse618 - cse489;
	jac(9, 4) = -cse278 * cse618 - cse514;
	jac(9, 5) = -cse296 * cse618 - cse512;
	jac(9, 6) = -cse302 * cse618;
	jac(9, 7) = -cse303 * cse618 + cse594;
	jac(9, 8) = -cse312 * cse618 + cse532;
	jac(9, 9) = -cse313 * cse618 - cse536 - cse620;
	jac(9, 10) = -cse315 * cse618 + cse596;
	jac(9, 11) = -cse316 * cse618 - cse473;
	jac(9, 12) = -cse324 * cse618;
	jac(9, 13) = -cse328 * cse618;
	jac(9, 14) = -cse329 * cse618;
	jac(9, 15) = -cse335 * cse618 - cse473;
	jac(9, 16) = -cse337 * cse618 - cse473 - cse621;
	jac(9, 17) = -cse349 * cse618 - cse553;
	jac(9, 18) = -cse358 * cse618;
	jac(9, 19) = cse618;
	jac(10, 1) = cse602 - cse622 * cse67;
	jac(10, 2) = -cse209 * cse622 + cse602;
	jac(10, 3) = -cse232 * cse622 + cse243;
	jac(10, 4) = -cse278 * cse622 - cse610;
	jac(10, 5) = -cse296 * cse622 + cse608;
	jac(10, 6) = -cse302 * cse622;
	jac(10, 7) = -cse303 * cse622 - cse594;
	jac(10, 8) = cse311 - cse312 * cse622;
	jac(10, 9) = -cse313 * cse622;
	jac(10, 10) = -cse315 * cse622 - cse623 - cse624;
	jac(10, 11) = -cse316 * cse622 + cse317 + cse602;
	jac(10, 12) = -cse324 * cse622 + cse326;
	jac(10, 13) = -cse328 * cse622;
	jac(10, 14) = -cse329 * cse622;
	jac(10, 15) = -cse335 * cse622 + cse625;
	jac(10, 16) = -cse337 * cse622 + cse613;
	jac(10, 17) = -cse349 * cse622 - cse587 + cse614;
	jac(10, 18) = -cse358 * cse622;
	jac(10, 19) = cse622;
	jac(11, 1) = -cse626 * cse67;
	jac(11, 2) = -cse209 * cse626;
	jac(11, 3) = -cse232 * cse626 - cse243;
	jac(11, 4) = -cse278 * cse626;
	jac(11, 5) = -cse296 * cse626;
	jac(11, 6) = -cse302 * cse626;
	jac(11, 7) = -cse303 * cse626;
	jac(11, 8) = -cse312 * cse626;
	jac(11, 9) = -cse313 * cse626 + cse620;
	jac(11, 10) = -cse315 * cse626 + cse623;
	jac(11, 11) = -cse316 * cse626 - cse317;
	jac(11, 12) = -cse324 * cse626;
	jac(11, 13) = -cse328 * cse626;
	jac(11, 14) = -cse329 * cse626;
	jac(11, 15) = -cse335 * cse626;
	jac(11, 16) = -cse337 * cse626 + cse621;
	jac(11, 17) = -cse349 * cse626 + cse587;
	jac(11, 18) = -cse358 * cse626;
	jac(11, 19) = cse626;
	jac(12, 1) = cse195 - cse602 - cse629 * cse67;
	jac(12, 2) = -cse111 - cse209 * cse629 - cse602;
	jac(12, 3) = -cse232 * cse629 - cse572;
	jac(12, 4) = -cse278 * cse629 + cse281 - cse282 - cse518 + cse607 - cse608;
	jac(12, 5) = -cse296 * cse629 - cse608;
	jac(12, 6) = -cse302 * cse629;
	jac(12, 7) = -cse303 * cse629 + cse594;
	jac(12, 8) = -cse312 * cse629;
	jac(12, 9) = -cse313 * cse629;
	jac(12, 10) = -cse315 * cse629 + cse624;
	jac(12, 11) = -cse316 * cse629 - cse602;
	jac(12, 12) = cse323 - cse324 * cse629 - cse327 - cse538;
	jac(12, 13) = -cse328 * cse629;
	jac(12, 14) = -cse329 * cse629;
	jac(12, 15) = -cse335 * cse629 - cse625;
	jac(12, 16) = -cse337 * cse629 + cse341 + cse603;
	jac(12, 17) = -cse349 * cse629 - cse547 - cse614;
	jac(12, 18) = cse355 + cse356 - cse358 * cse629;
	jac(12, 19) = cse629;
	jac(13, 1) = cse478 - cse631 * cse67;
	jac(13, 2) = -cse209 * cse631 + cse478;
	jac(13, 3) = -cse232 * cse631 + cse493;
	jac(13, 4) = -cse278 * cse631 + cse507;
	jac(13, 5) = -cse296 * cse631 + cse506;
	jac(13, 6) = -cse302 * cse631;
	jac(13, 7) = -cse303 * cse631;
	jac(13, 8) = -cse312 * cse631;
	jac(13, 9) = -cse313 * cse631;
	jac(13, 10) = -cse315 * cse631;
	jac(13, 11) = -cse316 * cse631 + cse478;
	jac(13, 12) = -cse324 * cse631;
	jac(13, 13) = -cse328 * cse631 - cse542;
	jac(13, 14) = -cse329 * cse631 + cse543;
	jac(13, 15) = -cse335 * cse631 + cse478;
	jac(13, 16) = -cse337 * cse631 + cse478;
	jac(13, 17) = -cse349 * cse631 + cse554;
	jac(13, 18) = -cse358 * cse631;
	jac(13, 19) = cse631;
	jac(14, 1) = -cse478 - cse632 * cse67;
	jac(14, 2) = -cse209 * cse632 - cse478;
	jac(14, 3) = -cse232 * cse632 - cse493;
	jac(14, 4) = -cse278 * cse632 - cse507;
	jac(14, 5) = -cse296 * cse632 - cse506;
	jac(14, 6) = -cse302 * cse632;
	jac(14, 7) = -cse303 * cse632;
	jac(14, 8) = -cse312 * cse632;
	jac(14, 9) = -cse313 * cse632;
	jac(14, 10) = -cse315 * cse632;
	jac(14, 11) = -cse316 * cse632 - cse478;
	jac(14, 12) = -cse324 * cse632;
	jac(14, 13) = -cse328 * cse632 + cse542;
	jac(14, 14) = -cse329 * cse632 - cse543;
	jac(14, 15) = -cse335 * cse632 - cse478;
	jac(14, 16) = -cse337 * cse632 - cse478;
	jac(14, 17) = -cse349 * cse632 - cse554;
	jac(14, 18) = -cse358 * cse632;
	jac(14, 19) = cse632;
	jac(15, 1) = -cse2 - cse633 * cse67;
	jac(15, 2) = -cse209 * cse633;
	jac(15, 3) = -cse232 * cse633;
	jac(15, 4) = -cse278 * cse633 + cse634 * state.xn[8];
	jac(15, 5) = -cse296 * cse633;
	jac(15, 6) = -cse302 * cse633;
	jac(15, 7) = -cse303 * cse633;
	jac(15, 8) = -cse312 * cse633 + cse529;
	jac(15, 9) = -cse313 * cse633 + cse634 * state.xn[3];
	jac(15, 10) = -cse315 * cse633;
	jac(15, 11) = -cse316 * cse633;
	jac(15, 12) = -cse324 * cse633 - cse326;
	jac(15, 13) = -cse328 * cse633;
	jac(15, 14) = -cse329 * cse633;
	jac(15, 15) = -cse330 - cse335 * cse633 - cse612;
	jac(15, 16) = -cse337 * cse633;
	jac(15, 17) = -cse349 * cse633 + cse546;
	jac(15, 18) = -cse358 * cse633;
	jac(15, 19) = cse633;
	jac(16, 1) = -cse635 * cse67;
	jac(16, 2) = -cse209 * cse635;
	jac(16, 3) = -cse232 * cse635 + cse497 - cse498 - cse573;
	jac(16, 4) = -cse278 * cse635 + cse519;
	jac(16, 5) = -cse296 * cse635;
	jac(16, 6) = -cse302 * cse635;
	jac(16, 7) = -cse303 * cse635 - cse307;
	jac(16, 8) = -cse311 - cse312 * cse635;
	jac(16, 9) = -cse313 * cse635 - cse620;
	jac(16, 10) = -cse315 * cse635;
	jac(16, 11) = -cse316 * cse635;
	jac(16, 12) = -cse323 - cse324 * cse635 + cse538;
	jac(16, 13) = -cse328 * cse635;
	jac(16, 14) = -cse329 * cse635;
	jac(16, 15) = -cse335 * cse635;
	jac(16, 16) = -cse337 * cse635 - cse342 - cse621;
	jac(16, 17) = -cse349 * cse635 + cse547;
	jac(16, 18) = cse357 - cse358 * cse635;
	jac(16, 19) = cse635;
	jac(17, 1) = -cse636 * cse67;
	jac(17, 2) = -cse209 * cse636;
	jac(17, 3) = -cse232 * cse636 - cse500;
	jac(17, 4) = -cse278 * cse636 + cse579;
	jac(17, 5) = cse295 - cse296 * cse636;
	jac(17, 6) = -cse302 * cse636;
	jac(17, 7) = -cse303 * cse636;
	jac(17, 8) = -cse312 * cse636 - cse529;
	jac(17, 9) = -cse313 * cse636;
	jac(17, 10) = -cse315 * cse636 - cse584;
	jac(17, 11) = -cse316 * cse636;
	jac(17, 12) = -cse324 * cse636 - cse539;
	jac(17, 13) = -cse328 * cse636;
	jac(17, 14) = -cse329 * cse636;
	jac(17, 15) = -cse335 * cse636;
	jac(17, 16) = -cse337 * cse636;
	jac(17, 17) = -cse349 * cse636 - cse548 - cse588;
	jac(17, 18) = -cse358 * cse636;
	jac(17, 19) = cse636;
	jac(18, 1) = -cse195 - cse637 * cse67;
	jac(18, 2) = cse111 - cse209 * cse637;
	jac(18, 3) = -cse232 * cse637 - cse494;
	jac(18, 4) = -cse278 * cse637 - cse580;
	jac(18, 5) = -cse296 * cse637;
	jac(18, 6) = -cse302 * cse637;
	jac(18, 7) = -cse303 * cse637 + cse307;
	jac(18, 8) = -cse312 * cse637;
	jac(18, 9) = -cse313 * cse637;
	jac(18, 10) = -cse315 * cse637;
	jac(18, 11) = -cse316 * cse637;
	jac(18, 12) = -cse324 * cse637 + cse325;
	jac(18, 13) = -cse328 * cse637;
	jac(18, 14) = -cse329 * cse637;
	jac(18, 15) = -cse335 * cse637;
	jac(18, 16) = -cse337 * cse637 + cse338;
	jac(18, 17) = -cse349 * cse637;
	jac(18, 18) = -cse355 - cse358 * cse637 - cse555;
	jac(18, 19) = cse637;
	jac(19, 1) = -1.6737729999999998e-24 * cse1021 +
		     cse1022 * (1.461185424e-21 * chi * cse1081 * cse1088 * cse272 * cse273 * state.xn[3] + 1.602177e-12 * crate * cse686 -
				cse1079 * (-cse1077 - cse774 * cse790 - cse799 * (cse1066 * cse1069 + cse1074 * (-cse1070 * cse1072 + cse774 * cse794))) -
				4.39089220777172e-29 * cse1080 * cse757 + 2.0187430200000004e-26 * cse1081 * cse1088 * cse11 * d2g * state.xn[0] +
				4.39089220777172e-29 * cse1090 * cse757 * cse762 * cse764 * cse768 * state.xn[8] -
				cse1092 * (cse1089 * cse1091 - cse1089 * cse765) + 3.0000000000000002e-15 * cse11 * cse745 * d2g - cse1110 -
				7.1777529600000012e-12 * cse559 - 7.1777529600000012e-12 * cse560 - cse687 - cse706 * cse715 - cse774 * cse803 - cse804 -
				state.xn[11] * (cse1031 * cse860 + cse1036 * cse861 + cse1039 * cse1040 + cse1039 * cse1041) -
				state.xn[3] * ((cse891) ? (0) : (cse1044 * cse922)) -
				state.xn[7] * ((cse1019) ? (cse1017 * cse1053 + cse1018 * cse1058 + cse1061 * cse1062 + cse1061 * cse1063) : (0))) -
		     cse1206 * cse67 + 5.0213189999997486e-24 * cse649 * cse679 * cse682 * state.T -
		     cse651 * (-cse201 * cse643 - 4.7173311503030642e-41 * cse203 - cse638 - cse639 - cse640 - cse642 - cse646) - cse681;
	jac(19, 2) = -1.6728620616289998e-24 * cse1021 +
		     cse1022 * (1.461185424e-21 * chi * cse1081 * cse1215 * cse272 * cse273 * state.xn[3] +
				2.0187430200000004e-26 * cse1081 * cse11 * cse1215 * d2g * state.xn[0] - cse1110 - cse1211 - cse1212 - cse1214 -
				state.xn[3] * ((cse891) ? (0) : (cse1044 * cse913))) -
		     cse1206 * cse209 + 5.0185861848867484e-24 * cse649 * cse679 * cse682 * state.T -
		     cse651 * (2.9387358770557188e-39 * cse111 + cse1114 * cse204 + cse1207 * state.xn[2] + cse1210) - cse681;
	jac(19, 3) =
	    -9.1093837099999994e-28 * cse1021 +
	    cse1022 * (1.7e-24 * chi * cse11 * cse1126 * cse155 * cse688 * d2g *
			   (0.00025272365919999998 * cse1127 * cse233 + 0.036605692000000002 * cse1128 * cse233) +
		       4.22974728e-11 * crate * cse11 * cse114 * cse683 * cse685 * state.xn[0] * (-cse1105 * cse1107 + 0.5 * cse12 * cse684) -
		       cse1079 * (-cse1073 * cse773 * cse799 - cse773 * cse790) - 2.6241084038530201e-25 * cse1080 * cse74 +
		       2.6241084038530201e-25 * cse1090 * cse74 * cse762 * cse764 * cse768 * state.xn[8] - cse1092 * (cse1091 * cse1226 - cse1226 * cse765) +
		       6.0732876035249664e-19 * cse1121 * cse1123 * cse1222 * cse709 * cse713 * state.xn[0] * state.xn[2] -
		       7.2151085033443149e-29 * cse1131 * cse1223 * cse1225 * cse174 * cse3 * state.xn[2] -
		       cse1133 * (5.9301787935659931e-17 * cse1121 * cse1123 * cse1222 * cse709 * state.xn[2] - cse704 * cse711) -
		       2.1789607199999999e-11 * cse244 - cse716 * state.xn[0] - cse727 -
		       cse728 * (0.086512999999999993 * chi * cse1130 * cse1224 * cse155 * cse3 * cse721 - cse1132 * cse1225 * cse719) - cse773 * cse803 -
		       state.xn[11] * (cse1040 * cse1233 + cse1041 * cse1233 + cse1230 * cse860 + cse1232 * cse861) -
		       state.xn[3] * ((cse891) ? (0) : (cse1044 * cse930)) -
		       state.xn[7] * ((cse1019) ? (cse1017 * cse1240 + cse1018 * cse1243 + cse1062 * cse1246 + cse1063 * cse1246) : (0))) -
	    cse1206 * cse232 + 2.732815112999863e-27 * cse649 * cse679 * cse682 * state.T -
	    cse651 * (cse1112 * cse496 - cse1113 * cse496 + cse1114 * cse245 + cse1115 * cse484 + cse1116 * cse488 + cse1207 * state.xn[1] +
		      cse1216 * state.xn[10] + cse1217 * state.xn[6] - cse1218 * cse228 - cse1219 * cse234 + cse1220 * cse234 - cse1221 * cse234 -
		      4.7173311503030642e-41 * cse244 + 4.7173311503030642e-41 * cse487 - cse492 * cse677 - cse658 - cse660) -
	    cse681;
	jac(19, 4) =
	    -3.3475459999999996e-24 * cse1021 +
	    cse1022 *
		(1.461185424e-21 * chi * cse1081 * cse1262 * cse272 * cse273 * state.xn[3] + 3.3999999999999999e-24 * chi * cse155 * cse688 * cse692 * d2g +
		 5.6999999999999997e-11 * chi * cse272 * cse273 * cse752 + 4.22974728e-11 * crate * cse11 * cse114 * cse1258 * cse683 * cse685 * state.xn[0] +
		 3.2043540000000001e-12 * crate * cse1259 * state.xn[3] + 3.2043540000000001e-12 * crate * cse751 -
		 cse1079 * (-cse1265 - cse790 - cse799 * (cse1069 * cse1263 + cse1074 * (-cse1070 * cse1264 + cse794))) +
		 2.0187430200000004e-26 * cse1081 * cse11 * cse1262 * d2g * state.xn[0] - cse1092 * (-cse1147 * cse1260 - cse761 * cse766) - cse1139 * cse761 -
		 cse1142 * cse1260 - cse1252 - cse1253 - cse1255 - 7.1777529600000012e-12 * cse254 - 7.1777529600000012e-12 * cse264 -
		 1.4355505920000002e-11 * cse265 * state.xn[3] - 7.1777529600000012e-12 * cse575 + 6.0000000000000005e-15 * cse745 * d2g * state.xn[0] -
		 cse803 - cse935 - state.xn[11] * (cse1040 * cse1271 + cse1041 * cse1271 + cse1267 * cse860 + cse1270 * cse861) -
		 state.xn[3] * ((cse891) ? (0) : (cse1044 * cse909)) -
		 state.xn[7] * ((cse1019) ? (cse1017 * cse1273 + cse1018 * cse1276 + cse1062 * cse1277 + cse1063 * cse1277) : (0))) -
	    cse1206 * cse278 + 1.0042637999999497e-23 * cse649 * cse679 * cse682 * state.T -
	    cse651 * (cse1112 * cse515 - cse1113 * cse515 + cse1218 * cse287 + cse1219 * cse279 - cse1220 * cse279 + cse1221 * cse279 + cse1247 * state.xn[12] +
		      cse1248 * state.xn[8] + cse1249 * state.xn[4] + cse1250 - cse285 * cse643 - 4.7173311503030642e-41 * cse286 - cse668 - cse673 - cse675 -
		      cse676) -
	    cse681;
	jac(19, 5) = -3.3466350616289996e-24 * cse1021 +
		     cse1022 * (-cse1079 * (-cse1265 - cse799 * (cse1067 * cse1068 * cse1263 - cse1213 * cse1264)) - cse1252 - cse1253 - cse1255 +
				cse1258 * cse1280 + cse1259 * cse1281 + 3.3999999999999999e-24 * cse1279 + cse1282 * cse1283 + cse1282 * cse1284 +
				cse58 * cse745 - 7.1777529600000012e-12 * cse581 - 7.1777529600000012e-12 * cse582) -
		     cse1206 * cse296 + 1.0039905184886497e-23 * cse649 * cse679 * cse682 * state.T -
		     cse651 * (cse1249 * state.xn[3] + cse1250 - cse299 * cse643 - 4.7173311503030642e-41 * cse300) - cse681;
	jac(19, 6) = 7.7835963979996672e-41 * crate * cse648 * cse649 * state.T - 6.6464729999999996e-24 * cse1021 - cse1206 * cse302 - cse1285 +
		     1.9939418999999001e-23 * cse649 * cse679 * cse682 * state.T;
	jac(19, 7) = -6.6455620616289996e-24 * cse1021 - cse1206 * cse303 - cse1285 + 1.9936686184886001e-23 * cse649 * cse679 * cse682 * state.T -
		     cse651 * (cse1115 * cse462 + cse1217 * state.xn[2] + 3.4713817547720679e-48 * cse665);
	jac(19, 8) = -cse1020 * cse1022 - 1.9944730000000003e-23 * cse1021 - cse1206 * cse312 + 5.9834189999997017e-23 * cse649 * cse679 * cse682 * state.T -
		     cse651 * (4.7173311503030642e-41 * cse528 * state.xn[16] - cse652 - cse664 - cse671 - cse672 * state.xn[3]) - cse681;
	jac(19, 9) = -1.9943819061629003e-23 * cse1021 - cse1022 * cse768 * cse771 - cse1206 * cse313 +
		     5.983145718488402e-23 * cse649 * cse679 * cse682 * state.T -
		     cse651 * (cse1116 * cse466 + cse1248 * state.xn[3] + 4.7173311503030642e-41 * cse534 - cse666) - cse681;
	jac(19, 10) =
	    -4.6512360000000008e-23 * cse1021 +
	    cse1022 * (-cse1079 * (cse1075 * cse1076 * cse1286 * cse775 - cse798 * (cse1075 * cse1076 * cse1286 * cse779 * cse791 - cse1287 * cse792) -
				   cse799 * (cse1067 * cse1068 * cse1286 - cse1213 * cse1287)) -
		       cse775 * cse802) -
	    cse1206 * cse315 + 1.3953707999999304e-22 * cse649 * cse679 * cse682 * state.T - cse651 * (-cse653 - cse656) - cse681;
	jac(19, 11) = -4.8185222061629005e-23 * cse1021 - cse1206 * cse316 - cse1288 + 1.4455566618487979e-22 * cse649 * cse679 * cse682 * state.T -
		      cse651 * (cse1210 + cse1216 * state.xn[2]);
	jac(19, 12) = -2.6567630000000005e-23 * cse1021 - cse1022 * cse862 - cse1206 * cse324 + 7.9702889999996024e-23 * cse649 * cse679 * cse682 * state.T -
		      cse651 * (2.9387358770557188e-39 * cse109 * cse94 * state.xn[1] + 9.3874889891030969e-50 * cse319 * cse376 * cse378 * state.xn[16] -
				cse654 - cse655 - cse670) -
		      cse681;
	jac(19, 13) = -4.6637070000000001e-23 * cse1021 - cse1206 * cse328 + 1.3991120999999301e-22 * cse649 * cse679 * cse682 * state.T -
		      cse651 * (cse1247 * state.xn[3] + 1.3012031544987096e-47 * cse541) - cse681;
	jac(19, 14) = -4.6636159061629004e-23 * cse1021 - cse1206 * cse329 + 1.3990847718488001e-22 * cse649 * cse679 * cse682 * state.T -
		      cse651 * (-cse657 * state.xn[2] - cse678) - cse681;
	jac(19, 15) = -2.1618503000000003e-23 * cse1021 - cse1206 * cse335 - cse1288 + 6.4855508999996759e-23 * cse649 * cse679 * cse682 * state.T -
		      cse651 * (-cse1289 - cse641 * state.xn[0] - cse661 - 2.5713938924237541e-50 * state.xn[11]);
	jac(19, 16) = -2.8241403000000005e-23 * cse1021 - cse1206 * cse337 - cse1288 + 8.4724208999995777e-23 * cse649 * cse679 * cse682 * state.T -
		      cse651 * (-cse1289 + 3.4713817547720679e-48 * cse305 * state.xn[6] - 2.6889433275059828e-48 * cse305 * state.xn[8] - cse662 -
				cse663 * state.xn[7] - 1.285696946211877e-50 * state.xn[11]);
	jac(19, 17) =
	    -5.0204080616289998e-24 * cse1021 +
	    cse1022 * (-cse1079 * (cse1075 * cse1076 * cse1291 * cse775 - cse798 * (cse1075 * cse1076 * cse1291 * cse779 * cse791 - cse1292 * cse792) -
				   cse799 * (cse1067 * cse1068 * cse1291 - cse1213 * cse1292)) -
		       cse1096 * (300.0 * cse1093 + 300.0 * cse1254) -
		       cse1103 * (6 * state.xn[0] + 6 * state.xn[1] + 6 * state.xn[10] + 6 * state.xn[14] + 6 * state.xn[15] + 18 * state.xn[16] +
				  12 * state.xn[3] + 12 * state.xn[4]) -
		       5.6994379667262474e-29 * cse1251 + 5.0999999999999995e-24 * cse1279 + cse1280 * (-1.5 * cse1256 + 1.5 * cse1257) +
		       cse1281 * ((cse747) ? (0) : ((cse748) ? ((9.0 / 2.0) * cse1098) : ((cse749) ? (4 * cse1098) : ((cse750) ? (cse1098) : (0))))) +
		       cse1283 * cse1290 + cse1284 * cse1290 - 7.1777529600000012e-12 * cse347 + cse585 * cse745 - 7.1777529600000012e-12 * cse586) -
	    cse1206 * cse349 + 1.5061224184886246e-23 * cse649 * cse679 * cse682 * state.T -
	    cse651 * (1.471807318894556e-37 * crate * cse36 * state.xn[3] * state.xn[7] + 1.7336191977363763e-51 * cse204 * d2g * state.xn[1] +
		      9.3874889891030969e-50 * cse319 * cse376 * cse378 * state.xn[11] - cse351 * cse643 - 4.7173311503030642e-41 * cse352 +
		      7.8854907508466019e-52 * cse462 * d2g * state.xn[6] + 6.4504786149244089e-51 * cse466 * d2g * state.xn[8] -
		      1.4573475330385548e-34 * cse477 - 1.8789373550961367e-50 * cse505 + 4.7173311503030642e-41 * cse528 * state.xn[7] - cse659 * state.xn[2] -
		      cse669 * state.xn[11] - 4.9958509909947217e-48 * state.xn[9]) -
	    cse681;
	jac(19, 18) = -2.6566719061629008e-23 * cse1021 - cse1206 * cse358 + 7.9700157184883038e-23 * cse649 * cse679 * cse682 * state.T -
		      cse651 * (-cse177 * cse674 * state.xn[3] - 2.9387358770557188e-39 * cse355 - cse667 * state.xn[3]) - cse681;
	jac(19, 19) = cse1206;
	// $JAFF END
}

AMREX_INLINE
void actual_rhs_init() {}

#endif
