# 1-D photodissociation region (PDR) with live radiation transport + the GOW
# (Gong, Ostriker & Wolfire 2017) 18-species chemistry/thermal network.
#
# The network (species set, CHEM_BANDS, EOS/GOW, sources) comes from the 'GOW'
# entry in src/networks/NetworkRegistry.cmake. network_properties.H is generated
# from it inside setup_target_for_microphysics_compilation() below, which also
# selects the Rosenbrock integrator (it uses the shipped analytic actual_jac).
set(microphysics_network_name "GOW")

set(JOB_NAME PDR)

setup_target_for_microphysics_compilation(${microphysics_network_name}
                                          "${CMAKE_CURRENT_BINARY_DIR}/"
                                          INTEGRATOR "Rosenbrock")

# The GOW RHS reads four per-cell self-shielding columns ncol_{H2,CO,C,H} through
# burn_t::aux[0..3]; override the global bare -DNAUX_NET (=1) for this problem only.
remove_definitions(-DNAUX_NET)
add_definitions(-DNAUX_NET=4)

# BEFORE keyword: give the GOW network + generated network_properties.H priority
# over the bundled Microphysics headers for targets in this directory.
include_directories(BEFORE ${${microphysics_network_name}_dirs} "${CMAKE_CURRENT_BINARY_DIR}/")

add_executable(
  ${JOB_NAME} testPDR.cpp "${QuokkaSourcesNoEOS}"
              ../../radiation/photochemistry.cpp
              ../../networks/GOW/gow_tables.cpp ${${microphysics_network_name}_sources})

# PHOTOCHEMISTRY enables the radiation<->chemistry coupling (rn[]/c_hat in burn_t,
# GetChemBandQuanta(), the ChemBands trait). NO SKIP_PHOTOCHEMFLUX: GOW is a
# 2-var-per-group network (it references the band-flux state.rn[2*g+1]).
# PDR_PARALLEL_NCOL_AUX: deliver the shielding columns per-cell via burn_t::aux[]
# (requires NAUX_NET>=4) instead of the network_rp::ncol_* globals -> thread/GPU safe.
target_compile_definitions(${JOB_NAME} PUBLIC PHOTOCHEMISTRY PDR_PARALLEL_NCOL_AUX)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(${JOB_NAME})
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(
  NAME ${JOB_NAME}
  COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}.toml ${QuokkaTestParams}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
