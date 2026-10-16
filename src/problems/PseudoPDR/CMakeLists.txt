# 1-D "pseudo" photodissociation region (PDR): a log-spaced column-density sweep
# of independent one-zone Gong, Ostriker & Wolfire (2017) chemistry/thermal
# problems, marched inward with per-zone Av and self-shielding columns.
#
# Uses the radiation-free 'GOW_nr' network (src/networks/NetworkRegistry.cmake):
# the FUV field is the scalar network_rp::chi attenuated by the per-zone
# network_rp::Av / ncol_* set in testPseudoPDR.cpp. No radiation transport.
set(microphysics_network_name "GOW_nr")

set(JOB_NAME PseudoPDR)

setup_target_for_microphysics_compilation(${microphysics_network_name}
                                          "${CMAKE_CURRENT_BINARY_DIR}/"
                                          INTEGRATOR "VODE")

# BEFORE keyword: give the GOW_nr network + generated network_properties.H
# priority over the bundled Microphysics headers for targets in this directory.
include_directories(BEFORE ${${microphysics_network_name}_dirs} "${CMAKE_CURRENT_BINARY_DIR}/")

add_executable(
  ${JOB_NAME} testPseudoPDR.cpp "${QuokkaSourcesNoEOS}"
              ../../chemistry/Chemistry.cpp
              ${${microphysics_network_name}_sources})

# this will add #define CHEMISTRY
target_compile_definitions(${JOB_NAME} PUBLIC CHEMISTRY)

if(AMReX_GPU_BACKEND MATCHES "CUDA")
    setup_target_for_cuda_compilation(${JOB_NAME})
endif(AMReX_GPU_BACKEND MATCHES "CUDA")

add_test(
  NAME ${JOB_NAME}
  COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}.toml ${QuokkaTestParams}
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
