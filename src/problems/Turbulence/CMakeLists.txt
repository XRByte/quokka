# Driven turbulence with the radiation-free Gong, Ostriker & Wolfire (2017)
# 18-species chemistry + thermal network ('GOW_nr', src/networks/NetworkRegistry.cmake).
#
# FUV shielding uses a local (Jeans-length) column-density approximation
# (Safranek-Shrader et al. 2017): each cell's Av and H2/CO/C/H columns are
# computed from its own state and delivered to the network RHS through
# burn_t::aux[0..4], so the burn is thread/GPU safe (no network_rp globals).
#
# The chemistry-free baseline lives in src/problems/TurbulenceNoChem.
if (AMReX_SPACEDIM EQUAL 3)
    set(microphysics_network_name "GOW_nr")

    set(JOB_NAME Turbulence)

    setup_target_for_microphysics_compilation(${microphysics_network_name}
                                              "${CMAKE_CURRENT_BINARY_DIR}/"
                                              INTEGRATOR "Rosenbrock")

    # aux[0..3] = ncol_{H2,CO,C,H}, aux[4] = Av; override the global bare -DNAUX_NET (=1)
    remove_definitions(-DNAUX_NET)
    add_definitions(-DNAUX_NET=5)

    # BEFORE keyword: give the GOW_nr network + generated network_properties.H
    # priority over the bundled Microphysics headers for targets in this directory.
    include_directories(BEFORE ${${microphysics_network_name}_dirs} "${CMAKE_CURRENT_BINARY_DIR}/")

    add_executable(
      ${JOB_NAME} testTurbulence.cpp "${QuokkaSourcesNoEOS}"
                  ../../chemistry/Chemistry.cpp
                  ${${microphysics_network_name}_sources})

    # CHEMISTRY: microphysics EOS/network; PDR_PARALLEL_NCOL_AUX: per-cell columns + Av via burn_t::aux
    target_compile_definitions(${JOB_NAME} PUBLIC CHEMISTRY PDR_PARALLEL_NCOL_AUX)

    if(AMReX_GPU_BACKEND MATCHES "CUDA")
        setup_target_for_cuda_compilation(${JOB_NAME})
    endif(AMReX_GPU_BACKEND MATCHES "CUDA")

    # the GOW_nr RHS loads its cooling tables from the run directory
    file(COPY "${CMAKE_SOURCE_DIR}/src/networks/GOW_nr/gow_cooling_tables.hdf5"
         DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
    file(COPY "${CMAKE_SOURCE_DIR}/src/networks/GOW_nr/gow_cooling_tables.hdf5"
         DESTINATION "${CMAKE_SOURCE_DIR}/tests")

    add_test(
      NAME ${JOB_NAME}
      COMMAND ${JOB_NAME} ../inputs/${JOB_NAME}.toml ${QuokkaTestParams}
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)
endif()
