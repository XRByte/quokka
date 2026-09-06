module purge
module load gcc/14.2.0
module load cuda/12.9.0

module unload openmpi
module load openmpi/5.0.8

# hdf5
module load hdf5/1.12.2

# cmake
module unload cmake
module load cmake/3.31.6

# python
module load python3/3.12.1

# CUDA arch for Gadi gpuvolta = Tesla V100 (Volta, sm_70)
export AMREX_CUDA_ARCH=7.0

# compiler environment hints
export CC=mpicc
export CXX=mpicxx
export CUDACXX=$(which nvcc)
export CUDAHOSTCXX=mpicxx

# mpicxx must wrap the gcc/14.2.0 backend (for <format>, C++20)
export OMPI_CXX=g++
export OMPI_CC=gcc
