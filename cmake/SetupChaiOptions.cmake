############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
# contributors. See the CHAI LICENSE and COPYRIGHT files for details.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################
option(CHAI_ENABLE_EXPERIMENTAL "Enable experimental chai features." Off)
option(CHAI_ENABLE_GPU_SIMULATION_MODE "Enable GPU Simulation Mode" Off)
option(CHAI_ENABLE_OPENMP "Enable OpenMP" Off)
option(CHAI_ENABLE_MPI "Enable MPI (for umpire replay only)" Off)

option(CHAI_DISABLE_RM "Make ManagedArray a thin wrapper" Off)
mark_as_advanced(CHAI_DISABLE_RM)

option(CHAI_ENABLE_UM "Use CUDA unified (managed) memory" Off)
option(CHAI_THIN_GPU_ALLOCATE "Single memory space model" Off)
option(CHAI_ENABLE_PINNED "Use pinned host memory" Off)
option(CHAI_ENABLE_RAJA_PLUGIN "Build plugin to set RAJA execution spaces" On)
option(CHAI_ENABLE_GPU_ERROR_CHECKING "Enable GPU error checking" On)
option(CHAI_ENABLE_MANAGED_PTR "Enable managed_ptr" On)
option(CHAI_DEBUG "Enable Debug Logging." Off)
option(CHAI_ENABLE_RAJA_NESTED_TEST "Enable raja-chai-nested-tests, which fails to build on Debug CUDA builds." On)
option(CHAI_ENABLE_MANAGED_PTR_ON_GPU "Enable managed_ptr on GPU" On)

option(CHAI_ENABLE_TESTS "Enable CHAI tests" On)
option(CHAI_ENABLE_BENCHMARKS "Enable benchmarks" Off)
option(CHAI_ENABLE_EXAMPLES "Enable CHAI examples" Off)
option(CHAI_ENABLE_REPRODUCERS "Enable CHAI reproducers" Off)
option(CHAI_ENABLE_DOCS "Enable CHAI documentation" Off)

# options for Umpire as TPL
set(ENABLE_GMOCK On CACHE BOOL "")
option(CHAI_ENABLE_ASSERTS "Build Umpire with assert() enabled" On)
set(ENABLE_GTEST_DEATH_TESTS ${CHAI_ENABLE_ASSERTS} CACHE BOOL "")

option(CHAI_ENABLE_COPY_HEADERS "Enable CHAI copy headers" Off)

set(ENABLE_CUDA Off CACHE BOOL "Enable CUDA")

if (CHAI_ENABLE_UM AND NOT ENABLE_CUDA AND NOT CHAI_THIN_GPU_ALLOCATE)
  message(FATAL_ERROR "Option CHAI_ENABLE_UM requires ENABLE_CUDA or CHAI_THIN_GPU_ALLOCATE")
endif()

if (CHAI_THIN_GPU_ALLOCATE AND NOT CHAI_DISABLE_RM)
  message(FATAL_ERROR "Option CHAI_THIN_GPU_ALLOCATE requires CHAI_DISABLE_RM")
endif()
