##############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
# contributors. See the CHAI LICENSE and COPYRIGHT files for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################
set (chai_benchmark_depends
  chai
  gbenchmark)

if (CHAI_ENABLE_CUDA)
  set (chai_benchmark_depends
    ${chai_benchmark_depends}
    cuda)
endif ()

if (CHAI_ENABLE_HIP)
  set (chai_benchmark_depends
    ${chai_benchmark_depends}
    blt::hip)
endif ()

blt_add_executable(
  NAME arraymanager_benchmarks
  SOURCES chai_arraymanager_benchmarks.cpp
  DEPENDS_ON ${chai_benchmark_depends})

blt_add_benchmark(
  NAME arraymanager_benchmarks
  COMMAND arraymanager_benchmarks)

blt_add_executable(
  NAME managedarray_benchmarks
  SOURCES chai_managedarray_benchmarks.cpp
  DEPENDS_ON ${chai_benchmark_depends})

blt_add_benchmark(
  NAME managedarray_benchmarks
  COMMAND managedarray_benchmarks)

if (CHAI_ENABLE_RAJA_PLUGIN)
  blt_add_executable(
    NAME raja_plugin_benchmarks
    SOURCES chai_raja_plugin_benchmarks.cpp
    DEPENDS_ON ${chai_benchmark_depends} RAJA)

  blt_add_benchmark(
    NAME raja_plugin_benchmarks
    COMMAND raja_plugin_benchmarks)
endif ()

if (CHAI_ENABLE_MANAGED_PTR)
  blt_add_executable(
    NAME managed_ptr_benchmarks
    SOURCES chai_managed_ptr_benchmarks.cpp
    DEPENDS_ON ${chai_benchmark_depends})

  blt_add_benchmark(
    NAME managed_ptr_benchmarks
    COMMAND managed_ptr_benchmarks)
endif ()

if (CHAI_ENABLE_EXPERIMENTAL)
  add_subdirectory(expt)
endif ()
