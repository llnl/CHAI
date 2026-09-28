//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#include <string>

#include "RAJA/RAJA.hpp"
#include "benchmark/benchmark.h"
#include "chai/ManagedArray.hpp"
#include "chai/RajaExecutionSpacePlugin.hpp"
#include "chai/config.hpp"

namespace
{

template <typename Policy>
void benchmark_raja_plugin_callbacks(benchmark::State& state)
{
  chai::RajaExecutionSpacePlugin plugin;
  const auto context = RAJA::util::make_context<Policy>(std::string{});

  // Initialize the plugin's cached ArrayManager pointer before timing.
  plugin.preCapture(context);
  plugin.postCapture(context);

  for (auto _ : state) {
    (void)_;
    plugin.preCapture(context);
    benchmark::ClobberMemory();
    plugin.postCapture(context);
  }

  state.SetItemsProcessed(state.iterations());
}

void benchmark_raja_empty_capture_host(benchmark::State& state)
{
  const RAJA::RangeSegment empty_range(0, 0);

  for (auto _ : state) {
    (void)_;
    RAJA::forall<RAJA::seq_exec>(empty_range, [](RAJA::Index_type) {});
  }

  state.SetItemsProcessed(state.iterations());
}

void benchmark_raja_managedarray_capture_host(benchmark::State& state)
{
  chai::ManagedArray<int> array(1, chai::CPU);
  const RAJA::RangeSegment empty_range(0, 0);

  for (auto _ : state) {
    (void)_;
    RAJA::forall<RAJA::seq_exec>(empty_range,
                                 [=](RAJA::Index_type) { (void)array; });
  }

  array.free();
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK_TEMPLATE(benchmark_raja_plugin_callbacks, RAJA::seq_exec);
BENCHMARK(benchmark_raja_empty_capture_host);
BENCHMARK(benchmark_raja_managedarray_capture_host);

#if defined(CHAI_ENABLE_CUDA)
BENCHMARK_TEMPLATE(benchmark_raja_plugin_callbacks, RAJA::cuda_exec<256>);
#endif

#if defined(CHAI_ENABLE_HIP)
BENCHMARK_TEMPLATE(benchmark_raja_plugin_callbacks, RAJA::hip_exec<256>);
#endif

}  // namespace

BENCHMARK_MAIN();
