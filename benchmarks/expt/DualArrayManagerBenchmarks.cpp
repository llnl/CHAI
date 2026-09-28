//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "benchmark/benchmark.h"

#include <cstddef>

#include "chai/ExecutionContextGuard.hpp"
#include "chai/expt/DualArrayManager.hpp"

namespace {
  using ::chai::ExecutionContext;
  using ::chai::ExecutionContextGuard;
  using ::chai::ExecutionContextManager;
  using ::chai::expt::DualArrayManager;

  static void DualArrayManager_DataSequence(benchmark::State& state,
                                            ExecutionContext initial_context,
                                            bool initial_touch,
                                            ExecutionContext call_context,
                                            bool call_touch)
  {
    const auto size = static_cast<std::size_t>(state.range(0));

    for (auto _ : state)
    {
      state.PauseTiming();

      auto& contextManager = ExecutionContextManager::getInstance();
      contextManager.reset();

      DualArrayManager<int> manager{size};
      {
        ExecutionContextGuard guard{initial_context};
        int* initial_data = manager.data(/*touch=*/initial_touch);
        benchmark::DoNotOptimize(initial_data);
      }

      {
        ExecutionContextGuard guard{call_context};
        state.ResumeTiming();  // measure only the data() call
        int* data = manager.data(/*touch=*/call_touch);
        benchmark::DoNotOptimize(data);
        state.PauseTiming();  // exclude guard destruction
      }
    }

    state.SetItemsProcessed(state.iterations());
    state.SetBytesProcessed(state.iterations() * size * sizeof(int));
  }

  static void DualArrayManager_DefaultConstruct(benchmark::State& state)
  {
    for (auto _ : state)
    {
      DualArrayManager<int> manager{};
      benchmark::DoNotOptimize(manager);
    }
  }

  BENCHMARK(DualArrayManager_DefaultConstruct);

  static void DualArrayManager_SizeConstruct(benchmark::State& state)
  {
    const auto size = static_cast<std::size_t>(state.range(0));

    for (auto _ : state)
    {
      DualArrayManager<int> manager{size};
      benchmark::DoNotOptimize(manager);
    }

    state.SetBytesProcessed(state.iterations() * size * sizeof(int));
  }

  BENCHMARK(DualArrayManager_SizeConstruct)
    ->Arg(0)
    ->RangeMultiplier(8)
    ->Range(1, 1 << 20);

  static void DualArrayManager_AfterHostRead_DataHostRead(benchmark::State& state)
  {
    DualArrayManager_DataSequence(state,
                                  /*initial_context=*/ExecutionContext::HOST,
                                  /*initial_touch=*/false,
                                  /*call_context=*/ExecutionContext::HOST,
                                  /*call_touch=*/false);
  }

  BENCHMARK(DualArrayManager_AfterHostRead_DataHostRead)
    ->Arg(0)
    ->RangeMultiplier(8)
    ->Range(1, 1 << 20)
    ->Unit(benchmark::kNanosecond);

  static void DualArrayManager_AfterHostWrite_DataHostRead(benchmark::State& state)
  {
    DualArrayManager_DataSequence(state,
                                  /*initial_context=*/ExecutionContext::HOST,
                                  /*initial_touch=*/true,
                                  /*call_context=*/ExecutionContext::HOST,
                                  /*call_touch=*/false);
  }

  BENCHMARK(DualArrayManager_AfterHostWrite_DataHostRead)
    ->Arg(0)
    ->RangeMultiplier(8)
    ->Range(1, 1 << 20)
    ->Unit(benchmark::kNanosecond);

  static void DualArrayManager_AfterHostRead_DataDeviceRead(benchmark::State& state)
  {
    DualArrayManager_DataSequence(state,
                                  /*initial_context=*/ExecutionContext::HOST,
                                  /*initial_touch=*/false,
                                  /*call_context=*/ExecutionContext::DEVICE,
                                  /*call_touch=*/false);
  }

  BENCHMARK(DualArrayManager_AfterHostRead_DataDeviceRead)
    ->Arg(0)
    ->RangeMultiplier(8)
    ->Range(1, 1 << 20)
    ->Unit(benchmark::kNanosecond);

  static void DualArrayManager_AfterHostWrite_DataDeviceRead(benchmark::State& state)
  {
    DualArrayManager_DataSequence(state,
                                  /*initial_context=*/ExecutionContext::HOST,
                                  /*initial_touch=*/true,
                                  /*call_context=*/ExecutionContext::DEVICE,
                                  /*call_touch=*/false);
  }

  BENCHMARK(DualArrayManager_AfterHostWrite_DataDeviceRead)
    ->Arg(0)
    ->RangeMultiplier(8)
    ->Range(1, 1 << 20)
    ->Unit(benchmark::kNanosecond);

  static void DualArrayManager_AfterDeviceRead_DataHostRead(benchmark::State& state)
  {
    DualArrayManager_DataSequence(state,
                                  /*initial_context=*/ExecutionContext::DEVICE,
                                  /*initial_touch=*/false,
                                  /*call_context=*/ExecutionContext::HOST,
                                  /*call_touch=*/false);
  }

  BENCHMARK(DualArrayManager_AfterDeviceRead_DataHostRead)
    ->Arg(0)
    ->RangeMultiplier(8)
    ->Range(1, 1 << 20)
    ->Unit(benchmark::kNanosecond);

  static void DualArrayManager_AfterDeviceWrite_DataHostRead(benchmark::State& state)
  {
    DualArrayManager_DataSequence(state,
                                  /*initial_context=*/ExecutionContext::DEVICE,
                                  /*initial_touch=*/true,
                                  /*call_context=*/ExecutionContext::HOST,
                                  /*call_touch=*/false);
  }

  BENCHMARK(DualArrayManager_AfterDeviceWrite_DataHostRead)
    ->Arg(0)
    ->RangeMultiplier(8)
    ->Range(1, 1 << 20)
    ->Unit(benchmark::kNanosecond);

  static void DualArrayManager_AfterDeviceRead_DataDeviceRead(benchmark::State& state)
  {
    DualArrayManager_DataSequence(state,
                                  /*initial_context=*/ExecutionContext::DEVICE,
                                  /*initial_touch=*/false,
                                  /*call_context=*/ExecutionContext::DEVICE,
                                  /*call_touch=*/false);
  }

  BENCHMARK(DualArrayManager_AfterDeviceRead_DataDeviceRead)
    ->Arg(0)
    ->RangeMultiplier(8)
    ->Range(1, 1 << 20)
    ->Unit(benchmark::kNanosecond);

  static void DualArrayManager_AfterDeviceWrite_DataDeviceRead(benchmark::State& state)
  {
    DualArrayManager_DataSequence(state,
                                  /*initial_context=*/ExecutionContext::DEVICE,
                                  /*initial_touch=*/true,
                                  /*call_context=*/ExecutionContext::DEVICE,
                                  /*call_touch=*/false);
  }

  BENCHMARK(DualArrayManager_AfterDeviceWrite_DataDeviceRead)
    ->Arg(0)
    ->RangeMultiplier(8)
    ->Range(1, 1 << 20)
    ->Unit(benchmark::kNanosecond);
}  // namespace

BENCHMARK_MAIN();
