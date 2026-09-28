//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#include <algorithm>
#include <cstdint>

#include "benchmark/benchmark.h"
#include "chai/ManagedArray.hpp"
#include "chai/config.hpp"

namespace
{

constexpr std::int64_t min_elements = 1;
constexpr std::int64_t max_elements = 1 << 24;
constexpr std::int64_t max_reallocate_elements = max_elements / 2;

void set_items_and_bytes_processed(benchmark::State& state,
                                   std::size_t elements,
                                   std::size_t operations = 1)
{
  const auto total_operations = static_cast<std::int64_t>(state.iterations()) *
                                static_cast<std::int64_t>(operations);
  state.SetItemsProcessed(total_operations *
                          static_cast<std::int64_t>(elements));
  state.SetBytesProcessed(total_operations *
                          static_cast<std::int64_t>(elements * sizeof(int)));
}

template <chai::ExecutionSpace Space>
void benchmark_managedarray_allocate(benchmark::State& state)
{
  const auto elements = static_cast<std::size_t>(state.range(0));

  for (auto _ : state) {
    (void)_;
    chai::ManagedArray<int> array(elements, Space);
    benchmark::DoNotOptimize(array.getActivePointer());
    array.free();
  }

  set_items_and_bytes_processed(state, elements);
}

void benchmark_managedarray_cached_host_access(benchmark::State& state)
{
  const auto elements = static_cast<std::size_t>(state.range(0));
  chai::ManagedArray<int> array(elements, chai::CPU);
  std::fill(array.data(), array.data() + elements, 1);

  for (auto _ : state) {
    (void)_;
    benchmark::DoNotOptimize(array.cdata());
  }

  array.free();
}

void benchmark_managedarray_shallow_copy(benchmark::State& state)
{
  const auto elements = static_cast<std::size_t>(state.range(0));
  chai::ManagedArray<int> source(elements, chai::CPU);

  for (auto _ : state) {
    (void)_;
    chai::ManagedArray<int> copy(source);
    benchmark::DoNotOptimize(copy.getActivePointer());
  }

  source.free();
}

void benchmark_managedarray_slice(benchmark::State& state)
{
  const auto elements = static_cast<std::size_t>(state.range(0));
  chai::ManagedArray<int> source(elements, chai::CPU);

  for (auto _ : state) {
    (void)_;
    auto slice = source.slice(0, elements);
    benchmark::DoNotOptimize(slice.getActivePointer());
  }

  source.free();
}

void benchmark_managedarray_clone(benchmark::State& state)
{
  const auto elements = static_cast<std::size_t>(state.range(0));
  chai::ManagedArray<int> source(elements, chai::CPU);
  std::fill(source.data(), source.data() + elements, 1);

  for (auto _ : state) {
    (void)_;
    auto copy = source.clone();
    benchmark::DoNotOptimize(copy.getActivePointer());
    copy.free();
  }

  source.free();
  set_items_and_bytes_processed(state, elements);
}

void benchmark_managedarray_reallocate_grow(benchmark::State& state)
{
  const auto elements = static_cast<std::size_t>(state.range(0));

  for (auto _ : state) {
    (void)_;
    state.PauseTiming();
    chai::ManagedArray<int> array(elements, chai::CPU);
    std::fill(array.data(), array.data() + elements, 1);
    state.ResumeTiming();

    array.reallocate(2 * elements);
    benchmark::DoNotOptimize(array.getActivePointer());

    state.PauseTiming();
    array.free();
    state.ResumeTiming();
  }

  set_items_and_bytes_processed(state, elements);
}

BENCHMARK_TEMPLATE(benchmark_managedarray_allocate, chai::NONE)
    ->RangeMultiplier(8)
    ->Range(min_elements, max_elements)
    ->ArgName("elements");
BENCHMARK_TEMPLATE(benchmark_managedarray_allocate, chai::CPU)
    ->RangeMultiplier(8)
    ->Range(min_elements, max_elements)
    ->ArgName("elements");
BENCHMARK(benchmark_managedarray_cached_host_access)
    ->RangeMultiplier(8)
    ->Range(min_elements, max_elements)
    ->ArgName("elements");
BENCHMARK(benchmark_managedarray_shallow_copy)
    ->RangeMultiplier(8)
    ->Range(min_elements, max_elements)
    ->ArgName("elements");
BENCHMARK(benchmark_managedarray_slice)
    ->RangeMultiplier(8)
    ->Range(min_elements, max_elements)
    ->ArgName("elements");
BENCHMARK(benchmark_managedarray_clone)
    ->RangeMultiplier(8)
    ->Range(min_elements, max_elements)
    ->ArgName("elements");
BENCHMARK(benchmark_managedarray_reallocate_grow)
    ->RangeMultiplier(8)
    ->Range(min_elements, max_reallocate_elements)
    ->ArgName("elements");

#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
void benchmark_managedarray_move_round_trip(benchmark::State& state)
{
  const auto elements = static_cast<std::size_t>(state.range(0));
  chai::ManagedArray<int> array(elements, chai::CPU);
  std::fill(array.data(), array.data() + elements, 1);

  // Establish both allocations before timing. The device-to-host move makes
  // each measured round trip synchronous.
  array.move(chai::GPU);
  array.move(chai::CPU);

  for (auto _ : state) {
    (void)_;
    array.move(chai::GPU);
    array.move(chai::CPU);
  }

  array.free();
  set_items_and_bytes_processed(state, elements, 2);
}

BENCHMARK_TEMPLATE(benchmark_managedarray_allocate, chai::GPU)
    ->RangeMultiplier(8)
    ->Range(min_elements, max_elements)
    ->ArgName("elements");
BENCHMARK(benchmark_managedarray_move_round_trip)
    ->RangeMultiplier(8)
    ->Range(min_elements, max_elements)
    ->ArgName("elements");
#endif

#if defined(CHAI_ENABLE_UM)
BENCHMARK_TEMPLATE(benchmark_managedarray_allocate, chai::UM)
    ->RangeMultiplier(8)
    ->Range(min_elements, max_elements)
    ->ArgName("elements");
#endif

#if defined(CHAI_ENABLE_PINNED)
BENCHMARK_TEMPLATE(benchmark_managedarray_allocate, chai::PINNED)
    ->RangeMultiplier(8)
    ->Range(min_elements, max_elements)
    ->ArgName("elements");
#endif

}  // namespace

BENCHMARK_MAIN();
