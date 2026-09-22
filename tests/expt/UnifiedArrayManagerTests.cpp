//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/config.hpp"
#include "chai/expt/Context.hpp"
#include "chai/expt/ContextGuard.hpp"
#include "chai/expt/ContextManager.hpp"
#include "chai/expt/UnifiedArrayManager.hpp"
#include "camp/helpers.hpp"
#include "gtest/gtest.h"

#include <cstddef>

#if defined(CHAI_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

#if defined(CHAI_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace {
  using ::chai::expt::Context;
  using ::chai::expt::ContextGuard;
  using ::chai::expt::ContextManager;
  using ::chai::expt::UnifiedArrayManager;

  template <typename T>
  T* malloc_managed(std::size_t count)
  {
    T* ptr = nullptr;

#if defined(CHAI_ENABLE_CUDA)
    CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMallocManaged, (void**)&ptr, sizeof(T) * count);
#elif defined(CHAI_ENABLE_HIP)
    CAMP_HIP_API_INVOKE_AND_CHECK(hipMallocManaged, (void**)&ptr, sizeof(T) * count);
#else
    static_cast<void>(count);
#endif

    return ptr;
  }

  inline void free_managed(void* ptr)
  {
#if defined(CHAI_ENABLE_CUDA)
    CAMP_CUDA_API_INVOKE_AND_CHECK(cudaFree, ptr);
#elif defined(CHAI_ENABLE_HIP)
    CAMP_HIP_API_INVOKE_AND_CHECK(hipFree, ptr);
#else
    static_cast<void>(ptr);
#endif
  }

#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
  __global__ void increment_kernel(int* data, std::size_t size)
  {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size)
    {
      data[i] += 1;
    }
  }

  __global__ void copy_kernel(const int* in, int* out, std::size_t size)
  {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size)
    {
      out[i] = in[i];
    }
  }

  inline void device_synchronize_raw()
  {
#if defined(CHAI_ENABLE_CUDA)
    CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
#elif defined(CHAI_ENABLE_HIP)
    CAMP_HIP_API_INVOKE_AND_CHECK(hipDeviceSynchronize);
#endif
  }

  inline void launch_increment(int* data, std::size_t size)
  {
    constexpr int BLOCK_SIZE = 256;
    const int grid_size = static_cast<int>((size + BLOCK_SIZE - 1) / BLOCK_SIZE);

#if defined(CHAI_ENABLE_CUDA)
    increment_kernel<<<grid_size, BLOCK_SIZE>>>(data, size);
    CAMP_CUDA_API_INVOKE_AND_CHECK(cudaGetLastError);
#elif defined(CHAI_ENABLE_HIP)
    hipLaunchKernelGGL(increment_kernel,
                       dim3(grid_size),
                       dim3(BLOCK_SIZE),
                       0,
                       0,
                       data,
                       size);
    CAMP_HIP_API_INVOKE_AND_CHECK(hipGetLastError);
#endif
  }

  inline void launch_copy(const int* in, int* out, std::size_t size)
  {
    constexpr int BLOCK_SIZE = 256;
    const int grid_size = static_cast<int>((size + BLOCK_SIZE - 1) / BLOCK_SIZE);

#if defined(CHAI_ENABLE_CUDA)
    copy_kernel<<<grid_size, BLOCK_SIZE>>>(in, out, size);
    CAMP_CUDA_API_INVOKE_AND_CHECK(cudaGetLastError);
#elif defined(CHAI_ENABLE_HIP)
    hipLaunchKernelGGL(copy_kernel,
                       dim3(grid_size),
                       dim3(BLOCK_SIZE),
                       0,
                       0,
                       in,
                       out,
                       size);
    CAMP_HIP_API_INVOKE_AND_CHECK(hipGetLastError);
#endif
  }
#endif

  class UnifiedArrayManagerTest : public ::testing::Test
  {
    protected:
      void SetUp() override
      {
        ContextManager::getInstance().reset();
      }

      void TearDown() override
      {
        ContextManager::getInstance().reset();
      }
  };
}  // namespace

TEST_F(UnifiedArrayManagerTest, DefaultConstructor)
{
  UnifiedArrayManager<int> manager{};
  EXPECT_EQ(manager.size(), 0);

  {
    ContextGuard guard{Context::HOST};
    EXPECT_EQ(manager.data(false), nullptr);
    EXPECT_EQ(manager.data(true), nullptr);
  }
}

TEST_F(UnifiedArrayManagerTest, DataWithoutContext)
{
  UnifiedArrayManager<int> manager{1};
  EXPECT_NE(manager.data(false), nullptr);
  EXPECT_NE(manager.data(true), nullptr);
}

TEST_F(UnifiedArrayManagerTest, AllocatorConstructor)
{
  constexpr std::size_t N = 32;
  auto& rm = ::umpire::ResourceManager::getInstance();
  umpire::Allocator allocator = rm.getAllocator("UM");

  UnifiedArrayManager<int> manager{N, allocator};
  EXPECT_EQ(manager.size(), N);

  {
    ContextGuard guard{Context::HOST};
    int* data = manager.data(true);
    ASSERT_NE(data, nullptr);
    for (std::size_t i = 0; i < N; ++i)
    {
      EXPECT_EQ(data[i], 0);
      data[i] = static_cast<int>(i * 2);
    }
  }

  {
    ContextGuard guard{Context::HOST};
    int* data = manager.data(false);
    ASSERT_NE(data, nullptr);
    for (std::size_t i = 0; i < N; ++i)
    {
      EXPECT_EQ(data[i], static_cast<int>(i * 2));
    }
  }
}

TEST_F(UnifiedArrayManagerTest, ResizeToZero)
{
  UnifiedArrayManager<int> manager{10};
  EXPECT_EQ(manager.size(), 10);

  {
    ContextGuard guard{Context::HOST};
    EXPECT_NE(manager.data(false), nullptr);
  }

  manager.resize(0);
  EXPECT_EQ(manager.size(), 0);

  {
    ContextGuard guard{Context::HOST};
    EXPECT_EQ(manager.data(false), nullptr);
    EXPECT_EQ(manager.data(true), nullptr);
  }
}

TEST_F(UnifiedArrayManagerTest, ResizeSmaller)
{
  constexpr std::size_t N0 = 16;
  constexpr std::size_t N1 = 6;

  UnifiedArrayManager<int> manager{N0};

  {
    ContextGuard guard{Context::HOST};
    int* data = manager.data(true);
    ASSERT_NE(data, nullptr);
    for (std::size_t i = 0; i < N0; ++i)
    {
      data[i] = static_cast<int>(i);
    }
  }

  manager.resize(N1);
  EXPECT_EQ(manager.size(), N1);

  {
    ContextGuard guard{Context::HOST};
    int* data = manager.data(false);
    ASSERT_NE(data, nullptr);
    for (std::size_t i = 0; i < N1; ++i)
    {
      EXPECT_EQ(data[i], static_cast<int>(i));
    }
  }
}

TEST_F(UnifiedArrayManagerTest, ResizeLarger)
{
  constexpr std::size_t N0 = 8;
  constexpr std::size_t N1 = 16;

  UnifiedArrayManager<int> manager{N0};

  {
    ContextGuard guard{Context::HOST};
    int* data = manager.data(true);
    ASSERT_NE(data, nullptr);
    for (std::size_t i = 0; i < N0; ++i)
    {
      data[i] = static_cast<int>(i + 10);
    }
  }

  manager.resize(N1);
  EXPECT_EQ(manager.size(), N1);

  {
    ContextGuard guard{Context::HOST};
    int* data = manager.data(false);
    ASSERT_NE(data, nullptr);

    for (std::size_t i = 0; i < N0; ++i)
    {
      EXPECT_EQ(data[i], static_cast<int>(i + 10));
    }

    for (std::size_t i = N0; i < N1; ++i)
    {
      EXPECT_EQ(data[i], 0);
    }
  }
}

TEST_F(UnifiedArrayManagerTest, HostReadThenHostRead)
{
  // Set up
  constexpr std::size_t N = 128;
  UnifiedArrayManager<int> manager{N};

  // Host read
  {
    ContextGuard guard{Context::HOST};
    const int* data = manager.data(false);
    ASSERT_NE(data, nullptr);

    for (std::size_t i = 0; i < N; ++i)
    {
      EXPECT_EQ(data[i], 0);
    }
  }

  // Host read
  {
    ContextGuard guard{Context::HOST};
    const int* data = manager.data(false);
    ASSERT_NE(data, nullptr);

    for (std::size_t i = 0; i < N; ++i)
    {
      EXPECT_EQ(data[i], 0);
    }
  }
}

TEST_F(UnifiedArrayManagerTest, HostWriteThenHostRead)
{
  // Set up
  constexpr std::size_t N = 128;
  UnifiedArrayManager<int> manager{N};

  // Host write
  {
    ContextGuard guard{Context::HOST};
    int* data = manager.data(true);
    ASSERT_NE(data, nullptr);

    for (std::size_t i = 0; i < N; ++i)
    {
      data[i] = static_cast<int>(i);
    }
  }

  // Host read
  {
    ContextGuard guard{Context::HOST};
    const int* data = manager.data(false);
    ASSERT_NE(data, nullptr);

    for (std::size_t i = 0; i < N; ++i)
    {
      EXPECT_EQ(data[i], static_cast<int>(i));
    }
  }
}

TEST_F(UnifiedArrayManagerTest, HostReadThenDeviceRead)
{
  // Set up
  constexpr std::size_t N = 256;
  UnifiedArrayManager<int> manager{N};

  // Host read
  {
    ContextGuard guard{Context::HOST};
    const int* data = manager.data(false);
    ASSERT_NE(data, nullptr);

    for (std::size_t i = 0; i < N; ++i)
    {
      EXPECT_EQ(data[i], 0);
    }
  }

  // Set up
  int* out = malloc_managed<int>(N);
  ASSERT_NE(out, nullptr);

  // Device read
  {
    ContextGuard guard{Context::DEVICE};
    const int* data = manager.data(false);
    ASSERT_NE(data, nullptr);
    launch_copy(data, out, N);
  }

  // Synchronize
  device_synchronize_raw();

  // Check result
  for (std::size_t i = 0; i < N; ++i)
  {
    EXPECT_EQ(out[i], 0);
  }

  // Clean up
  free_managed(out);
}

TEST_F(UnifiedArrayManagerTest, HostWriteThenDeviceRead)
{
  // Set up
  constexpr std::size_t N = 256;
  UnifiedArrayManager<int> manager{N};

  // Host write
  {
    ContextGuard guard{Context::HOST};
    int* data = manager.data(true);
    ASSERT_NE(data, nullptr);

    for (std::size_t i = 0; i < N; ++i)
    {
      data[i] = static_cast<int>(i);
    }
  }

  // Set up
  int* out = malloc_managed<int>(N);
  ASSERT_NE(out, nullptr);

  // Device read
  {
    ContextGuard guard{Context::DEVICE};
    const int* data = manager.data(false);
    ASSERT_NE(data, nullptr);
    launch_copy(data, out, N);
  }

  // Synchronize
  device_synchronize_raw();

  // Check result
  for (std::size_t i = 0; i < N; ++i)
  {
    EXPECT_EQ(out[i], static_cast<int>(i));
  }

  // Clean up
  free_managed(out);
}

TEST_F(UnifiedArrayManagerTest, DeviceReadThenHostRead)
{
  // Set up
  constexpr std::size_t N = 256;
  UnifiedArrayManager<int> manager{N};

  int* out = malloc_managed<int>(N);
  ASSERT_NE(out, nullptr);

  // Device read
  {
    ContextGuard guard{Context::DEVICE};
    const int* data = manager.data(false);
    ASSERT_NE(data, nullptr);
    launch_copy(data, out, N);
  }

  // Host read
  {
    ContextGuard guard{Context::HOST};
    const int* data = manager.data(false);
    ASSERT_NE(data, nullptr);

    for (std::size_t i = 0; i < N; ++i)
    {
      EXPECT_EQ(data[i], 0);
    }
  }

  // Synchronize
  // Note: The host read should have caused a synchronize, but the following
  //       checks should independent of whether than part works.
  device_synchronize_raw();

  // Checks
  for (std::size_t i = 0; i < N; ++i)
  {
    EXPECT_EQ(out[i], 0);
  }

  // Clean up
  free_managed(out);
}

TEST_F(UnifiedArrayManagerTest, DeviceWriteThenHostRead)
{
  // Set up
  constexpr std::size_t N = 256;
  UnifiedArrayManager<int> manager{N};

  // Device write
  {
    ContextGuard guard{Context::DEVICE};
    int* data = manager.data(true);
    ASSERT_NE(data, nullptr);
    launch_increment(data, N);
  }

  // Host read
  {
    ContextGuard guard{Context::HOST};
    const int* data = manager.data(false);
    ASSERT_NE(data, nullptr);

    for (std::size_t i = 0; i < N; ++i)
    {
      EXPECT_EQ(data[i], 1);
    }
  }
}

TEST_F(UnifiedArrayManagerTest, DeviceReadThenDeviceRead)
{
  // Set up
  constexpr std::size_t N = 256;
  UnifiedArrayManager<int> manager{N};

  int* out0 = malloc_managed<int>(N);
  int* out1 = malloc_managed<int>(N);
  ASSERT_NE(out0, nullptr);
  ASSERT_NE(out1, nullptr);

  // Device read
  {
    ContextGuard guard{Context::DEVICE};
    const int* data = manager.data(false);
    ASSERT_NE(data, nullptr);
    launch_copy(data, out0, N);
  }

  // Device read
  {
    ContextGuard guard{Context::DEVICE};
    const int* data = manager.data(false);
    ASSERT_NE(data, nullptr);
    launch_copy(data, out1, N);
  }

  // Synchronize
  device_synchronize_raw();

  // Checks
  for (std::size_t i = 0; i < N; ++i)
  {
    EXPECT_EQ(out0[i], 0);
    EXPECT_EQ(out1[i], 0);
  }

  // Clean up
  free_managed(out0);
  free_managed(out1);
}

TEST_F(UnifiedArrayManagerTest, DeviceWriteThenDeviceRead)
{
  // Set up
  constexpr std::size_t N = 256;
  UnifiedArrayManager<int> manager{N};

  // Device write
  {
    ContextGuard guard{Context::DEVICE};
    int* data = manager.data(true);
    ASSERT_NE(data, nullptr);
    launch_increment(data, N);
  }

  // Set up
  int* out = malloc_managed<int>(N);
  ASSERT_NE(out, nullptr);

  // Device read
  {
    ContextGuard guard{Context::DEVICE};
    const int* data = manager.data(false);
    ASSERT_NE(data, nullptr);
    launch_copy(data, out, N);
  }

  // Synchronize
  device_synchronize_raw();

  // Checks
  for (std::size_t i = 0; i < N; ++i)
  {
    EXPECT_EQ(out[i], 1);
  }

  // Clean up
  free_managed(out);
}
