//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/ArrayManager.hpp"
#include "chai/ChaiMacros.hpp"
#include "chai/ExecutionContext.hpp"
#include "chai/ExecutionContextManager.hpp"

#include "RAJA/RAJA.hpp"
#include "gtest/gtest.h"

#define CUDA_TEST(X, Y)                 \
  static void cuda_test_##X##_##Y();    \
  TEST(X, Y) { cuda_test_##X##_##Y(); } \
  static void cuda_test_##X##_##Y()

class ExecutionContextRAJAPluginTester
{
  public:
    ExecutionContextRAJAPluginTester() = default;

    CHAI_HOST_DEVICE ExecutionContextRAJAPluginTester(
        const ExecutionContextRAJAPluginTester& other)
      : m_context{other.m_context},
        m_execution_space{other.m_execution_space}
    {
#if !defined(CHAI_DEVICE_COMPILE)
      const auto context =
          chai::ExecutionContextManager::getInstance().getContext();
      if (context != chai::ExecutionContext::NONE) {
        m_context = context;
      }

      const auto execution_space =
          chai::ArrayManager::getInstance()->getExecutionSpace();
      if (execution_space != chai::NONE) {
        m_execution_space = execution_space;
      }
#endif
    }

    CHAI_HOST_DEVICE chai::ExecutionContext getContext() const
    {
      return m_context;
    }

    CHAI_HOST_DEVICE chai::ExecutionSpace getExecutionSpace() const
    {
      return m_execution_space;
    }

  private:
    chai::ExecutionContext m_context{chai::ExecutionContext::NONE};
    chai::ExecutionSpace m_execution_space{chai::NONE};
};

TEST(ExecutionContextRAJAPlugin, RegisteredOnce)
{
  int chai_plugin_count = 0;
  for (auto plugin = RAJA::util::PluginRegistry::begin();
       plugin != RAJA::util::PluginRegistry::end(); ++plugin) {
    if (plugin->getName() == "ExecutionContextRAJAPlugin") {
      ++chai_plugin_count;
    }
  }

  EXPECT_EQ(chai_plugin_count, 1);
}

TEST(ExecutionContextRAJAPlugin, HOST)
{
  chai::ExecutionContextManager::getInstance().reset();
  ExecutionContextRAJAPluginTester tester{};

  RAJA::forall<RAJA::seq_exec>(RAJA::TypedRangeSegment<int>(0, 1),
      [=](int) {
        EXPECT_EQ(tester.getContext(), chai::ExecutionContext::HOST);
        EXPECT_EQ(tester.getExecutionSpace(), chai::CPU);
        EXPECT_EQ(chai::ExecutionContextManager::getInstance().getContext(),
                  chai::ExecutionContext::NONE);
        EXPECT_EQ(chai::ArrayManager::getInstance()->getExecutionSpace(),
                  chai::NONE);
      });

  EXPECT_EQ(tester.getContext(), chai::ExecutionContext::NONE);
  EXPECT_EQ(tester.getExecutionSpace(), chai::NONE);
}

#if defined(CHAI_ENABLE_CUDA)
CUDA_TEST(ExecutionContextRAJAPlugin, CUDA)
{
  chai::ExecutionContextManager::getInstance().reset();
  ExecutionContextRAJAPluginTester tester{};
  chai::ExecutionContext* result = nullptr;
  CAMP_CUDA_API_INVOKE_AND_CHECK(
      cudaMallocManaged, (void**)&result, sizeof(chai::ExecutionContext));

  RAJA::forall<RAJA::cuda_exec_async<256>>(
      RAJA::TypedRangeSegment<int>(0, 1), [=] __device__(int) {
        *result = tester.getContext();
      });

  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
  EXPECT_EQ(*result, chai::ExecutionContext::DEVICE);
  EXPECT_EQ(tester.getContext(), chai::ExecutionContext::NONE);
  EXPECT_EQ(tester.getExecutionSpace(), chai::NONE);

  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaFree, (void*)result);
}
#endif

#if defined(CHAI_ENABLE_HIP)
TEST(ExecutionContextRAJAPlugin, HIP)
{
  chai::ExecutionContextManager::getInstance().reset();
  ExecutionContextRAJAPluginTester tester{};
  chai::ExecutionContext* result = nullptr;
  CAMP_HIP_API_INVOKE_AND_CHECK(
      hipMallocManaged, (void**)&result, sizeof(chai::ExecutionContext));

  RAJA::forall<RAJA::hip_exec_async<256>>(
      RAJA::TypedRangeSegment<int>(0, 1), [=] __device__(int) {
        *result = tester.getContext();
      });

  CAMP_HIP_API_INVOKE_AND_CHECK(hipDeviceSynchronize);
  EXPECT_EQ(*result, chai::ExecutionContext::DEVICE);
  EXPECT_EQ(tester.getContext(), chai::ExecutionContext::NONE);
  EXPECT_EQ(tester.getExecutionSpace(), chai::NONE);

  CAMP_HIP_API_INVOKE_AND_CHECK(hipFree, (void*)result);
}
#endif

#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
TEST(ExecutionContextRAJAPlugin, GPUSimulation)
{
  auto& context_manager = chai::ExecutionContextManager::getInstance();
  context_manager.reset();
  context_manager.setGPUSimMode(true);
  ExecutionContextRAJAPluginTester tester{};

  RAJA::forall<RAJA::seq_exec>(RAJA::TypedRangeSegment<int>(0, 1),
      [=](int) {
        EXPECT_EQ(tester.getContext(), chai::ExecutionContext::DEVICE);
        EXPECT_EQ(tester.getExecutionSpace(), chai::GPU);
      });

  EXPECT_EQ(context_manager.getContext(), chai::ExecutionContext::NONE);
  context_manager.reset();
}
#endif
