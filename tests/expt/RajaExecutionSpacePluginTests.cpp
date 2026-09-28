//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/config.hpp"
#include "chai/ArrayManager.hpp"
#include "chai/ChaiMacros.hpp"
#include "chai/expt/Context.hpp"
#include "chai/expt/ContextManager.hpp"
#include "RAJA/RAJA.hpp"
#include "gtest/gtest.h"
#include "TestHelpers.hpp"

namespace {

struct PluginState {
  ::chai::ExecutionSpace execution_space{::chai::NONE};
  ::chai::expt::Context context{::chai::expt::Context::NONE};
};

/*!
 * \brief Captures both CHAI manager states when RAJA copies it.
 */
class RajaExecutionSpacePluginTester {
  public:
    RajaExecutionSpacePluginTester() = default;

    CHAI_HOST_DEVICE
    RajaExecutionSpacePluginTester(const RajaExecutionSpacePluginTester& other)
      : m_state{other.m_state}
    {
#if !defined(CHAI_DEVICE_COMPILE)
      auto execution_space = ::chai::ArrayManager::getInstance()->getExecutionSpace();
      auto context = ::chai::expt::ContextManager::getInstance().getContext();

      if (execution_space != ::chai::NONE) {
        m_state.execution_space = execution_space;
      }
      if (context != ::chai::expt::Context::NONE) {
        m_state.context = context;
      }
#endif
    }

    CHAI_HOST_DEVICE PluginState getState() const { return m_state; }

  private:
    PluginState m_state{};
};

}  // namespace

TEST(RajaExecutionSpacePlugin, HOST) {
  RajaExecutionSpacePluginTester tester{};
  EXPECT_EQ(tester.getState().execution_space, ::chai::NONE);
  EXPECT_EQ(tester.getState().context, ::chai::expt::Context::NONE);

  ::RAJA::forall<::RAJA::seq_exec>(
    ::RAJA::TypedRangeSegment<int>(0, 1), [=] (int) {
      EXPECT_EQ(tester.getState().execution_space, ::chai::CPU);
      EXPECT_EQ(tester.getState().context, ::chai::expt::Context::HOST);
      EXPECT_EQ(::chai::ArrayManager::getInstance()->getExecutionSpace(), ::chai::NONE);
      EXPECT_EQ(::chai::expt::ContextManager::getInstance().getContext(),
                ::chai::expt::Context::NONE);
    });

  EXPECT_EQ(tester.getState().execution_space, ::chai::NONE);
  EXPECT_EQ(tester.getState().context, ::chai::expt::Context::NONE);
}

#if defined(CHAI_ENABLE_CUDA)
CUDA_TEST(RajaExecutionSpacePlugin, CUDA) {
  RajaExecutionSpacePluginTester tester{};
  PluginState* result = nullptr;
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMallocManaged,
                                 (void**)&result,
                                 sizeof(PluginState));

  ::RAJA::forall<::RAJA::cuda_exec_async<256>>(
    ::RAJA::TypedRangeSegment<int>(0, 1), [=] __device__ (int) {
      *result = tester.getState();
    });

  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);

  EXPECT_EQ(result->execution_space, ::chai::GPU);
  EXPECT_EQ(result->context, ::chai::expt::Context::DEVICE);
  EXPECT_EQ(tester.getState().execution_space, ::chai::NONE);
  EXPECT_EQ(tester.getState().context, ::chai::expt::Context::NONE);

  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaFree, (void*) result);
}
#endif

#if defined(CHAI_ENABLE_HIP)
TEST(RajaExecutionSpacePlugin, HIP) {
  RajaExecutionSpacePluginTester tester{};
  PluginState* result = nullptr;
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMallocManaged,
                                (void**)&result,
                                sizeof(PluginState));

  ::RAJA::forall<::RAJA::hip_exec_async<256>>(
    ::RAJA::TypedRangeSegment<int>(0, 1), [=] __device__ (int) {
      *result = tester.getState();
    });

  CAMP_HIP_API_INVOKE_AND_CHECK(hipDeviceSynchronize);

  EXPECT_EQ(result->execution_space, ::chai::GPU);
  EXPECT_EQ(result->context, ::chai::expt::Context::DEVICE);
  EXPECT_EQ(tester.getState().execution_space, ::chai::NONE);
  EXPECT_EQ(tester.getState().context, ::chai::expt::Context::NONE);

  CAMP_HIP_API_INVOKE_AND_CHECK(hipFree, (void*) result);
}
#endif
