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

/*!
 * \brief Captures both CHAI APIs' views of the execution context.
 */
class ContextRAJAPluginTester {
  public:
    /*!
     * @brief Construct a tester with an initial context of NONE.
     */
    ContextRAJAPluginTester() = default;

    /*!
     * @brief Copy-construct and capture the current ContextManager context.
     */
    CHAI_HOST_DEVICE ContextRAJAPluginTester(const ContextRAJAPluginTester& other)
      : m_context{other.m_context},
        m_execution_space{other.m_execution_space}
    {
#if !defined(CHAI_DEVICE_COMPILE)
      ::chai::expt::Context context = ::chai::expt::ContextManager::getInstance().getContext();

      if (context != ::chai::expt::Context::NONE) {
        m_context = context;
      }

      ::chai::ExecutionSpace execution_space =
        ::chai::ArrayManager::getInstance()->getExecutionSpace();
      if (execution_space != ::chai::NONE) {
        m_execution_space = execution_space;
      }
#endif
    }

    /*!
     * @brief Get the stored context.
     *
     * @return The stored ::chai::expt::Context value.
     */
    CHAI_HOST_DEVICE ::chai::expt::Context getContext() const {
      return m_context;
    }

    CHAI_HOST_DEVICE ::chai::ExecutionSpace getExecutionSpace() const {
      return m_execution_space;
    }

  private:
    /*!
     * @brief Stored context value.
     */
    ::chai::expt::Context m_context{::chai::expt::Context::NONE};
    ::chai::ExecutionSpace m_execution_space{::chai::NONE};
};

TEST(ContextRAJAPlugin, RegisteredOnce) {
  int chai_plugin_count = 0;
  for (auto plugin = ::RAJA::util::PluginRegistry::begin();
       plugin != ::RAJA::util::PluginRegistry::end(); ++plugin) {
    if (plugin->getName() == "RajaExecutionSpacePlugin" ||
        plugin->getName() == "CHAIContextPlugin") {
      ++chai_plugin_count;
    }
  }

  EXPECT_EQ(chai_plugin_count, 1);
}

// Test that the tester object got the updated context and that the current context
// is NONE inside the loop.
TEST(ContextRAJAPlugin, HOST) {
  ContextRAJAPluginTester tester{};
  EXPECT_EQ(tester.getContext(), ::chai::expt::Context::NONE);
  EXPECT_EQ(tester.getExecutionSpace(), ::chai::NONE);

  ::RAJA::forall<::RAJA::seq_exec>(::RAJA::TypedRangeSegment<int>(0, 1), [=] (int) {
    EXPECT_EQ(tester.getContext(), ::chai::expt::Context::HOST);
    EXPECT_EQ(tester.getExecutionSpace(), ::chai::CPU);
    EXPECT_EQ(::chai::expt::ContextManager::getInstance().getContext(), ::chai::expt::Context::NONE);
    EXPECT_EQ(::chai::ArrayManager::getInstance()->getExecutionSpace(), ::chai::NONE);
  });

  EXPECT_EQ(tester.getContext(), ::chai::expt::Context::NONE);
  EXPECT_EQ(tester.getExecutionSpace(), ::chai::NONE);
}

#if defined(CHAI_ENABLE_CUDA)
// Test that the tester object got the updated context.
CUDA_TEST(ContextRAJAPlugin, CUDA) {
  ContextRAJAPluginTester tester{};
  EXPECT_EQ(tester.getContext(), ::chai::expt::Context::NONE);

  ::chai::expt::Context* result = nullptr;
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMallocManaged, (void**)&result, sizeof(::chai::expt::Context));

  ::RAJA::forall<::RAJA::cuda_exec_async<256>>(::RAJA::TypedRangeSegment<int>(0, 1), [=] __device__ (int) {
    *result = tester.getContext();
  });

  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);

  EXPECT_EQ(*result, ::chai::expt::Context::DEVICE);
  EXPECT_EQ(tester.getContext(), ::chai::expt::Context::NONE);
  EXPECT_EQ(tester.getExecutionSpace(), ::chai::NONE);

  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaFree, (void*) result);
}
#endif

#if defined(CHAI_ENABLE_HIP)
// Test that the tester object got the updated context.
TEST(ContextRAJAPlugin, HIP) {
  ContextRAJAPluginTester tester{};
  EXPECT_EQ(tester.getContext(), ::chai::expt::Context::NONE);

  ::chai::expt::Context* result = nullptr;
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMallocManaged, (void**)&result, sizeof(::chai::expt::Context));

  ::RAJA::forall<::RAJA::hip_exec_async<256>>(::RAJA::TypedRangeSegment<int>(0, 1), [=] __device__ (int) {
    *result = tester.getContext();
  });

  CAMP_HIP_API_INVOKE_AND_CHECK(hipDeviceSynchronize);

  EXPECT_EQ(*result, ::chai::expt::Context::DEVICE);
  EXPECT_EQ(tester.getContext(), ::chai::expt::Context::NONE);
  EXPECT_EQ(tester.getExecutionSpace(), ::chai::NONE);

  CAMP_HIP_API_INVOKE_AND_CHECK(hipFree, (void*) result);
}
#endif
