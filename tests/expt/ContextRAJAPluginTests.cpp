//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/config.hpp"
#include "chai/ChaiMacros.hpp"
#include "chai/expt/Context.hpp"
#include "chai/expt/ContextManager.hpp"
#include "chai/expt/ContextRAJAPlugin.hpp"
#include "RAJA/RAJA.hpp"
#include "gtest/gtest.h"
#include "TestHelpers.hpp"

#include <optional>

// Pre-main registration of plugin with RAJA
static ::RAJA::util::PluginRegistry::add<::chai::expt::ContextRAJAPlugin> P(
  "CHAIContextPlugin",
  "Plugin that integrates CHAI context management with RAJA.");

/*!
 * \brief Tests whether the plugin was actually called.
 */
class ContextRAJAPluginTester {
  public:
    /*!
     * @brief Construct a tester with no stored context.
     */
    ContextRAJAPluginTester() = default;

    /*!
     * @brief Copy-construct and capture the current ContextManager context.
     */
    CHAI_HOST_DEVICE ContextRAJAPluginTester(const ContextRAJAPluginTester& other)
      : m_context{other.m_context}
      , m_has_context{other.m_has_context}
    {
#if !defined(CHAI_DEVICE_COMPILE)
      std::optional<::chai::expt::Context> context =
        ::chai::expt::ContextManager::getInstance().getContext();

      if (context.has_value()) {
        m_context = *context;
        m_has_context = true;
      }
#endif
    }

    /*!
     * @brief Query whether a context was stored by this tester.
     */
    CHAI_HOST_DEVICE bool hasContext() const {
      return m_has_context;
    }

    /*!
     * @brief Get the stored context.
     *
     * @return The stored ::chai::expt::Context value.
     */
    CHAI_HOST_DEVICE ::chai::expt::Context getContext() const {
      return m_context;
    }

  private:
    /*!
     * @brief Stored context value.
     */
    ::chai::expt::Context m_context{::chai::expt::Context::HOST};

    /*!
     * @brief Whether a valid stored context is present.
     */
    bool m_has_context{false};
};

// Test that the tester object got the updated context and that the current context
// is unset inside the loop.
TEST(ContextRAJAPlugin, HOST) {
  ::chai::expt::ContextManager::getInstance().reset();
  ContextRAJAPluginTester tester{};
  EXPECT_FALSE(tester.hasContext());

  ::RAJA::forall<::RAJA::seq_exec>(::RAJA::TypedRangeSegment<int>(0, 1), [=] (int) {
    EXPECT_TRUE(tester.hasContext());
    EXPECT_EQ(tester.getContext(), ::chai::expt::Context::HOST);
    EXPECT_FALSE(::chai::expt::ContextManager::getInstance().getContext().has_value());
  });

  EXPECT_FALSE(tester.hasContext());
}

#if defined(CHAI_ENABLE_CUDA)
// Test that the tester object got the updated context.
CUDA_TEST(ContextRAJAPlugin, CUDA) {
  ::chai::expt::ContextManager::getInstance().reset();
  ContextRAJAPluginTester tester{};
  EXPECT_FALSE(tester.hasContext());

  ::chai::expt::Context* result = nullptr;
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMallocManaged, (void**)&result, sizeof(::chai::expt::Context));

  ::RAJA::forall<::RAJA::cuda_exec_async<256>>(::RAJA::TypedRangeSegment<int>(0, 1), [=] __device__ (int) {
    *result = tester.getContext();
  });

  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);

  EXPECT_EQ(*result, ::chai::expt::Context::DEVICE);
  EXPECT_FALSE(tester.hasContext());
  EXPECT_FALSE(::chai::expt::ContextManager::getInstance().getContext().has_value());
  EXPECT_FALSE(::chai::expt::ContextManager::getInstance().isSynchronized(
    ::chai::expt::Context::DEVICE));

  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaFree, (void*) result);
  ::chai::expt::ContextManager::getInstance().reset();
}
#endif

#if defined(CHAI_ENABLE_HIP)
// Test that the tester object got the updated context.
TEST(ContextRAJAPlugin, HIP) {
  ::chai::expt::ContextManager::getInstance().reset();
  ContextRAJAPluginTester tester{};
  EXPECT_FALSE(tester.hasContext());

  ::chai::expt::Context* result = nullptr;
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMallocManaged, (void**)&result, sizeof(::chai::expt::Context));

  ::RAJA::forall<::RAJA::hip_exec_async<256>>(::RAJA::TypedRangeSegment<int>(0, 1), [=] __device__ (int) {
    *result = tester.getContext();
  });

  CAMP_HIP_API_INVOKE_AND_CHECK(hipDeviceSynchronize);

  EXPECT_EQ(*result, ::chai::expt::Context::DEVICE);
  EXPECT_FALSE(tester.hasContext());
  EXPECT_FALSE(::chai::expt::ContextManager::getInstance().getContext().has_value());
  EXPECT_FALSE(::chai::expt::ContextManager::getInstance().isSynchronized(
    ::chai::expt::Context::DEVICE));

  CAMP_HIP_API_INVOKE_AND_CHECK(hipFree, (void*) result);
  ::chai::expt::ContextManager::getInstance().reset();
}
#endif
