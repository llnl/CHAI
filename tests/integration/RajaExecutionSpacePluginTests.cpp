//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/config.hpp"
#include "chai/ArrayManager.hpp"
#include "chai/ChaiMacros.hpp"
#if defined(CHAI_ENABLE_EXPERIMENTAL)
#include "chai/expt/Context.hpp"
#include "chai/expt/ContextManager.hpp"
#endif
#include "RAJA/RAJA.hpp"
#include "gtest/gtest.h"

#define CUDA_TEST(X, Y)              \
  static void cuda_test_##X##Y();    \
  TEST(X, Y) { cuda_test_##X##Y(); } \
  static void cuda_test_##X##Y()

namespace {

/*!
 * \brief Captures the ArrayManager state when RAJA copies it.
 */
class ArrayManagerStateTester {
  public:
    ArrayManagerStateTester() = default;

    CHAI_HOST_DEVICE
    ArrayManagerStateTester(const ArrayManagerStateTester& other)
      : m_execution_space{other.m_execution_space}
    {
#if !defined(CHAI_DEVICE_COMPILE)
      auto execution_space = ::chai::ArrayManager::getInstance()->getExecutionSpace();

      if (execution_space != ::chai::NONE) {
        m_execution_space = execution_space;
      }
#endif
    }

    CHAI_HOST_DEVICE ::chai::ExecutionSpace getExecutionSpace() const {
      return m_execution_space;
    }

  private:
    ::chai::ExecutionSpace m_execution_space{::chai::NONE};
};

#if defined(CHAI_ENABLE_EXPERIMENTAL)
/*!
 * \brief Captures the ContextManager state when RAJA copies it.
 */
class ContextManagerStateTester {
  public:
    ContextManagerStateTester() = default;

    CHAI_HOST_DEVICE
    ContextManagerStateTester(const ContextManagerStateTester& other)
      : m_context{other.m_context}
    {
#if !defined(CHAI_DEVICE_COMPILE)
      auto context = ::chai::expt::ContextManager::getInstance().getContext();

      if (context != ::chai::expt::Context::NONE) {
        m_context = context;
      }
#endif
    }

    CHAI_HOST_DEVICE ::chai::expt::Context getContext() const {
      return m_context;
    }

  private:
    ::chai::expt::Context m_context{::chai::expt::Context::NONE};
};
#endif

}  // namespace

TEST(RajaExecutionSpacePlugin, ArrayManagerHOST) {
  ArrayManagerStateTester tester{};
  EXPECT_EQ(tester.getExecutionSpace(), ::chai::NONE);

  ::RAJA::forall<::RAJA::seq_exec>(
    ::RAJA::TypedRangeSegment<int>(0, 1), [=] (int) {
      EXPECT_EQ(tester.getExecutionSpace(), ::chai::CPU);
      EXPECT_EQ(::chai::ArrayManager::getInstance()->getExecutionSpace(), ::chai::NONE);
    });

  EXPECT_EQ(tester.getExecutionSpace(), ::chai::NONE);
}

#if defined(CHAI_ENABLE_CUDA)
CUDA_TEST(RajaExecutionSpacePlugin, ArrayManagerCUDA) {
  ArrayManagerStateTester tester{};
  ::chai::ExecutionSpace* result = nullptr;
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMallocManaged,
                                 (void**)&result,
                                 sizeof(::chai::ExecutionSpace));

  ::RAJA::forall<::RAJA::cuda_exec_async<256>>(
    ::RAJA::TypedRangeSegment<int>(0, 1), [=] __device__ (int) {
      *result = tester.getExecutionSpace();
    });

  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);

  EXPECT_EQ(*result, ::chai::GPU);
  EXPECT_EQ(tester.getExecutionSpace(), ::chai::NONE);

  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaFree, (void*) result);
}
#endif

#if defined(CHAI_ENABLE_HIP)
TEST(RajaExecutionSpacePlugin, ArrayManagerHIP) {
  ArrayManagerStateTester tester{};
  ::chai::ExecutionSpace* result = nullptr;
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMallocManaged,
                                (void**)&result,
                                sizeof(::chai::ExecutionSpace));

  ::RAJA::forall<::RAJA::hip_exec_async<256>>(
    ::RAJA::TypedRangeSegment<int>(0, 1), [=] __device__ (int) {
      *result = tester.getExecutionSpace();
    });

  CAMP_HIP_API_INVOKE_AND_CHECK(hipDeviceSynchronize);

  EXPECT_EQ(*result, ::chai::GPU);
  EXPECT_EQ(tester.getExecutionSpace(), ::chai::NONE);

  CAMP_HIP_API_INVOKE_AND_CHECK(hipFree, (void*) result);
}
#endif

#if defined(CHAI_ENABLE_EXPERIMENTAL)
TEST(RajaExecutionSpacePlugin, ContextManagerHOST) {
  ContextManagerStateTester tester{};
  EXPECT_EQ(tester.getContext(), ::chai::expt::Context::NONE);

  ::RAJA::forall<::RAJA::seq_exec>(
    ::RAJA::TypedRangeSegment<int>(0, 1), [=] (int) {
      EXPECT_EQ(tester.getContext(), ::chai::expt::Context::HOST);
      EXPECT_EQ(::chai::expt::ContextManager::getInstance().getContext(),
                ::chai::expt::Context::NONE);
    });

  EXPECT_EQ(tester.getContext(), ::chai::expt::Context::NONE);
}

#if defined(CHAI_ENABLE_CUDA)
CUDA_TEST(RajaExecutionSpacePlugin, ContextManagerCUDA) {
  ContextManagerStateTester tester{};
  ::chai::expt::Context* result = nullptr;
  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaMallocManaged,
                                 (void**)&result,
                                 sizeof(::chai::expt::Context));

  ::RAJA::forall<::RAJA::cuda_exec_async<256>>(
    ::RAJA::TypedRangeSegment<int>(0, 1), [=] __device__ (int) {
      *result = tester.getContext();
    });

  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);

  EXPECT_EQ(*result, ::chai::expt::Context::DEVICE);
  EXPECT_EQ(tester.getContext(), ::chai::expt::Context::NONE);

  CAMP_CUDA_API_INVOKE_AND_CHECK(cudaFree, (void*) result);
}
#endif

#if defined(CHAI_ENABLE_HIP)
TEST(RajaExecutionSpacePlugin, ContextManagerHIP) {
  ContextManagerStateTester tester{};
  ::chai::expt::Context* result = nullptr;
  CAMP_HIP_API_INVOKE_AND_CHECK(hipMallocManaged,
                                (void**)&result,
                                sizeof(::chai::expt::Context));

  ::RAJA::forall<::RAJA::hip_exec_async<256>>(
    ::RAJA::TypedRangeSegment<int>(0, 1), [=] __device__ (int) {
      *result = tester.getContext();
    });

  CAMP_HIP_API_INVOKE_AND_CHECK(hipDeviceSynchronize);

  EXPECT_EQ(*result, ::chai::expt::Context::DEVICE);
  EXPECT_EQ(tester.getContext(), ::chai::expt::Context::NONE);

  CAMP_HIP_API_INVOKE_AND_CHECK(hipFree, (void*) result);
}
#endif
#endif
