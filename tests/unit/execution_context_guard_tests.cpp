//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/ExecutionContextGuard.hpp"

#include "gtest/gtest.h"

TEST(ExecutionContextGuard, RestoresNestedContexts)
{
  auto& manager = chai::ExecutionContextManager::getInstance();
  manager.reset();

  {
    chai::ExecutionContextGuard host_guard{chai::ExecutionContext::HOST};
    EXPECT_EQ(manager.getContext(), chai::ExecutionContext::HOST);

    {
      chai::ExecutionContextGuard device_guard{chai::ExecutionContext::DEVICE};
      EXPECT_EQ(manager.getContext(), chai::ExecutionContext::DEVICE);
    }

    EXPECT_EQ(manager.getContext(), chai::ExecutionContext::HOST);
  }

  EXPECT_EQ(manager.getContext(), chai::ExecutionContext::NONE);
}
