//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/expt/ContextGuard.hpp"
#include "gtest/gtest.h"

// Test that ContextGuard updates the current context in scope
// and restores the previous context on destruction.
TEST(ContextGuard, HOST) {
  ::chai::expt::ContextManager& contextManager = ::chai::expt::ContextManager::getInstance();
  contextManager.reset();

  {
    ::chai::expt::Context tempContext = ::chai::expt::Context::HOST;
    ::chai::expt::ContextGuard contextGuard(tempContext);
    ASSERT_TRUE(contextManager.getContext().has_value());
    EXPECT_EQ(*contextManager.getContext(), tempContext);
  }

  EXPECT_FALSE(contextManager.getContext().has_value());
}

// Test that ContextGuard updates the current context in scope
// and restores the previous context on destruction.
TEST(ContextGuard, DEVICE) {
  ::chai::expt::ContextManager& contextManager = ::chai::expt::ContextManager::getInstance();
  contextManager.reset();
  contextManager.setContext(::chai::expt::Context::HOST);

  {
    ::chai::expt::Context tempContext = ::chai::expt::Context::DEVICE;
    ::chai::expt::ContextGuard contextGuard(tempContext);
    ASSERT_TRUE(contextManager.getContext().has_value());
    EXPECT_EQ(*contextManager.getContext(), tempContext);
  }

  ASSERT_TRUE(contextManager.getContext().has_value());
  EXPECT_EQ(*contextManager.getContext(), ::chai::expt::Context::HOST);
  EXPECT_FALSE(contextManager.isSynchronized(::chai::expt::Context::DEVICE));
  contextManager.reset();
}
