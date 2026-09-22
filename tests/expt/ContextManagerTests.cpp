//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/config.hpp"
#include "chai/expt/ContextManager.hpp"
#include "gtest/gtest.h"

// Test that getInstance returns the same object at the same place in memory
TEST(ContextManager, SingletonInstance) {
  ::chai::expt::ContextManager& contextManager1 = ::chai::expt::ContextManager::getInstance();
  ::chai::expt::ContextManager& contextManager2 = ::chai::expt::ContextManager::getInstance();
  EXPECT_EQ(&contextManager1, &contextManager2);
}

// Test that the default context is unset
TEST(ContextManager, DefaultContext) {
  ::chai::expt::ContextManager& contextManager = ::chai::expt::ContextManager::getInstance();
  contextManager.reset();
  EXPECT_FALSE(contextManager.getContext().has_value());
  EXPECT_TRUE(contextManager.isSynchronized(::chai::expt::Context::DEVICE));
}

// Test setting the HOST context
TEST(ContextManager, HOST) {
  ::chai::expt::ContextManager& contextManager = ::chai::expt::ContextManager::getInstance();
  ::chai::expt::Context context = ::chai::expt::Context::HOST;
  contextManager.setContext(context);
  ASSERT_TRUE(contextManager.getContext().has_value());
  EXPECT_EQ(*contextManager.getContext(), context);
  EXPECT_EQ(contextManager.isSynchronized(context), true);
  contextManager.resetContext();
  EXPECT_FALSE(contextManager.getContext().has_value());
}

// Test setting the DEVICE context
TEST(ContextManager, DEVICE) {
  ::chai::expt::ContextManager& contextManager = ::chai::expt::ContextManager::getInstance();
  ::chai::expt::Context context = ::chai::expt::Context::DEVICE;
  contextManager.setContext(context);
  ASSERT_TRUE(contextManager.getContext().has_value());
  EXPECT_EQ(*contextManager.getContext(), context);
  EXPECT_EQ(contextManager.isSynchronized(context), false);

  contextManager.resetContext();
  EXPECT_FALSE(contextManager.getContext().has_value());
  EXPECT_EQ(contextManager.isSynchronized(context), false);

  contextManager.setDeviceSynchronized(true);
  EXPECT_EQ(contextManager.isSynchronized(context), true);
  contextManager.reset();
}
