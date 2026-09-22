//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/config.hpp"
#include "chai/ArrayManager.hpp"
#include "chai/expt/ContextManager.hpp"
#include "gtest/gtest.h"

// Test that getInstance returns the same object at the same place in memory
TEST(ContextManager, SingletonInstance) {
  ::chai::expt::ContextManager& contextManager1 = ::chai::expt::ContextManager::getInstance();
  ::chai::expt::ContextManager& contextManager2 = ::chai::expt::ContextManager::getInstance();
  EXPECT_EQ(&contextManager1, &contextManager2);
}

// Test that the default context is NONE
TEST(ContextManager, DefaultContext) {
  ::chai::expt::ContextManager& contextManager = ::chai::expt::ContextManager::getInstance();
  contextManager.reset();
  EXPECT_EQ(contextManager.getContext(), ::chai::expt::Context::NONE);
}

// Test setting the HOST context
TEST(ContextManager, HOST) {
  ::chai::expt::ContextManager& contextManager = ::chai::expt::ContextManager::getInstance();
  contextManager.reset();
  ::chai::expt::Context context = ::chai::expt::Context::HOST;
  contextManager.setContext(context);
  EXPECT_EQ(contextManager.getContext(), context);
  EXPECT_EQ(contextManager.isSynchronized(context), true);
  contextManager.setContext(::chai::expt::Context::NONE);
}

// Test setting the DEVICE context
TEST(ContextManager, DEVICE) {
  ::chai::expt::ContextManager& contextManager = ::chai::expt::ContextManager::getInstance();
  contextManager.reset();
  ::chai::expt::Context context = ::chai::expt::Context::DEVICE;
  contextManager.setContext(context);
  EXPECT_EQ(contextManager.getContext(), context);
  EXPECT_EQ(contextManager.isSynchronized(context), false);
  contextManager.setDeviceSynchronized(true);
  EXPECT_EQ(contextManager.isSynchronized(context), true);
  contextManager.setContext(::chai::expt::Context::NONE);
}

TEST(ContextManager, SharesExecutionSpaceWithArrayManager) {
  auto& contextManager = ::chai::expt::ContextManager::getInstance();
  auto* arrayManager = ::chai::ArrayManager::getInstance();
  contextManager.reset();

  contextManager.setContext(::chai::expt::Context::HOST);
  EXPECT_EQ(arrayManager->getExecutionSpace(), ::chai::CPU);

  arrayManager->setExecutionSpace(::chai::GPU);
  EXPECT_EQ(contextManager.getContext(), ::chai::expt::Context::DEVICE);

  contextManager.setContext(::chai::expt::Context::DEVICE);
  EXPECT_EQ(arrayManager->getExecutionSpace(), ::chai::GPU);

  arrayManager->setExecutionSpace(::chai::NONE);
  EXPECT_EQ(contextManager.getContext(), ::chai::expt::Context::NONE);
}

TEST(ContextManager, SharesSynchronizationStateWithArrayManager) {
  auto& contextManager = ::chai::expt::ContextManager::getInstance();
  auto* arrayManager = ::chai::ArrayManager::getInstance();
  contextManager.reset();

  arrayManager->setExecutionSpace(::chai::GPU);
  EXPECT_FALSE(contextManager.isSynchronized(::chai::expt::Context::DEVICE));

  contextManager.setDeviceSynchronized(true);
  EXPECT_FALSE(arrayManager->syncIfNeeded());

  contextManager.setContext(::chai::expt::Context::DEVICE);
  EXPECT_FALSE(contextManager.isSynchronized(::chai::expt::Context::DEVICE));
  contextManager.setDeviceSynchronized(true);
  EXPECT_FALSE(arrayManager->syncIfNeeded());
}
