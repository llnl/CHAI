//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/ArrayManager.hpp"
#include "chai/ExecutionContextManager.hpp"

#include "gtest/gtest.h"

TEST(ExecutionContextManager, SingletonInstance)
{
  auto& manager1 = chai::ExecutionContextManager::getInstance();
  auto& manager2 = chai::ExecutionContextManager::getInstance();

  EXPECT_EQ(&manager1, &manager2);
}

TEST(ExecutionContextManager, DefaultContext)
{
  auto& manager = chai::ExecutionContextManager::getInstance();
  manager.reset();

  EXPECT_EQ(manager.getContext(), chai::ExecutionContext::NONE);
  EXPECT_TRUE(manager.isSynchronized(chai::ExecutionContext::NONE));

  manager.reset();
}

TEST(ExecutionContextManager, HostContext)
{
  auto& manager = chai::ExecutionContextManager::getInstance();
  manager.reset();

  manager.setContext(chai::ExecutionContext::HOST);
  EXPECT_EQ(manager.getContext(), chai::ExecutionContext::HOST);
  EXPECT_TRUE(manager.isSynchronized(chai::ExecutionContext::HOST));

  manager.reset();
}

TEST(ExecutionContextManager, DeviceSynchronization)
{
  auto& manager = chai::ExecutionContextManager::getInstance();
  manager.reset();

  manager.setContext(chai::ExecutionContext::DEVICE);
  EXPECT_EQ(manager.getContext(), chai::ExecutionContext::DEVICE);
  EXPECT_FALSE(manager.isSynchronized(chai::ExecutionContext::DEVICE));

  manager.setDeviceSynchronized(true);
  EXPECT_TRUE(manager.isSynchronized(chai::ExecutionContext::DEVICE));

  manager.reset();
}

TEST(ExecutionContextManager, SharesStateWithArrayManager)
{
  auto& context_manager = chai::ExecutionContextManager::getInstance();
  auto* array_manager = chai::ArrayManager::getInstance();
  context_manager.reset();

  context_manager.setContext(chai::ExecutionContext::HOST);
  EXPECT_EQ(array_manager->getExecutionSpace(), chai::CPU);

  array_manager->setExecutionSpace(chai::GPU);
  EXPECT_EQ(context_manager.getContext(), chai::ExecutionContext::DEVICE);
  EXPECT_FALSE(context_manager.isSynchronized(chai::ExecutionContext::DEVICE));

  context_manager.setDeviceSynchronized(true);
  EXPECT_FALSE(array_manager->syncIfNeeded());

  array_manager->setExecutionSpace(chai::NONE);
  EXPECT_EQ(context_manager.getContext(), chai::ExecutionContext::NONE);

  context_manager.reset();
}
