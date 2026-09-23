//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/ArrayManager.hpp"
#include "chai/ExecutionContextManager.hpp"
#include "chai/config.hpp"

#if defined(CHAI_ENABLE_EXPERIMENTAL)
#include "chai/SharedPtrManager.hpp"
#endif

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

TEST(ExecutionContextManager, DeviceContext)
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

#if defined(CHAI_ENABLE_EXPERIMENTAL)
TEST(ExecutionContextManager, SharesStateWithSharedPtrManager)
{
  auto& context_manager = chai::ExecutionContextManager::getInstance();
  auto* shared_ptr_manager = chai::expt::SharedPtrManager::getInstance();
  context_manager.reset();

  context_manager.setContext(chai::ExecutionContext::HOST);
  EXPECT_EQ(shared_ptr_manager->getExecutionSpace(), chai::CPU);

  shared_ptr_manager->setExecutionSpace(chai::GPU);
  EXPECT_EQ(context_manager.getContext(), chai::ExecutionContext::DEVICE);
  EXPECT_FALSE(context_manager.isSynchronized(chai::ExecutionContext::DEVICE));

  context_manager.setDeviceSynchronized(true);
  EXPECT_FALSE(shared_ptr_manager->syncIfNeeded());

  shared_ptr_manager->setExecutionSpace(chai::NONE);
  EXPECT_EQ(context_manager.getContext(), chai::ExecutionContext::NONE);

  context_manager.reset();
}
#endif

#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
TEST(ExecutionContextManager, GPUSimulationMapsHostToDevice)
{
  auto& manager = chai::ExecutionContextManager::getInstance();
  manager.reset();

  manager.setGPUSimMode(true);
  manager.setContext(chai::ExecutionContext::HOST);
  EXPECT_EQ(manager.getContext(), chai::ExecutionContext::DEVICE);
  EXPECT_FALSE(manager.isSynchronized(chai::ExecutionContext::DEVICE));

  manager.setContext(chai::ExecutionContext::NONE);
  EXPECT_EQ(manager.getContext(), chai::ExecutionContext::NONE);

  manager.setGPUSimMode(false);
  manager.setContext(chai::ExecutionContext::HOST);
  EXPECT_EQ(manager.getContext(), chai::ExecutionContext::HOST);

  manager.reset();
}

TEST(ExecutionContextManager, SharesGPUSimulationStateWithManagers)
{
  auto& context_manager = chai::ExecutionContextManager::getInstance();
  auto* array_manager = chai::ArrayManager::getInstance();
  context_manager.reset();

  array_manager->setGPUSimMode(true);
  EXPECT_TRUE(context_manager.isGPUSimMode());

#if defined(CHAI_ENABLE_EXPERIMENTAL)
  auto* shared_ptr_manager = chai::expt::SharedPtrManager::getInstance();
  EXPECT_TRUE(shared_ptr_manager->isGPUSimMode());

  shared_ptr_manager->setGPUSimMode(false);
  EXPECT_FALSE(array_manager->isGPUSimMode());
  shared_ptr_manager->setGPUSimMode(true);
#endif

  array_manager->setExecutionSpace(chai::CPU);
  EXPECT_EQ(context_manager.getContext(), chai::ExecutionContext::DEVICE);

  context_manager.reset();
}
#endif
