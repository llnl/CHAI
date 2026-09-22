//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/ExecutionContextManager.hpp"

#include "chai/DeviceHelpers.hpp"
#include "chai/config.hpp"

#if defined(CHAI_ENABLE_RAJA_PLUGIN)
#include "chai/pluginLinker.hpp"
#endif

namespace chai
{
namespace
{
struct ExecutionContextState
{
  ExecutionContext context{ExecutionContext::NONE};
  bool device_synchronized{true};
};

thread_local ExecutionContextState execution_context_state;
}  // namespace

ExecutionContextManager& ExecutionContextManager::getInstance()
{
  static ExecutionContextManager instance;
  return instance;
}

ExecutionContext ExecutionContextManager::getContext() const
{
  return execution_context_state.context;
}

void ExecutionContextManager::setContext(ExecutionContext context)
{
  execution_context_state.context = context;

  if (context == ExecutionContext::DEVICE) {
    execution_context_state.device_synchronized = false;
  }
}

void ExecutionContextManager::synchronize(ExecutionContext context)
{
  if (context == ExecutionContext::DEVICE &&
      !execution_context_state.device_synchronized) {
    chai::synchronize();
    execution_context_state.device_synchronized = true;
  }
}

bool ExecutionContextManager::isSynchronized(ExecutionContext context) const
{
  return context == ExecutionContext::DEVICE
      ? execution_context_state.device_synchronized
      : true;
}

void ExecutionContextManager::setDeviceSynchronized(bool synchronized)
{
  execution_context_state.device_synchronized = synchronized;
}

void ExecutionContextManager::reset()
{
  execution_context_state.context = ExecutionContext::NONE;
  execution_context_state.device_synchronized = true;
}
}  // namespace chai
