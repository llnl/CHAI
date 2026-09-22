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
  ExecutionSpace execution_space{NONE};
  bool device_synchronized{true};
};

thread_local ExecutionContextState execution_context_state;

#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
bool gpu_sim_mode{false};
#endif
}  // namespace

ExecutionContextManager& ExecutionContextManager::getInstance()
{
  static ExecutionContextManager instance;
  return instance;
}

ExecutionContext ExecutionContextManager::getContext() const
{
  switch (execution_context_state.execution_space) {
    case CPU:
      return ExecutionContext::HOST;
    case GPU:
      return ExecutionContext::DEVICE;
    default:
      return ExecutionContext::NONE;
  }
}

void ExecutionContextManager::setContext(ExecutionContext context)
{
  switch (context) {
    case ExecutionContext::HOST:
      setExecutionSpace(CPU);
      break;
    case ExecutionContext::DEVICE:
      setExecutionSpace(GPU);
      break;
    default:
      setExecutionSpace(NONE);
      break;
  }
}

void ExecutionContextManager::synchronize(ExecutionContext context)
{
  if (context == ExecutionContext::DEVICE) {
    syncIfNeeded();
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
  execution_context_state.execution_space = NONE;
  execution_context_state.device_synchronized = true;
}

void ExecutionContextManager::setExecutionSpace(ExecutionSpace space)
{
#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
  if (gpu_sim_mode && space != NONE) {
    space = GPU;
  }
#endif

  if (space == GPU) {
    execution_context_state.device_synchronized = false;
  }

#if defined(CHAI_THIN_GPU_ALLOCATE)
  if (space == CPU) {
    syncIfNeeded();
  }
#endif

  execution_context_state.execution_space = space;
}

ExecutionSpace ExecutionContextManager::getExecutionSpace() const
{
  return execution_context_state.execution_space;
}

bool ExecutionContextManager::syncIfNeeded()
{
  if (!execution_context_state.device_synchronized) {
    chai::synchronize();
    execution_context_state.device_synchronized = true;
    return true;
  }

  return false;
}

#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
void ExecutionContextManager::setGPUSimMode(bool enabled)
{
  gpu_sim_mode = enabled;
}

bool ExecutionContextManager::isGPUSimMode() const
{
  return gpu_sim_mode;
}
#endif
}  // namespace chai
