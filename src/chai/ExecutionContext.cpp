//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/detail/ExecutionContext.hpp"

#include "chai/DeviceHelpers.hpp"

#if defined(CHAI_ENABLE_RAJA_PLUGIN)
#include "chai/pluginLinker.hpp"
#endif

namespace chai
{
namespace detail
{
namespace
{

thread_local ExecutionSpace execution_space = NONE;
thread_local bool device_synchronized = true;

#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
bool gpu_sim_mode = false;
#endif

}  // namespace

bool syncIfNeeded()
{
  if (!device_synchronized) {
    synchronize();
    device_synchronized = true;
    return true;
  }

  return false;
}

void setExecutionSpace(ExecutionSpace space)
{
#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
  if (gpu_sim_mode && space != NONE) {
    space = GPU;
  }
#endif

  if (space == GPU) {
    device_synchronized = false;
  }

#if defined(CHAI_THIN_GPU_ALLOCATE)
  if (space == CPU) {
    syncIfNeeded();
  }
#endif

  execution_space = space;
}

ExecutionSpace getExecutionSpace()
{
  return execution_space;
}

bool isDeviceSynchronized()
{
  return device_synchronized;
}

void setDeviceSynchronized(bool synchronized)
{
  device_synchronized = synchronized;
}

void resetExecutionContext()
{
  execution_space = NONE;
  device_synchronized = true;
}

#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
void setGPUSimMode(bool enabled)
{
  gpu_sim_mode = enabled;
}

bool isGPUSimMode()
{
  return gpu_sim_mode;
}
#endif

}  // namespace detail
}  // namespace chai
