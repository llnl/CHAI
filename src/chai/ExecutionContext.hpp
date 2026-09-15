//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_ExecutionContext_HPP
#define CHAI_ExecutionContext_HPP

#include "chai/DeviceHelpers.hpp"
#include "chai/ExecutionSpaces.hpp"

namespace chai
{
namespace detail
{

inline thread_local ExecutionSpace execution_space = NONE;
inline thread_local bool device_synchronized = true;

#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
inline bool gpu_sim_mode = false;
#endif

inline bool syncIfNeeded()
{
  if (!device_synchronized) {
    synchronize();
    device_synchronized = true;
    return true;
  }

  return false;
}

inline void setExecutionSpace(ExecutionSpace space)
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

inline ExecutionSpace getExecutionSpace()
{
  return execution_space;
}

inline bool isDeviceSynchronized()
{
  return device_synchronized;
}

inline void setDeviceSynchronized(bool synchronized)
{
  device_synchronized = synchronized;
}

inline void resetExecutionContext()
{
  execution_space = NONE;
  device_synchronized = true;
}

#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
inline void setGPUSimMode(bool enabled)
{
  gpu_sim_mode = enabled;
}

inline bool isGPUSimMode()
{
  return gpu_sim_mode;
}
#endif

}  // namespace detail
}  // namespace chai

#endif  // CHAI_ExecutionContext_HPP
