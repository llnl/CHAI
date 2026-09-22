//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_detail_ExecutionContext_HPP
#define CHAI_detail_ExecutionContext_HPP

#include "chai/Types.hpp"

namespace chai
{
namespace detail
{

// Internal accessors for CHAI's shared execution context. Per-thread state has
// a single definition in ExecutionContext.cpp.
CHAISHAREDDLL_API bool syncIfNeeded();

CHAISHAREDDLL_API void setExecutionSpace(ExecutionSpace space);

CHAISHAREDDLL_API ExecutionSpace getExecutionSpace();

CHAISHAREDDLL_API bool isDeviceSynchronized();

CHAISHAREDDLL_API void setDeviceSynchronized(bool synchronized);

CHAISHAREDDLL_API void resetExecutionContext();

#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
CHAISHAREDDLL_API void setGPUSimMode(bool enabled);

CHAISHAREDDLL_API bool isGPUSimMode();
#endif

}  // namespace detail
}  // namespace chai

#endif  // CHAI_detail_ExecutionContext_HPP
