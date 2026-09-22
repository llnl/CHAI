//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/ExecutionContextRAJAPlugin.hpp"

#include "chai/ExecutionContext.hpp"
#include "chai/ExecutionContextManager.hpp"
#include "chai/config.hpp"

namespace chai
{
void ExecutionContextRAJAPlugin::preCapture(
    const RAJA::util::PluginContext& p)
{
  ExecutionContext context = ExecutionContext::NONE;

  switch (p.platform) {
    case RAJA::Platform::host:
      context = ExecutionContext::HOST;
      break;
#if defined(CHAI_ENABLE_CUDA)
    case RAJA::Platform::cuda:
      context = ExecutionContext::DEVICE;
      break;
#endif
#if defined(CHAI_ENABLE_HIP)
    case RAJA::Platform::hip:
      context = ExecutionContext::DEVICE;
      break;
#endif
    default:
      break;
  }

  ExecutionContextManager::getInstance().setContext(context);
}

void ExecutionContextRAJAPlugin::postCapture(
    const RAJA::util::PluginContext&)
{
  ExecutionContextManager::getInstance().setContext(ExecutionContext::NONE);
}
}  // namespace chai

RAJA_INSTANTIATE_REGISTRY(RAJA::util::PluginRegistry);

// This is needed to link a dynamic lib as RAJA does not provide an exported
// definition of this symbol.
#if defined(_WIN32) && !defined(CHAISTATICLIB)
#ifdef CHAISHAREDDLL_EXPORTS
namespace RAJA
{
namespace util
{
PluginStrategy::PluginStrategy() = default;
}  // namespace util
}  // namespace RAJA
#endif
#endif

static RAJA::util::PluginRegistry::add<chai::ExecutionContextRAJAPlugin> P(
    "ExecutionContextRAJAPlugin",
    "Plugin to set the CHAI execution context from the RAJA platform");

namespace chai
{
void linkExecutionContextRAJAPlugin()
{
}
}  // namespace chai
