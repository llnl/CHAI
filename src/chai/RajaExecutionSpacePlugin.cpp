//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#include "chai/config.hpp"

#include "chai/RajaExecutionSpacePlugin.hpp"

#include "chai/detail/ExecutionContext.hpp"

namespace chai {

void
RajaExecutionSpacePlugin::preCapture(const RAJA::util::PluginContext& p)
{
  switch (p.platform) {
    case RAJA::Platform::host:
      detail::setExecutionSpace(chai::CPU); break;
#if defined(CHAI_ENABLE_CUDA)
    case RAJA::Platform::cuda:
      detail::setExecutionSpace(chai::GPU); break;
#endif
#if defined(CHAI_ENABLE_HIP)
    case RAJA::Platform::hip:
      detail::setExecutionSpace(chai::GPU); break;
#endif
    default:
      detail::setExecutionSpace(chai::NONE);
  }
}

void
RajaExecutionSpacePlugin::postCapture(const RAJA::util::PluginContext&)
{
  detail::setExecutionSpace(chai::NONE);
}

}
RAJA_INSTANTIATE_REGISTRY(RAJA::util::PluginRegistry);

// this is needed to link a dynamic lib as RAJA does not provide an exported definition of this symbol.
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

// Register plugin with RAJA
static RAJA::util::PluginRegistry::add<chai::RajaExecutionSpacePlugin> P(
     "RajaExecutionSpacePlugin",
     "Plugin to set CHAI execution space based on RAJA execution platform");


namespace chai {

  void linkRajaPlugin() {}

}
