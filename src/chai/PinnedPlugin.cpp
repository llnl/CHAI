//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#include "chai/config.hpp"
#include "chai/PinnedPlugin.hpp"

namespace chai {

RajaExecutionSpacePlugin::RajaExecutionSpacePlugin()
{
}

void
RajaExecutionSpacePlugin::preCapture(const RAJA::util::PluginContext& p)
{
  if (!m_arraymanager) {
    m_arraymanager = chai::ArrayManager::getInstance();
  }

  switch (p.platform) {
    case RAJA::Platform::host:
      m_arraymanager->setExecutionSpace(chai::CPU); break;
#if defined(CHAI_ENABLE_CUDA)
    case RAJA::Platform::cuda:
      m_arraymanager->setExecutionSpace(chai::GPU); break;
#endif
#if defined(CHAI_ENABLE_HIP)
    case RAJA::Platform::hip:
      m_arraymanager->setExecutionSpace(chai::GPU); break;
#endif
    default:
      m_arraymanager->setExecutionSpace(chai::NONE);
  }
}

void
RajaExecutionSpacePlugin::postCapture(const RAJA::util::PluginContext&)
{
  m_arraymanager->setExecutionSpace(chai::NONE);
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
RAJA::util::PluginRegistry::add<chai::RajaExecutionSpacePlugin> P(
     "RajaExecutionSpacePlugin",
     "Plugin to set CHAI execution space based on RAJA execution platform");


namespace chai {

  void linkRajaPlugin() {}

}

