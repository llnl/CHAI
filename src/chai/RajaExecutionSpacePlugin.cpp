//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#include "chai/config.hpp"

#include "chai/RajaExecutionSpacePlugin.hpp"

#include "chai/ArrayManager.hpp"
#if defined(CHAI_ENABLE_EXPERIMENTAL)
#include "chai/expt/Context.hpp"
#include "chai/expt/ContextManager.hpp"
#endif

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
      m_arraymanager->setExecutionSpace(chai::CPU);
#if defined(CHAI_ENABLE_EXPERIMENTAL)
      chai::expt::ContextManager::getInstance().setContext(chai::expt::Context::HOST);
#endif
      break;
#if defined(CHAI_ENABLE_CUDA)
    case RAJA::Platform::cuda:
      m_arraymanager->setExecutionSpace(chai::GPU);
#if defined(CHAI_ENABLE_EXPERIMENTAL)
      chai::expt::ContextManager::getInstance().setContext(chai::expt::Context::DEVICE);
#endif
      break;
#endif
#if defined(CHAI_ENABLE_HIP)
    case RAJA::Platform::hip:
      m_arraymanager->setExecutionSpace(chai::GPU);
#if defined(CHAI_ENABLE_EXPERIMENTAL)
      chai::expt::ContextManager::getInstance().setContext(chai::expt::Context::DEVICE);
#endif
      break;
#endif
    default:
      m_arraymanager->setExecutionSpace(chai::NONE);
#if defined(CHAI_ENABLE_EXPERIMENTAL)
      chai::expt::ContextManager::getInstance().setContext(chai::expt::Context::NONE);
#endif
  }
}

void
RajaExecutionSpacePlugin::postCapture(const RAJA::util::PluginContext&)
{
  m_arraymanager->setExecutionSpace(chai::NONE);
#if defined(CHAI_ENABLE_EXPERIMENTAL)
  chai::expt::ContextManager::getInstance().setContext(chai::expt::Context::NONE);
#endif
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
     "Plugin to set CHAI manager state based on RAJA execution platform");


namespace chai {

  void linkRajaPlugin() {}

}
