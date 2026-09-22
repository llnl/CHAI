//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/config.hpp"
#include "chai/expt/Context.hpp"
#include "chai/expt/ContextManager.hpp"
#include "chai/expt/ContextRAJAPlugin.hpp"

#include <optional>

namespace chai::expt {
  void ContextRAJAPlugin::preCapture(const ::RAJA::util::PluginContext& p) {
    std::optional<Context> context{};

    switch (p.platform) {
      case ::RAJA::Platform::host:
        context = Context::HOST;
        break;
#if defined(CHAI_ENABLE_CUDA)
      case ::RAJA::Platform::cuda:
        context = Context::DEVICE;
        break;
#endif
#if defined(CHAI_ENABLE_HIP)
      case ::RAJA::Platform::hip:
        context = Context::DEVICE;
        break;
#endif
      default:
        break;
    }

    if (context.has_value())
    {
      ContextManager::getInstance().setContext(*context);
    }
    else
    {
      ContextManager::getInstance().resetContext();
    }
  }

  void ContextRAJAPlugin::postCapture(const ::RAJA::util::PluginContext&) {
    ContextManager::getInstance().resetContext();
  }
}  // namespace chai::expt
