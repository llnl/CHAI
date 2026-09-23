//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/config.hpp"
#include "chai/ExecutionContext.hpp"
#include "chai/ExecutionContextManager.hpp"
#include "chai/ExecutionContextRAJAPlugin.hpp"

namespace chai {
  void ExecutionContextRAJAPlugin::preCapture(const ::RAJA::util::PluginContext& p) {
    ExecutionContext context = ExecutionContext::NONE;

    switch (p.platform) {
      case ::RAJA::Platform::host:
        context = ExecutionContext::HOST;
        break;
#if defined(CHAI_ENABLE_CUDA)
      case ::RAJA::Platform::cuda:
        context = ExecutionContext::DEVICE;
        break;
#endif
#if defined(CHAI_ENABLE_HIP)
      case ::RAJA::Platform::hip:
        context = ExecutionContext::DEVICE;
        break;
#endif
      default:
        context = ExecutionContext::NONE;
        break;
    }

    ExecutionContextManager::getInstance().setContext(context);
  }

  void ExecutionContextRAJAPlugin::postCapture(const ::RAJA::util::PluginContext&) {
    ExecutionContextManager::getInstance().setContext(ExecutionContext::NONE);
  }
}  // namespace chai
