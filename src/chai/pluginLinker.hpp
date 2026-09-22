//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_pluginLinker_HPP
#define CHAI_pluginLinker_HPP

#include "chai/ExecutionContextRAJAPlugin.hpp"

namespace {
  namespace anonymous_chai {
    struct pluginLinker {
      pluginLinker() {
        (void) chai::linkExecutionContextRAJAPlugin();
      }
    } pluginLinker;
  }
}

#endif // CHAI_pluginLinker_HPP
