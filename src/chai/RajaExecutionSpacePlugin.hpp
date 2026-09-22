//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_RajaExecutionSpacePlugin_HPP
#define CHAI_RajaExecutionSpacePlugin_HPP

#include "RAJA/util/PluginStrategy.hpp"

namespace chai {

class RajaExecutionSpacePlugin :
  public RAJA::util::PluginStrategy
{
  public:
    RajaExecutionSpacePlugin() = default;

    void preCapture(const RAJA::util::PluginContext& p) override;

    void postCapture(const RAJA::util::PluginContext& p) override;
};

void linkRajaPlugin();

}

#endif // CHAI_RajaExecutionSpacePlugin_HPP
