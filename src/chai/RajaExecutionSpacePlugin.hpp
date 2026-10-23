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

class ArrayManager;

/*!
 * \brief Plugin that sets CHAI manager state from the RAJA execution platform.
 *
 * The plugin updates ArrayManager and, when experimental features are enabled,
 * expt::ContextManager around RAJA lambda capture.
 */
class RajaExecutionSpacePlugin :
  public RAJA::util::PluginStrategy
{
  public:
    RajaExecutionSpacePlugin();

    void preCapture(const RAJA::util::PluginContext& p) override;

    void postCapture(const RAJA::util::PluginContext& p) override;

  private:
    chai::ArrayManager* m_arraymanager{nullptr};
};

void linkRajaPlugin();

}

#endif // CHAI_RajaExecutionSpacePlugin_HPP
