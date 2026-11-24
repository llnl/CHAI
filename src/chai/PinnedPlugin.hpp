//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC and CHAI
// project contributors. See the CHAI LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#if !defined(CHAI_PINNED_PLUGIN_HPP)
#define CHAI_PINNED_PLUGIN_HPP

#include "RAJA/util/PluginStrategy.hpp"

namespace chai {

class PinnedPlugin : public RAJA::util::PluginStrategy
{
  public:
    PinnedPlugin();

    void preCapture(const RAJA::util::PluginContext& p) override;

    void postCapture(const RAJA::util::PluginContext& p) override;

    void postLaunch(const RAJA::util::PluginContext& p) override;

  private:
    // Some list of callbacks
};

#endif  // CHAI_PINNED_PLUGIN_HPP
