//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CHAI_EXECUTION_CONTEXT_RAJA_PLUGIN_HPP
#define CHAI_EXECUTION_CONTEXT_RAJA_PLUGIN_HPP

#include "chai/Types.hpp"

#include "RAJA/util/PluginStrategy.hpp"

namespace chai
{
  /*!
   * \brief Plugin that integrates CHAI execution contexts with RAJA.
   */
  class CHAISHAREDDLL_API ExecutionContextRAJAPlugin
    : public RAJA::util::PluginStrategy
  {
    public:
      ExecutionContextRAJAPlugin() = default;

      void preCapture(const RAJA::util::PluginContext& p) override;
      void postCapture(const RAJA::util::PluginContext& p) override;
  };  // class ExecutionContextRAJAPlugin

  CHAISHAREDDLL_API void linkExecutionContextRAJAPlugin();
}  // namespace chai

#endif  // CHAI_EXECUTION_CONTEXT_RAJA_PLUGIN_HPP
