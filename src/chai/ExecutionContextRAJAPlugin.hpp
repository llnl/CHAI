//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CHAI_EXECUTION_CONTEXT_RAJA_PLUGIN_HPP
#define CHAI_EXECUTION_CONTEXT_RAJA_PLUGIN_HPP

#include "RAJA/util/PluginStrategy.hpp"

namespace chai {
  /*!
   * \brief Plugin that integrates CHAI context management with RAJA.
   *
   * CHAI data structures rely on being copy constructed in the correct context.
   * Their typical usage is to capture them by copy into a lambda expression that
   * is passed to RAJA. The lambda capture happens before the context is set, so
   * RAJA calls the `preCapture` method, which sets the current execution context.
   * Then RAJA copies the lambda, which triggers the copy constructors of the CHAI
   * data structures, making their data coherent in the current execution context.
   * Then RAJA calls the `postCapture` method, which unsets the current execution
   * context so that the CHAI data structures do not update data coherence in an
   * unexpected or unnecessary way. Finally, RAJA executes the lambda.
   */
  class ExecutionContextRAJAPlugin :
    public ::RAJA::util::PluginStrategy
  {
    public:
      /*!
       * \brief Sets the current context to match the RAJA execution context.
       * \param p RAJA plugin context.
       */
      void preCapture(const ::RAJA::util::PluginContext& p) override;

      /*!
       * \brief Resets the current context.
       * \param p RAJA plugin context for the capture.
       */
      void postCapture(const ::RAJA::util::PluginContext& p) override;
  };  // class ExecutionContextRAJAPlugin
}  // namespace chai

#endif // CHAI_EXECUTION_CONTEXT_RAJA_PLUGIN_HPP
