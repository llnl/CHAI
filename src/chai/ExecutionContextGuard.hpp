//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CHAI_EXECUTION_CONTEXT_GUARD_HPP
#define CHAI_EXECUTION_CONTEXT_GUARD_HPP

#include "chai/ExecutionContext.hpp"
#include "chai/ExecutionContextManager.hpp"

namespace chai
{
  /*!
   * \brief RAII guard that temporarily sets the active execution context.
   */
  class ExecutionContextGuard
  {
    public:
      explicit ExecutionContextGuard(ExecutionContext context)
      {
        m_context_manager.setContext(context);
      }

      ~ExecutionContextGuard()
      {
        m_context_manager.setContext(m_saved_context);
      }

    private:
      ExecutionContextManager& m_context_manager{
          ExecutionContextManager::getInstance()};
      ExecutionContext m_saved_context{m_context_manager.getContext()};
  };  // class ExecutionContextGuard
}  // namespace chai

#endif  // CHAI_EXECUTION_CONTEXT_GUARD_HPP
