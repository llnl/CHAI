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

namespace chai {
  /*!
   * \brief RAII guard that temporarily sets the active ExecutionContext and restores the
   *        previously active ExecutionContext upon destruction.
   */
  class ExecutionContextGuard {
    public:
      /*!
       * \brief Sets the active ExecutionContext for the lifetime of this guard.
       * \param context The ExecutionContext to set as active.
       */
      explicit ExecutionContextGuard(ExecutionContext context) {
        m_context_manager.setContext(context);
      }

      /*!
       * \brief Restores the ExecutionContext that was active when this guard was created.
       */
      ~ExecutionContextGuard() {
        m_context_manager.setContext(m_saved_context);
      }

    private:
      /*!
       * \brief Reference to the global ExecutionContextManager instance.
       */
      ExecutionContextManager& m_context_manager{ExecutionContextManager::getInstance()};

      /*!
       * ExecutionContext that was active at guard construction time.
       */
      ExecutionContext m_saved_context{m_context_manager.getContext()};
  };  // class ExecutionContextGuard
}  // namespace chai

#endif  // CHAI_EXECUTION_CONTEXT_GUARD_HPP
