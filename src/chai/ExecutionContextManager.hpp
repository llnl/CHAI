//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CHAI_EXECUTION_CONTEXT_MANAGER_HPP
#define CHAI_EXECUTION_CONTEXT_MANAGER_HPP

#include "chai/ExecutionContext.hpp"
#include "chai/Types.hpp"

namespace chai
{
  /*!
   * \brief Singleton that manages the current execution context and device
   *        synchronization state.
   *
   * The managed state is local to each host thread.
   */
  class ExecutionContextManager
  {
    public:
      /*!
       * \brief Get the singleton instance.
       */
      CHAISHAREDDLL_API static ExecutionContextManager& getInstance();

      ExecutionContextManager(const ExecutionContextManager&) = delete;
      ExecutionContextManager& operator=(const ExecutionContextManager&) = delete;

      /*!
       * \brief Get the current execution context.
       */
      CHAISHAREDDLL_API ExecutionContext getContext() const;

      /*!
       * \brief Set the current execution context.
       *
       * Setting the context to DEVICE marks the device as not synchronized.
       */
      CHAISHAREDDLL_API void setContext(ExecutionContext context);

      /*!
       * \brief Synchronize the requested context if needed.
       */
      CHAISHAREDDLL_API void synchronize(ExecutionContext context);

      /*!
       * \brief Query whether the requested context is synchronized.
       */
      CHAISHAREDDLL_API bool isSynchronized(ExecutionContext context) const;

      /*!
       * \brief Explicitly set the synchronization state for DEVICE.
       */
      CHAISHAREDDLL_API void setDeviceSynchronized(bool synchronized);

      /*!
       * \brief Reset the current thread's execution context state.
       */
      CHAISHAREDDLL_API void reset();

    private:
      ExecutionContextManager() = default;
  };  // class ExecutionContextManager
}  // namespace chai

#endif  // CHAI_EXECUTION_CONTEXT_MANAGER_HPP
