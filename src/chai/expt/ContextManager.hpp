//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CHAI_CONTEXT_MANAGER_HPP
#define CHAI_CONTEXT_MANAGER_HPP

#include "chai/config.hpp"
#include "chai/ExecutionContext.hpp"
#include "chai/expt/Context.hpp"

#if defined(CHAI_ENABLE_RAJA_PLUGIN)
#include "chai/pluginLinker.hpp"
#endif

namespace chai::expt {
  /*!
   * \brief Singleton class for managing the current context
   *        and context synchronization across the application.
   */
  class ContextManager
  {
    public:
      /*!
       * \brief Get the singleton instance.
       */
      static ContextManager& getInstance()
      {
        static ContextManager s_instance;
        return s_instance;
      }

      /*!
       * \brief Disable copy construction.
       *
       * ContextManager is a singleton and must not be copied.
       */
      ContextManager(const ContextManager&) = delete;

      /*!
       * \brief Disable copy assignment.
       *
       * ContextManager is a singleton and must not be assigned.
       */
      ContextManager& operator=(const ContextManager&) = delete;

      /*!
       * \brief Get the current context.
       */
      Context getContext() const
      {
        switch (::chai::detail::getExecutionSpace())
        {
          case CPU:
            return Context::HOST;
          case GPU:
            return Context::DEVICE;
          default:
            return Context::NONE;
        }
      }

      /*!
       * \brief Set the current context.
       *
       * Setting the context to DEVICE marks the device as not synchronized.
       */
      void setContext(Context context)
      {
        switch (context)
        {
          case Context::HOST:
            ::chai::detail::setExecutionSpace(CPU);
            break;
          case Context::DEVICE:
            ::chai::detail::setExecutionSpace(GPU);
            break;
          default:
            ::chai::detail::setExecutionSpace(NONE);
            break;
        }
      }

      /*!
       * \brief Synchronize the requested context (no-op if already synchronized).
       */
      void synchronize(Context context)
      {
        if (context == Context::DEVICE) {
          ::chai::detail::syncIfNeeded();
        }
      }

      /*!
       * \brief Query whether the requested context is synchronized.
       */
      bool isSynchronized(Context context) const
      {
        return context == Context::DEVICE
          ? ::chai::detail::isDeviceSynchronized()
          : true;
      }

      /*!
       * \brief Explicitly set the synchronization state for the DEVICE context.
       */
      void setDeviceSynchronized(bool synchronized)
      {
        ::chai::detail::setDeviceSynchronized(synchronized);
      }

      /*!
       * \brief Reset manager state to defaults.
       */
      void reset()
      {
        ::chai::detail::resetExecutionContext();
      }

    private:
      /*!
       * \brief Default constructor.
       *
       * Private to enforce singleton access via getInstance().
       */
      ContextManager() = default;

  };  // class ContextManager
}  // namespace chai::expt

#endif  // CHAI_CONTEXT_MANAGER_HPP
