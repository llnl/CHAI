//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CHAI_CONTEXT_MANAGER_HPP
#define CHAI_CONTEXT_MANAGER_HPP

#include "chai/config.hpp"
#include "chai/expt/Context.hpp"
#include "camp/helpers.hpp"

#include <optional>

#if defined(CHAI_ENABLE_CUDA)
#include <cuda_runtime.h>
#elif defined(CHAI_ENABLE_HIP)
#include <hip/hip_runtime.h>
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
       * \brief Get the current context, if one is set.
       */
      std::optional<Context> getContext() const
      {
        return m_context;
      }

      /*!
       * \brief Set the current context.
       *
       * Setting the context to DEVICE marks the device as not synchronized.
       */
      void setContext(Context context)
      {
        m_context = context;

        if (context == Context::DEVICE)
        {
          m_device_synchronized = false;
        }
      }

      /*!
       * \brief Clear the current context without changing synchronization state.
       */
      void resetContext()
      {
        m_context.reset();
      }

      /*!
       * \brief Synchronize the requested context (no-op if already synchronized).
       */
      void synchronize(Context context)
      {
        if (context == Context::DEVICE && !m_device_synchronized)
        {
#if defined(CHAI_ENABLE_CUDA)
          CAMP_CUDA_API_INVOKE_AND_CHECK(cudaDeviceSynchronize);
#elif defined(CHAI_ENABLE_HIP)
          CAMP_HIP_API_INVOKE_AND_CHECK(hipDeviceSynchronize);
#endif
          m_device_synchronized = true;
        }
      }

      /*!
       * \brief Query whether the requested context is synchronized.
       */
      bool isSynchronized(Context context) const
      {
        return context == Context::DEVICE ? m_device_synchronized : true;
      }

      /*!
       * \brief Explicitly set the synchronization state for the DEVICE context.
       */
      void setDeviceSynchronized(bool synchronized)
      {
        m_device_synchronized = synchronized;
      }

      /*!
       * \brief Reset manager state to defaults.
       */
      void reset()
      {
        resetContext();
        m_device_synchronized = true;
      }

    private:
      /*!
       * \brief Default constructor.
       *
       * Private to enforce singleton access via getInstance().
       */
      ContextManager() = default;

      /*!
       * \brief Current context for the application.
       *
       * Defaults to unset until explicitly set.
       */
      std::optional<Context> m_context{};

      /*!
       * \brief Device synchronization state.
       *
       * True if the device context has been synchronized since the last time the
       * context was set to DEVICE.
       */
      bool m_device_synchronized{true};
  };  // class ContextManager
}  // namespace chai::expt

#endif  // CHAI_CONTEXT_MANAGER_HPP
