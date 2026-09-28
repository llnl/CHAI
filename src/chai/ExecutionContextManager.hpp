//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CHAI_EXECUTION_CONTEXT_MANAGER_HPP
#define CHAI_EXECUTION_CONTEXT_MANAGER_HPP

#include "chai/config.hpp"
#include "chai/ExecutionContext.hpp"
#include "camp/helpers.hpp"

#if defined(CHAI_ENABLE_RAJA_PLUGIN)
#include "chai/pluginLinker.hpp"
#endif

#if defined(CHAI_ENABLE_CUDA)
#include <cuda_runtime.h>
#elif defined(CHAI_ENABLE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace chai {
  /*!
   * \brief Singleton class for managing the current context
   *        and context synchronization across the application.
   */
  class ExecutionContextManager
  {
    public:
      /*!
       * \brief Get the singleton instance.
       */
      static ExecutionContextManager& getInstance()
      {
        static ExecutionContextManager s_instance;
        return s_instance;
      }

      /*!
       * \brief Disable copy construction.
       *
       * ExecutionContextManager is a singleton and must not be copied.
       */
      ExecutionContextManager(const ExecutionContextManager&) = delete;

      /*!
       * \brief Disable copy assignment.
       *
       * ExecutionContextManager is a singleton and must not be assigned.
       */
      ExecutionContextManager& operator=(const ExecutionContextManager&) = delete;

      /*!
       * \brief Get the current context.
       */
      ExecutionContext getContext() const
      {
        return m_context;
      }

      /*!
       * \brief Set the current context.
       *
       * In GPU simulation mode, non-NONE contexts are treated as DEVICE.
       *
       * Setting the context to DEVICE marks the device as not synchronized.
       */
      void setContext(ExecutionContext context)
      {
#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
        if (m_gpu_sim_mode && context != ExecutionContext::NONE)
        {
          context = ExecutionContext::DEVICE;
        }
#endif

        m_context = context;

        if (context == ExecutionContext::DEVICE)
        {
          m_device_synchronized = false;
        }
      }

      /*!
       * \brief Synchronize the requested context (no-op if already synchronized).
       */
      void synchronize(ExecutionContext context)
      {
        if (context == ExecutionContext::DEVICE && !m_device_synchronized)
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
      bool isSynchronized(ExecutionContext context) const
      {
        return context == ExecutionContext::DEVICE ? m_device_synchronized : true;
      }

      /*!
       * \brief Explicitly set the synchronization state for the DEVICE context.
       */
      void setDeviceSynchronized(bool synchronized)
      {
        m_device_synchronized = synchronized;
      }

#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
      /*!
       * \brief Turn GPU simulation mode on or off.
       */
      void setGPUSimMode(bool gpu_sim_mode)
      {
        m_gpu_sim_mode = gpu_sim_mode;
      }

      /*!
       * \brief Return whether GPU simulation mode is active.
       */
      bool isGPUSimMode() const
      {
        return m_gpu_sim_mode;
      }
#endif

      /*!
       * \brief Reset manager state to defaults.
       */
      void reset()
      {
        m_context = ExecutionContext::NONE;
        m_device_synchronized = true;
#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
        m_gpu_sim_mode = false;
#endif
      }

    private:
      /*!
       * \brief Default constructor.
       *
       * Private to enforce singleton access via getInstance().
       */
      ExecutionContextManager() = default;

      /*!
       * \brief Current context for the application.
       *
       * Defaults to NONE until explicitly set.
       */
      ExecutionContext m_context{ExecutionContext::NONE};

      /*!
       * \brief Device synchronization state.
       *
       * True if the device context has been synchronized since the last time the
       * context was set to DEVICE.
       */
      bool m_device_synchronized{true};

#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
      /*!
       * \brief Whether host execution should simulate device execution.
       */
      bool m_gpu_sim_mode{false};
#endif
  };  // class ExecutionContextManager
}  // namespace chai

#endif  // CHAI_EXECUTION_CONTEXT_MANAGER_HPP
