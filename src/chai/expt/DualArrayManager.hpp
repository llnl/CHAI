//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CHAI_DUAL_ARRAY_MANAGER_HPP
#define CHAI_DUAL_ARRAY_MANAGER_HPP

#include "chai/ExecutionContext.hpp"
#include "chai/ExecutionContextManager.hpp"
#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <type_traits>

namespace chai::expt
{
  /*!
   * \brief This class manages mirrored host/device arrays with explicit
   *        synchronization between the two copies.
   *
   * \tparam ElementType The type of elements contained in this array.
   *
   * \note DualArrayManager uses raw host/device allocations and byte-wise copies
   *       between them. As a result, ElementType must be trivially copyable,
   *       default constructible, and trivially destructible.
   */
  template <typename ElementType>
  class DualArrayManager {
    static_assert(std::is_trivially_copyable_v<ElementType>,
                  "DualArrayManager requires trivially copyable element types.");
    static_assert(std::is_default_constructible_v<ElementType>,
                  "DualArrayManager requires default-constructible element types.");
    static_assert(std::is_trivially_destructible_v<ElementType>,
                  "DualArrayManager requires trivially destructible element types.");

    public:
      /*!
       * \brief Default-constructs a DualArrayManager with zero elements
       *        and default host/device allocators.
       */
      DualArrayManager() = default;

      /*!
       * \brief Constructs a DualArrayManager with zero elements using the
       *        provided host and device allocators.
       *
       * \param host_allocator Allocator used for host memory allocations.
       * \param device_allocator Allocator used for device memory allocations.
       */
      DualArrayManager(const umpire::Allocator& host_allocator,
                       const umpire::Allocator& device_allocator)
        : m_host_allocator{host_allocator},
          m_device_allocator{device_allocator}
      {
      }

      /*!
       * \brief Constructs a DualArrayManager with \p size elements using
       *        default host/device allocators.
       *
       * \param size Number of elements to allocate when the data is first touched.
       */
      explicit DualArrayManager(std::size_t size)
        : m_size{size}
      {
      }

      /*!
       * \brief Constructs a DualArrayManager with \p size elements using the
       *        provided host and device allocators.
       *
       * \param size Number of elements to allocate when the data is first touched.
       * \param host_allocator Allocator used for host memory allocations.
       * \param device_allocator Allocator used for device memory allocations.
       */
      DualArrayManager(std::size_t size,
                       const umpire::Allocator& host_allocator,
                       const umpire::Allocator& device_allocator)
        : m_size{size},
          m_host_allocator{host_allocator},
          m_device_allocator{device_allocator}
      {
      }

      /*!
       * \brief Copy-constructs a DualArrayManager, duplicating any allocated
       *        host/device buffers.
       *
       * \param other Manager to copy from.
       */
      DualArrayManager(const DualArrayManager& other)
        : m_size{other.m_size},
          m_modified{other.m_modified},
          m_host_allocator{other.m_host_allocator},
          m_device_allocator{other.m_device_allocator}
      {
        clone(other);
      }

      /*!
       * \brief Copy-assigns a DualArrayManager, duplicating any allocated
       *        host/device buffers.
       *
       * \param other Manager to copy from.
       *
       * \return Reference to this manager.
       */
      DualArrayManager& operator=(const DualArrayManager& other)
      {
        if (&other != this)
        {
          clear();

          m_size = other.m_size;
          m_modified = other.m_modified;
          m_host_allocator = other.m_host_allocator;
          m_device_allocator = other.m_device_allocator;

          clone(other);
        }

        return *this;
      }

      /*!
       * \brief Move-constructs a DualArrayManager, transferring ownership of
       *        any allocated host/device buffers.
       *
       * \param other Manager to move from.
       */
      DualArrayManager(DualArrayManager&& other)
        : m_host_data{other.m_host_data},
          m_device_data{other.m_device_data},
          m_size{other.m_size},
          m_modified{other.m_modified},
          m_host_allocator{other.m_host_allocator},
          m_device_allocator{other.m_device_allocator}
      {
        other.m_host_data = nullptr;
        other.m_device_data = nullptr;
        other.m_size = 0;
        other.m_modified = ExecutionContext::NONE;
      }

      /*!
       * \brief Move-assigns a DualArrayManager, transferring ownership of any
       *        allocated host/device buffers.
       *
       * \param other Manager to move from.
       *
       * \return Reference to this manager.
       */
      DualArrayManager& operator=(DualArrayManager&& other)
      {
        if (&other != this)
        {
          clear();

          m_host_data = other.m_host_data;
          m_device_data = other.m_device_data;
          m_size = other.m_size;
          m_modified = other.m_modified;
          m_host_allocator = other.m_host_allocator;
          m_device_allocator = other.m_device_allocator;

          other.m_host_data = nullptr;
          other.m_device_data = nullptr;
          other.m_size = 0;
          other.m_modified = ExecutionContext::NONE;
        }

        return *this;
      }

      /*!
       * \brief Destructor.
       *
       * Deallocates any host/device buffers owned by this manager.
       */
      ~DualArrayManager()
      {
        clear();
      }

      /*!
       * \brief Resizes the managed allocation to \p new_size elements.
       *
       * Existing contents are preserved up to \c min(old_size, new_size). If
       * storage has already been allocated, the resize happens in the most
       * authoritative space and the opposite copy is discarded.
       *
       * \param new_size New number of elements.
       */
      void resize(std::size_t new_size)
      {
        if (new_size == m_size)
        {
          return;
        }

        if (new_size == 0)
        {
          clear();
          return;
        }

        if (m_host_data == nullptr && m_device_data == nullptr)
        {
          m_size = new_size;
          m_modified = ExecutionContext::NONE;
          return;
        }

        const ExecutionContext resize_context = choose_resize_context();
        const std::size_t old_size_bytes = m_size * sizeof(ElementType);
        const std::size_t new_size_bytes = new_size * sizeof(ElementType);

        ElementType*& data = pointer(resize_context);
        umpire::Allocator& allocator = getAllocator(resize_context);

        ElementType* new_data = static_cast<ElementType*>(allocator.allocate(new_size_bytes));
        valueInitialize(new_data, new_size);
        ::umpire::ResourceManager::getInstance().copy(
            new_data,
            data,
            std::min(old_size_bytes, new_size_bytes));

        deallocate(data, allocator);
        data = new_data;

        if (resize_context == ExecutionContext::HOST)
        {
          deallocate(m_device_data, m_device_allocator);
        }
        else
        {
          deallocate(m_host_data, m_host_allocator);
        }

        m_size = new_size;
        m_modified = resize_context;
      }

      /*!
       * \brief Returns the number of elements currently managed.
       *
       * \return Number of managed elements.
       */
      std::size_t size() const
      {
        return m_size;
      }

      /*!
       * \brief Returns a pointer to the managed data in the current context.
       *
       * \param touch Whether the caller intends to modify the returned data.
       *
       * \return Pointer to the managed data, or nullptr if the current context
       *         is ExecutionContext::NONE or the array is empty.
       */
      ElementType* data(bool touch)
      {
        return data(ExecutionContextManager::getInstance().getContext(), touch);
      }

    private:
      /*!
       * \brief Pointer to data in the HOST context.
       */
      ElementType* m_host_data{nullptr};

      /*!
       * \brief Pointer to data in the DEVICE context.
       */
      ElementType* m_device_data{nullptr};

      /*!
       * \brief Number of elements currently managed.
       */
      std::size_t m_size{0};

      /*!
       * \brief ExecutionContext that holds the most recently modified copy.
       */
      ExecutionContext m_modified{ExecutionContext::NONE};

      /*!
       * \brief Allocator used for host memory allocations.
       */
      umpire::Allocator m_host_allocator{::umpire::ResourceManager::getInstance().getAllocator("HOST")};

      /*!
       * \brief Allocator used for device memory allocations.
       */
      umpire::Allocator m_device_allocator{
#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)
        ::umpire::ResourceManager::getInstance().getAllocator("DEVICE")
#else
        ::umpire::ResourceManager::getInstance().getAllocator("HOST")
#endif
      };

      /*!
       * \brief Deep-copies any allocated buffers from \p other.
       *
       * \param other Manager to copy from.
       */
      void clone(const DualArrayManager& other)
      {
        const std::size_t size_bytes = other.m_size * sizeof(ElementType);

        if (size_bytes == 0)
        {
          return;
        }

        ExecutionContextManager::getInstance().synchronize(other.m_modified);

        if (other.m_host_data != nullptr)
        {
          m_host_data = static_cast<ElementType*>(m_host_allocator.allocate(size_bytes));
          ::umpire::ResourceManager::getInstance().copy(m_host_data, other.m_host_data, size_bytes);
        }

        if (other.m_device_data != nullptr)
        {
          m_device_data = static_cast<ElementType*>(m_device_allocator.allocate(size_bytes));
          ::umpire::ResourceManager::getInstance().copy(m_device_data, other.m_device_data, size_bytes);
        }
      }

      /*!
       * \brief Releases both host and device storage and resets the state.
       */
      void clear()
      {
        deallocate(m_host_data, m_host_allocator);
        deallocate(m_device_data, m_device_allocator);
        m_size = 0;
        m_modified = ExecutionContext::NONE;
      }

      /*!
       * \brief Deallocates \p data when it is non-null.
       *
       * \param data Pointer to deallocate.
       * \param allocator Allocator that owns \p data.
       */
      static void deallocate(ElementType*& data, umpire::Allocator& allocator)
      {
        if (data != nullptr)
        {
          allocator.deallocate(data);
          data = nullptr;
        }
      }

      /*!
       * \brief Returns the allocator for \p context.
       *
       * \param context Desired context.
       *
       * \return Allocator for \p context.
       */
      umpire::Allocator& getAllocator(ExecutionContext context)
      {
        return context == ExecutionContext::DEVICE ? m_device_allocator : m_host_allocator;
      }

      /*!
       * \brief Returns the storage pointer for \p context.
       *
       * \param context Desired context.
       *
       * \return Pointer reference for \p context.
       */
      ElementType*& pointer(ExecutionContext context)
      {
        return context == ExecutionContext::DEVICE ? m_device_data : m_host_data;
      }

      /*!
       * \brief Returns the storage pointer for \p context.
       *
       * \param context Desired context.
       *
       * \return Pointer for \p context.
       */
      ElementType* pointer(ExecutionContext context) const
      {
        return context == ExecutionContext::DEVICE ? m_device_data : m_host_data;
      }

      /*!
       * \brief Returns the opposite context for \p context.
       *
       * \param context ExecutionContext whose opposite is requested.
       *
       * \return HOST for DEVICE, DEVICE for HOST.
       */
      static ExecutionContext otherContext(ExecutionContext context)
      {
        return context == ExecutionContext::DEVICE ? ExecutionContext::HOST : ExecutionContext::DEVICE;
      }

      /*!
       * \brief Chooses the context whose allocation should be preserved during
       *        resize.
       *
       * \return ExecutionContext to resize in.
       */
      ExecutionContext choose_resize_context() const
      {
        if (m_modified != ExecutionContext::NONE)
        {
          return m_modified;
        }

        if (m_host_data != nullptr && m_device_data == nullptr)
        {
          return ExecutionContext::HOST;
        }

        if (m_device_data != nullptr && m_host_data == nullptr)
        {
          return ExecutionContext::DEVICE;
        }

        return ExecutionContextManager::getInstance().getContext() == ExecutionContext::DEVICE
            ? ExecutionContext::DEVICE
            : ExecutionContext::HOST;
      }

      /*!
       * \brief Fills \p destination with value-initialized elements by staging
       *        from a temporary host buffer.
       *
       * \param destination Allocation to initialize.
       * \param count Number of elements to initialize.
       */
      static void valueInitialize(ElementType* destination, std::size_t count)
      {
        if (destination == nullptr || count == 0)
        {
          return;
        }

        auto& resource_manager = ::umpire::ResourceManager::getInstance();
        umpire::Allocator host_allocator = resource_manager.getAllocator("HOST");
        ElementType* initialized = static_cast<ElementType*>(
            host_allocator.allocate(count * sizeof(ElementType)));

        try
        {
          std::uninitialized_value_construct_n(initialized, count);
          resource_manager.copy(
              destination,
              initialized,
              count * sizeof(ElementType));
        }
        catch (...)
        {
          host_allocator.deallocate(initialized);
          throw;
        }

        host_allocator.deallocate(initialized);
      }

      /*!
       * \brief Returns a pointer to the managed data in \p context, allocating
       *        and synchronizing as needed.
       *
       * \param context Desired context.
       * \param touch Whether the caller intends to modify the returned data.
       *
       * \return Pointer to data in \p context, or nullptr if \p context is
       *         ExecutionContext::NONE or the array is empty.
       */
      ElementType* data(ExecutionContext context, bool touch = true)
      {
        if (context == ExecutionContext::NONE || m_size == 0)
        {
          return nullptr;
        }

        ElementType*& destination = pointer(context);
        ElementType* source = pointer(otherContext(context));
        const bool destination_was_missing = destination == nullptr;
        const std::size_t size_bytes = m_size * sizeof(ElementType);

        if (destination_was_missing)
        {
          destination = static_cast<ElementType*>(getAllocator(context).allocate(size_bytes));
          valueInitialize(destination, m_size);
        }

        if (source != nullptr &&
            (destination_was_missing || m_modified == otherContext(context)))
        {
          ExecutionContextManager::getInstance().synchronize(m_modified);
          ::umpire::ResourceManager::getInstance().copy(destination, source, size_bytes);
        }

        if (touch)
        {
          m_modified = context;
        }
        else
        {
          m_modified = ExecutionContext::NONE;
        }

        return destination;
      }
  };  // class DualArrayManager
}  // namespace chai::expt

#endif  // CHAI_DUAL_ARRAY_MANAGER_HPP
