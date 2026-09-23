//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CHAI_UNIFIED_ARRAY_MANAGER_HPP
#define CHAI_UNIFIED_ARRAY_MANAGER_HPP

#include "chai/ExecutionContext.hpp"
#include "chai/ExecutionContextManager.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/TypedAllocator.hpp"

#include <cstddef>
#include <vector>

namespace chai::expt
{
  /*!
   * \brief This class manages a unified memory array. It is designed for use
   *        with ManagedArrayPointer.
   *
   * \tparam ElementType The type of elements contained in this array.
   *
   * \note UnifiedArrayManager performs value initialization of each array element.
   *       That is to say, numeric types will be initialized to zero and nontrivial
   *       types will be default constructed. This initialization occurs on the host.
   *       In the future, this behavior may change to default initialization for
   *       performance reasons, such that numeric types will be left in an
   *       indeterminate state and nontrivial types will be default constructed.
   */
  template <typename ElementType>
  class UnifiedArrayManager {
    private:
      /*!
       * \brief Allocator used by the managed unified memory storage.
       */
      using AllocatorType = ::umpire::TypedAllocator<ElementType>;

      /*!
       * \brief Underlying contiguous unified memory storage type for managed elements.
       */
      using StorageType = std::vector<ElementType, AllocatorType>;

    public:
      /*!
       * \brief Default-constructs a UnifiedArrayManager with zero elements
       *        and a default allocator for unified memory allocations.
       */
      UnifiedArrayManager() = default;

      /*!
       * \brief Constructs a UnifiedArrayManager with zero elements
       *        and \p allocator for unified memory allocations.
       *
       * \param allocator Allocator used for unified memory allocations.
       */
      explicit UnifiedArrayManager(const umpire::Allocator& allocator)
        : m_storage{StorageType(AllocatorType(allocator))}
      {
      }

      /*!
       * \brief Constructs a UnifiedArrayManager with \p size elements
       *        using the default allocator for unified memory allocations.
       *
       * \param size Number of elements to allocate.
       */
      explicit UnifiedArrayManager(std::size_t size)
        : m_storage{StorageType(size, AllocatorType(::umpire::ResourceManager::getInstance().getAllocator("UM")))}
      {
      }

      /*!
       * \brief Constructs a UnifiedArrayManager with \p size elements
       *        using \p allocator for unified memory allocations.
       *
       * \param size Number of elements to allocate.
       * \param allocator Allocator used for unified memory allocations.
       */
      UnifiedArrayManager(std::size_t size,
                       const umpire::Allocator& allocator)
        : m_storage{StorageType(size, AllocatorType(allocator))}
      {
      }

      /*!
       * \brief Resizes the managed storage to \p new_size elements.
       *
       * \param new_size New number of elements.
       */
      void resize(std::size_t new_size)
      {
        // TODO: Investigate resize in the last modified space.
        ExecutionContext context = ExecutionContext::HOST;

        if (context != m_modified)
        {
          ExecutionContextManager::getInstance().synchronize(m_modified);
        }

        m_storage.resize(new_size);
        m_modified = context;
      }

      /*!
       * \brief Returns the number of elements currently managed.
       *
       * \return Number of elements in the managed storage.
       */
      std::size_t size() const
      {
        return m_storage.size();
      }

      /*!
       * \brief Returns a pointer to the underlying contiguous storage.
       *
       * \return Pointer to the first element, or nullptr if the storage is empty.
       */
      ElementType* data(bool touch)
      {
        ExecutionContextManager& contextManager = ExecutionContextManager::getInstance();
        ExecutionContext context = contextManager.getContext();

        if (context != m_modified)
        {
          contextManager.synchronize(m_modified);
        }

        if (touch)
        {
          m_modified = context;
        }
        else
        {
          m_modified = ExecutionContext::NONE;
        }

        return m_storage.empty() ? nullptr : m_storage.data();
      }

    private:
      /*!
       * \brief Underlying unified memory storage for the managed elements.
       */
      StorageType m_storage{AllocatorType(::umpire::ResourceManager::getInstance().getAllocator("UM"))};

      /*!
       * \brief ExecutionContext in which the managed storage was most recently modified.
       *
       * \note Used to determine when synchronization is required before accessing
       *       the underlying storage from the current context.
       */
      ExecutionContext m_modified{ExecutionContext::NONE};
  };  // class UnifiedArrayManager
}  // namespace chai::expt

#endif  // CHAI_UNIFIED_ARRAY_MANAGER_HPP
