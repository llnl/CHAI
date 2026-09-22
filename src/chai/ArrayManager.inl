//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_ArrayManager_INL
#define CHAI_ArrayManager_INL

#include "chai/config.hpp"

#include "chai/ArrayManager.hpp"
#include "chai/ChaiMacros.hpp"

#include <iostream>

#include "umpire/ResourceManager.hpp"

#if defined(CHAI_ENABLE_UM)
#if !defined(CHAI_THIN_GPU_ALLOCATE)
#include <cuda_runtime_api.h>
#endif
#endif

namespace chai {

template<typename T>
CHAI_INLINE
void* ArrayManager::reallocate(void* pointer, size_t elems, PointerRecord* pointer_record)
{
  ExecutionSpace my_space = CPU;

  for (int iSpace = CPU; iSpace < NUM_EXECUTION_SPACES; ++iSpace) {
    if (pointer_record->m_pointers[iSpace] == pointer) {
      my_space = static_cast<ExecutionSpace>(iSpace);
      break;
    }
  }

  for (int iSpace = CPU; iSpace < NUM_EXECUTION_SPACES; ++iSpace) {
    if (!pointer_record->m_owned[iSpace]) {
      CHAI_LOG(Debug, "Cannot reallocate unowned pointer");
      return pointer_record->m_pointers[my_space];
    }
  }

  // Call callback with ACTION_FREE before changing the size
  for (int iSpace = CPU; iSpace < NUM_EXECUTION_SPACES; ++iSpace) {
    void* space_ptr = pointer_record->m_pointers[iSpace];
    int actualSpace = iSpace;
    if (space_ptr) {
#if defined(CHAI_ENABLE_UM)
      if (space_ptr == pointer_record->m_pointers[UM]) {
        actualSpace = UM;
      } else
#endif
#if defined(CHAI_ENABLE_PINNED)
      if (space_ptr == pointer_record->m_pointers[PINNED]) {
        actualSpace = PINNED;
      }
#endif
      callback(pointer_record, ACTION_FREE, ExecutionSpace(actualSpace));
      if (actualSpace == UM || actualSpace == PINNED) {
        // stop the loop over spaces
        break;
      }
    }
  }

  // Update the pointer record size
  size_t old_size = pointer_record->m_size;
  size_t new_size = sizeof(T) * elems;
  pointer_record->m_size = new_size;

  // only copy however many bytes overlap
  size_t num_bytes_to_copy = std::min(old_size, new_size);

  for (int iSpace = CPU; iSpace < NUM_EXECUTION_SPACES; ++iSpace) {
    void* space_ptr = pointer_record->m_pointers[iSpace];
    auto alloc = m_resource_manager.getAllocator(pointer_record->m_allocators[iSpace]);
    int actualSpace = iSpace;

    if (space_ptr) {
#if defined(CHAI_ENABLE_UM)
      if (space_ptr == pointer_record->m_pointers[UM]) {
        alloc = m_resource_manager.getAllocator(pointer_record->m_allocators[UM]);
        actualSpace = UM;
      } else
#endif
#if defined(CHAI_ENABLE_PINNED)
      if (space_ptr == pointer_record->m_pointers[PINNED]) {
        alloc = m_resource_manager.getAllocator(pointer_record->m_allocators[PINNED]);
        actualSpace = PINNED;
      } else
#endif
      {
        alloc = m_resource_manager.getAllocator(pointer_record->m_allocators[iSpace]);
      }
      void* new_ptr = alloc.allocate(new_size);
#if CHAI_ENABLE_ZERO_INITIALIZED_MEMORY
      m_resource_manager.memset(new_ptr, 0, new_size);
#endif
      m_resource_manager.copy(new_ptr, space_ptr, num_bytes_to_copy);
      alloc.deallocate(space_ptr);

      pointer_record->m_pointers[actualSpace] = new_ptr;
      callback(pointer_record, ACTION_ALLOC, ExecutionSpace(actualSpace));

      m_pointer_map.erase(space_ptr);
      m_pointer_map.insert(new_ptr, pointer_record);

      if (actualSpace == UM || actualSpace == PINNED) {
        for (int aliasedSpace = CPU; aliasedSpace < NUM_EXECUTION_SPACES; ++aliasedSpace) {
           if (aliasedSpace != UM && aliasedSpace != PINNED) {
              pointer_record->m_pointers[aliasedSpace] = new_ptr;
           }
        }
        // stop the loop over spaces
        break;
      }
    }
  }

  return pointer_record->m_pointers[my_space];
}

template<typename T>
CHAI_INLINE
typename ArrayManager::T_non_const<T> ArrayManager::pick(T* src_ptr, size_t index)
{
  T_non_const<T> val;
  m_resource_manager.registerAllocation(const_cast<T_non_const<T>*>(&val), umpire::util::AllocationRecord{const_cast<T_non_const<T>*>(&val), sizeof(T), m_resource_manager.getAllocator("HOST").getAllocationStrategy()});
  m_resource_manager.copy(const_cast<T_non_const<T>*>(&val), const_cast<T_non_const<T>*>(src_ptr+index), sizeof(T));
  m_resource_manager.deregisterAllocation(&val);
  return val;
}

template<typename T>
CHAI_INLINE
void ArrayManager::set(T* dst_ptr, size_t index, const T& val)
{
  m_resource_manager.registerAllocation(const_cast<T_non_const<T>*>(&val), umpire::util::AllocationRecord{const_cast<T_non_const<T>*>(&val), sizeof(T), m_resource_manager.getAllocator("HOST").getAllocationStrategy()});
  m_resource_manager.copy(const_cast<T_non_const<T>*>(dst_ptr+index), const_cast<T_non_const<T>*>(&val), sizeof(T));
  m_resource_manager.deregisterAllocation(const_cast<T_non_const<T>*>(&val));
}

CHAI_INLINE
void ArrayManager::copy(void * dst, void * src, size_t size) {
   m_resource_manager.copy(dst,src,size);
}

CHAI_INLINE
umpire::Allocator ArrayManager::getAllocator(ExecutionSpace space) {
   return *m_allocators[space];
}

CHAI_INLINE
umpire::Allocator ArrayManager::getAllocator(int allocator_id) {
   return m_resource_manager.getAllocator(allocator_id);
}

CHAI_INLINE
void ArrayManager::setAllocator(ExecutionSpace space, umpire::Allocator &allocator) {
   *m_allocators[space] = allocator;
}

CHAI_INLINE
bool ArrayManager::syncIfNeeded() {
  return detail::syncIfNeeded();
}
} // end of namespace chai

#endif // CHAI_ArrayManager_INL
