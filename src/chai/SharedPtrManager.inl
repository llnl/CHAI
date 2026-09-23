//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_SharedPtrManager_INL
#define CHAI_SharedPtrManager_INL

#include "chai/config.hpp"

#include "chai/SharedPtrManager.hpp"
#include "chai/ChaiMacros.hpp"

#include <iostream>

#include "umpire/ResourceManager.hpp"

namespace chai {
namespace expt {

template<typename Ptr>
msp_pointer_record* SharedPtrManager::makeSharedPtrRecord(std::initializer_list<Ptr*> pointers,
                                                          std::initializer_list<chai::ExecutionSpace> spaces,
                                                          size_t size,
                                                          bool owned)
{
  int i = 0;
  for (Ptr* ptr : pointers) {
    if (ptr == nullptr) return &s_null_record;
    m_resource_manager.registerAllocation(ptr, 
        {ptr, size, m_allocators[spaces.begin()[i++]]->getAllocationStrategy()}
    );
  }

  Ptr* lookup_pointer = const_cast<Ptr*>(pointers.begin()[0]);

  auto pointer_record = getPointerRecord(lookup_pointer);

  if (pointer_record == &s_null_record) {
     if (lookup_pointer) {
        pointer_record = new msp_pointer_record();
     } else {
        return pointer_record;
     }
  }
  else {
     CHAI_LOG(Warning, "SharedPtrManager::makeManaged found abandoned pointer record!!!");
     //callback(pointer_record, ACTION_FOUND_ABANDONED, space);
  }

  i=0;
  for (void const* c_ptr : pointers) {
    void* ptr = const_cast<void*>(c_ptr);
    chai::ExecutionSpace space = spaces.begin()[i];

    pointer_record->m_pointers[space] = ptr;
    pointer_record->m_owned[space] = owned;
    registerPointer(pointer_record, space, owned);

    i++;
  }

  for (int space = CPU; space < NUM_EXECUTION_SPACES; ++space) {
    pointer_record->m_allocators[space] = getAllocatorId(ExecutionSpace(space));
  }

  return pointer_record;
}


#if defined(CHAI_ENABLE_PICK)
template<typename T>
CHAI_INLINE
typename SharedPtrManager::T_non_const<T> SharedPtrManager::pick(T* src_ptr, size_t index)
{
  T_non_const<T> val;
  m_resource_manager.registerAllocation(const_cast<T_non_const<T>*>(&val), umpire::util::AllocationRecord{const_cast<T_non_const<T>*>(&val), sizeof(T), m_resource_manager.getAllocator("HOST").getAllocationStrategy()});
  m_resource_manager.copy(const_cast<T_non_const<T>*>(&val), const_cast<T_non_const<T>*>(src_ptr+index), sizeof(T));
  m_resource_manager.deregisterAllocation(&val);
  return val;
}

template<typename T>
CHAI_INLINE
void SharedPtrManager::set(T* dst_ptr, size_t index, const T& val)
{
  m_resource_manager.registerAllocation(const_cast<T_non_const<T>*>(&val), umpire::util::AllocationRecord{const_cast<T_non_const<T>*>(&val), sizeof(T), m_resource_manager.getAllocator("HOST").getAllocationStrategy()});
  m_resource_manager.copy(const_cast<T_non_const<T>*>(dst_ptr+index), const_cast<T_non_const<T>*>(&val), sizeof(T));
  m_resource_manager.deregisterAllocation(const_cast<T_non_const<T>*>(&val));
}
#endif

CHAI_INLINE
void SharedPtrManager::copy(void * dst, void * src, size_t size) {
   m_resource_manager.copy(dst,src,size);
}

CHAI_INLINE
umpire::Allocator SharedPtrManager::getAllocator(ExecutionSpace space) {
   return *m_allocators[space];
}

CHAI_INLINE
void SharedPtrManager::setAllocator(ExecutionSpace space, umpire::Allocator &allocator) {
   *m_allocators[space] = allocator;
}

CHAI_INLINE
bool SharedPtrManager::syncIfNeeded() {
  auto& context_manager = ExecutionContextManager::getInstance();
  if (!context_manager.isSynchronized(ExecutionContext::DEVICE)) {
    context_manager.synchronize(ExecutionContext::DEVICE);
    return true;
  }
  return false;
}

} // end of namespace expt
} // end of namespace chai

#endif // CHAI_SharedPtrManager_INL
