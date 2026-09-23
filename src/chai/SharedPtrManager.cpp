//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#include "chai/SharedPtrManager.hpp"
#include <initializer_list>

#include "chai/ExecutionSpaces.hpp"
#include "chai/config.hpp"

#if defined(CHAI_ENABLE_CUDA)
#if !defined(CHAI_THIN_GPU_ALLOCATE)
#include "cuda_runtime_api.h"
#endif
#endif

#include "umpire/ResourceManager.hpp"

namespace chai
{
namespace expt
{

msp_pointer_record SharedPtrManager::s_null_record = msp_pointer_record();

SharedPtrManager* SharedPtrManager::getInstance()
{
  static SharedPtrManager s_resource_manager_instance;
  return &s_resource_manager_instance;
}

SharedPtrManager::SharedPtrManager() :
  m_pointer_map{},
  m_allocators{},
  m_resource_manager{umpire::ResourceManager::getInstance()}
 //,m_callbacks_active{true}
{
  m_pointer_map.clear();
  m_default_allocation_space = CPU;

  m_allocators[CPU] =
      new umpire::Allocator(m_resource_manager.getAllocator("HOST"));

#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
  m_allocators[GPU] =
      new umpire::Allocator(m_resource_manager.getAllocator("HOST"));
#else
  m_allocators[GPU] =
      new umpire::Allocator(m_resource_manager.getAllocator("DEVICE"));
#endif
#endif

#if defined(CHAI_ENABLE_UM)
  m_allocators[UM] =
      new umpire::Allocator(m_resource_manager.getAllocator("UM"));
#endif

#if defined(CHAI_ENABLE_PINNED)
#if (defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP)) && !defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
    m_allocators[PINNED] =
             new umpire::Allocator(m_resource_manager.getAllocator("PINNED"));
#else
  m_allocators[PINNED] =
      new umpire::Allocator(m_resource_manager.getAllocator("HOST"));
#endif
#endif
}

void SharedPtrManager::registerPointer(
   msp_pointer_record* record,
   ExecutionSpace space,
   bool owned)
{
  std::lock_guard<std::mutex> lock(m_mutex);
  auto pointer = record->m_pointers[space];

  // if we are registering a new pointer record for a pointer where there is already
  // a pointer record, we assume the old record was somehow abandoned by the host
  // application and trigger an ACTION_FOUND_ABANDONED callback
  auto found_pointer_record_pair = m_pointer_map.find(pointer);
  if (found_pointer_record_pair != m_pointer_map.end()) {
     msp_pointer_record ** found_pointer_record_addr = found_pointer_record_pair->second;
     if (found_pointer_record_addr != nullptr) {

        msp_pointer_record *foundRecord = *found_pointer_record_addr;
        // if it's actually the same pointer record, then we're OK. If it's a different
        // one, delete the old one.
        if (foundRecord != record) {
           CHAI_LOG(Warning, "SharedPtrManager::registerPointer found a record for " <<
                      pointer << " already there.  Deleting abandoned pointer record.");

           //callback(foundRecord, ACTION_FOUND_ABANDONED, space);

           for (int fspace = CPU; fspace < NUM_EXECUTION_SPACES; ++fspace) {
              foundRecord->m_pointers[fspace] = nullptr;
           }

           delete foundRecord;
        }
     }
  }

  CHAI_LOG(Debug, "Registering " << pointer << " in space " << space);

  m_pointer_map.insert(pointer, record);

  for (int i = 0; i < NUM_EXECUTION_SPACES; i++) {
    if (!record->m_pointers[i]) record->m_owned[i] = true;
  }
  record->m_owned[space] = owned;

  if (pointer) {
     // register with umpire if it's not there so that umpire can perform data migrations
     if (!m_resource_manager.hasAllocator(pointer)) {
        umpire::util::AllocationRecord new_allocation_record;
        new_allocation_record.ptr = pointer;
        new_allocation_record.strategy = m_resource_manager.getAllocator(record->m_allocators[space]).getAllocationStrategy();

        m_resource_manager.registerAllocation(pointer, new_allocation_record);
     }
  }
}

void SharedPtrManager::deregisterPointer(msp_pointer_record* record, bool deregisterFromUmpire)
{
  std::lock_guard<std::mutex> lock(m_mutex);
  for (int i = 0; i < NUM_EXECUTION_SPACES; i++) {
    void * pointer = record->m_pointers[i];
    if (pointer) {
       if (deregisterFromUmpire) {
          m_resource_manager.deregisterAllocation(pointer);
       }
       CHAI_LOG(Debug, "De-registering " << pointer);
       m_pointer_map.erase(pointer);
    }
  }
  if (record != &s_null_record) {
     delete record;
  }
}

void SharedPtrManager::setExecutionSpace(ExecutionSpace space)
{
#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
  if (isGPUSimMode() && space != NONE) {
    space = GPU;
  }
#endif

  CHAI_LOG(Debug, "Setting execution space to " << space);

#if defined(CHAI_THIN_GPU_ALLOCATE)
  if (space == CPU) {
    syncIfNeeded();
  }
#endif

  ExecutionContextManager::getInstance().setContext(
      toExecutionContext(space));
}

void* SharedPtrManager::move(void* pointer,
                         msp_pointer_record* pointer_record,
                         ExecutionSpace space, bool poly)
{
  // Check for default arg (NONE)
  if (space == NONE) {
    space = getExecutionSpace();
  }

  if (space == NONE) {
    return pointer;
  }

  move(pointer_record, space, poly);

  return pointer_record->m_pointers[space];
}

ExecutionSpace SharedPtrManager::getExecutionSpace()
{
  return toExecutionSpace(
      ExecutionContextManager::getInstance().getContext());
}

void SharedPtrManager::registerTouch(msp_pointer_record* pointer_record)
{
  registerTouch(pointer_record, getExecutionSpace());
}

void SharedPtrManager::registerTouch(msp_pointer_record* pointer_record,
                                 ExecutionSpace space)
{
  if (pointer_record && pointer_record != &s_null_record) {

     if (space != NONE) {
       CHAI_LOG(Debug, pointer_record->m_pointers[space] << " touched in space " << space);
       pointer_record->m_touched[space] = true;
       pointer_record->m_last_space = space;
     }
  }
}


void SharedPtrManager::resetTouch(msp_pointer_record* pointer_record)
{
  if (pointer_record && pointer_record!= &s_null_record) {
    for (int space = CPU; space < NUM_EXECUTION_SPACES; ++space) {
      pointer_record->m_touched[space] = false;
    }
  }
}


/* Not all GPU platform runtimes (notably HIP), will give you asynchronous copies to the device by default, so we leverage
 * umpire's API for asynchronous copies using camp resources in this method, based off of the CHAI destination space
 * */
static void copy(void * dst_pointer, void * src_pointer, umpire::ResourceManager & manager, ExecutionSpace dst_space, ExecutionSpace src_space, bool poly=false) {

#ifdef CHAI_ENABLE_CUDA
   camp::resources::Resource device_resource(camp::resources::Cuda::get_default());
#elif defined(CHAI_ENABLE_HIP)
   camp::resources::Resource device_resource(camp::resources::Hip::get_default());
#else
   camp::resources::Resource device_resource(camp::resources::Host::get_default());
#endif


  camp::resources::Resource host_resource(camp::resources::Host::get_default());
  if (dst_space == GPU || src_space == GPU) {
    // Do the copy using the device resource
    
    if (poly) {
      std::size_t vtable_size = sizeof(void*); 
      void* poly_src_ptr = ((char*)src_pointer + vtable_size);
      void* poly_dst_ptr = ((char*)dst_pointer + vtable_size);
      manager.copy(poly_dst_ptr, poly_src_ptr, device_resource);
    } else {
      manager.copy(dst_pointer, src_pointer, device_resource);
    }

  } else {
    // Do the copy using the host resource
    manager.copy(dst_pointer, src_pointer, host_resource);
  }
  // Ensure device to host copies are synchronous
  if (dst_space == CPU && src_space == GPU) {
    device_resource.wait();
  }
}

void SharedPtrManager::move(msp_pointer_record* record, ExecutionSpace space, bool poly)
{
  if (space == NONE) {
    return;
  }

  //callback(record, ACTION_CAPTURED, space);

  if (space == record->m_last_space) {
    return;
  }

  ExecutionSpace prev_space = record->m_last_space;

  void* src_pointer = record->m_pointers[prev_space];
  void* dst_pointer = record->m_pointers[space];

  if ( (!record->m_touched[record->m_last_space]) || (! src_pointer )) {
    return;
  } else if (dst_pointer != src_pointer) {
    // Exclude the copy if src and dst are the same (can happen for PINNED memory)
    {
      chai::expt::copy(dst_pointer, src_pointer, m_resource_manager, space, prev_space, poly);
    }

    //callback(record, ACTION_MOVE, space);
  }

  resetTouch(record);
}

void SharedPtrManager::allocate(
    msp_pointer_record* pointer_record,
           ExecutionSpace space)
{
  auto alloc = m_resource_manager.getAllocator(pointer_record->m_allocators[space]);

  pointer_record->m_pointers[space] = alloc.allocate(1);
  //callback(pointer_record, ACTION_ALLOC, space);
  registerPointer(pointer_record, space);

  CHAI_LOG(Debug, "Allocated array at: " << pointer_record->m_pointers[space]);
}

void SharedPtrManager::free(msp_pointer_record* pointer_record, ExecutionSpace spaceToFree)
{
  if (!pointer_record) return;

  for (int space = CPU; space < NUM_EXECUTION_SPACES; ++space) {
    if (space == spaceToFree || spaceToFree == NONE) {
      if (pointer_record->m_pointers[space]) {
        void* space_ptr = pointer_record->m_pointers[space];
        if (pointer_record->m_owned[space]) {
#if defined(CHAI_ENABLE_UM)
          if (space_ptr == pointer_record->m_pointers[UM]) {
            //callback(pointer_record,
            //         ACTION_FREE,
            //         ExecutionSpace(UM));

            auto alloc = m_resource_manager.getAllocator(pointer_record->m_allocators[UM]);
            alloc.deallocate(space_ptr);

            for (int space_t = CPU; space_t < NUM_EXECUTION_SPACES; ++space_t) {
              if (space_ptr == pointer_record->m_pointers[space_t]) {
                pointer_record->m_pointers[space_t] = nullptr;
              }
            }
          } else
#endif
#if defined(CHAI_ENABLE_PINNED)
          if (space_ptr == pointer_record->m_pointers[PINNED]) {
            //callback(pointer_record,
            //         ACTION_FREE,
            //         ExecutionSpace(PINNED));

            auto alloc = m_resource_manager.getAllocator(
                pointer_record->m_allocators[PINNED]);
            alloc.deallocate(space_ptr);

            for (int space_t = CPU; space_t < NUM_EXECUTION_SPACES; ++space_t) {
              if (space_ptr == pointer_record->m_pointers[space_t]) {
                pointer_record->m_pointers[space_t] = nullptr;
              }
            }
          } else
#endif
          {
   //         callback(pointer_record,
   //                  ACTION_FREE,
   //                  ExecutionSpace(space));

            auto alloc = m_resource_manager.getAllocator(
                pointer_record->m_allocators[space]);
            alloc.deallocate(space_ptr);

            pointer_record->m_pointers[space] = nullptr;
          }
        }
        else
        {
          m_resource_manager.deregisterAllocation(space_ptr);
        }
        {
          CHAI_LOG(Debug, "DeRegistering " << space_ptr);
          std::lock_guard<std::mutex> lock(m_mutex);
          m_pointer_map.erase(space_ptr);
        }
      }
    }
  }
  
  if (pointer_record != &s_null_record && spaceToFree == NONE) {
    delete pointer_record;
  }
}


void SharedPtrManager::setDefaultAllocationSpace(ExecutionSpace space)
{
  m_default_allocation_space = space;
}

ExecutionSpace SharedPtrManager::getDefaultAllocationSpace()
{
  return m_default_allocation_space;
}


//void SharedPtrManager::setUserCallback(void* pointer, UserCallback const& f)
//{
//  // TODO ??
//  auto pointer_record = getPointerRecord(pointer);
//  pointer_record->m_user_callback = f;
//}
//
//void SharedPtrManager::setGlobalUserCallback(UserCallback const& f)
//{
//  m_user_callback = f;
//}

msp_pointer_record* SharedPtrManager::getPointerRecord(void* pointer)
{
  std::lock_guard<std::mutex> lock(m_mutex);
  auto record = m_pointer_map.find(pointer);
  return record->second ? *record->second : &s_null_record;
}

// TODO: Need a better way of dealing with non-cuda builds here...
msp_pointer_record* SharedPtrManager::makeSharedPtrRecord(void const* c_pointer, void const* c_d_pointer,
                                                          size_t size,
                                                          bool owned)
{
  void* pointer = const_cast<void*>(c_pointer);
#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
  void* d_pointer = const_cast<void*>(c_d_pointer);
#else
  CHAI_UNUSED_VAR(c_d_pointer);
#endif

  if (pointer == nullptr) {
     return &s_null_record ;
  }

  m_resource_manager.registerAllocation(
      pointer,
      {pointer, size, m_allocators[chai::CPU]->getAllocationStrategy()});

#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
  m_resource_manager.registerAllocation(
      d_pointer,
      {d_pointer, size, m_allocators[chai::GPU]->getAllocationStrategy()});
#endif

  auto pointer_record = getPointerRecord(pointer);

  if (pointer_record == &s_null_record) {
     if (pointer) {
        pointer_record = new msp_pointer_record();
     } else {
        return pointer_record;
     }
  }
  else {
     CHAI_LOG(Warning, "SharedPtrManager::makeManaged found abandoned pointer record!!!");
     //callback(pointer_record, ACTION_FOUND_ABANDONED, space);
  }

  pointer_record->m_pointers[chai::CPU] = pointer;
  pointer_record->m_owned[chai::CPU] = owned;
#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
  pointer_record->m_pointers[chai::GPU] = d_pointer;
  pointer_record->m_owned[chai::GPU] = owned;
#endif
  //pointer_record->m_user_callback = [] (const msp_pointer_record*, Action, ExecutionSpace) {};
  
  for (int space = CPU; space < NUM_EXECUTION_SPACES; ++space) {
    pointer_record->m_allocators[space] = getAllocatorId(ExecutionSpace(space));
  }

  if (pointer) {
     registerPointer(pointer_record, chai::CPU, owned);
#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
     registerPointer(pointer_record, chai::GPU, owned);
#endif
  }

  return pointer_record;
}

msp_pointer_record* SharedPtrManager::deepCopyRecord(msp_pointer_record const* record, bool poly = false)
{
  msp_pointer_record* new_record = new msp_pointer_record{};
  //new_record->m_user_callback = [] (const msp_pointer_record*, Action, ExecutionSpace) {};

  const ExecutionSpace last_space = record->m_last_space;
  new_record->m_last_space = last_space;
  for (int space = CPU; space < NUM_EXECUTION_SPACES; ++space) {
    new_record->m_allocators[space] = record->m_allocators[space];
  }

  allocate(new_record, last_space);

  for (int space = CPU; space < NUM_EXECUTION_SPACES; ++space) {
    new_record->m_owned[space] = true;
    new_record->m_touched[space] = false;
  }

  new_record->m_touched[last_space] = true;

  void* dst_pointer = new_record->m_pointers[last_space];
  void* src_pointer = record->m_pointers[last_space];

  chai::expt::copy(dst_pointer, src_pointer, m_resource_manager, last_space, last_space, poly);

  return new_record;
}

std::unordered_map<void*, const msp_pointer_record*>
SharedPtrManager::getPointerMap() const
{
  std::lock_guard<std::mutex> lock(m_mutex);
  std::unordered_map<void*, const msp_pointer_record*> mapCopy;

  for (const auto& entry : m_pointer_map) {
    mapCopy[entry.first] = *entry.second;
  }

  return mapCopy;
}

size_t SharedPtrManager::getTotalNumSharedPtrs() const { return m_pointer_map.size(); }

//void SharedPtrManager::reportLeaks() const
//{
//  std::lock_guard<std::mutex> lock(m_mutex);
//  for (const auto& entry : m_pointer_map) {
//    const void* pointer = entry.first;
//    const msp_pointer_record* record = *entry.second;
//
//    for (int s = CPU; s < NUM_EXECUTION_SPACES; ++s) {
//      if (pointer == record->m_pointers[s]) {
//        callback(record, ACTION_LEAKED, ExecutionSpace(s));
//      }
//    }
//  }
//}

int
SharedPtrManager::getAllocatorId(ExecutionSpace space) const
{
  return m_allocators[space]->getId();
}

void SharedPtrManager::evict(ExecutionSpace space, ExecutionSpace destinationSpace) {
   // Check arguments
   if (space == NONE) {
      // Nothing to be done
      return;
   }

   if (destinationSpace == NONE) {
      // If the destination space is NONE, evicting invalidates all data and
      // leaves us in a bad state (if the last touch was in the eviction space).
      CHAI_LOG(Warning, "evict does nothing with destinationSpace == NONE!");
      return;
   }

   if (space == destinationSpace) {
      // It doesn't make sense to evict to the same space, so do nothing
      CHAI_LOG(Warning, "evict does nothing with space == destinationSpace!");
      return;
   }

   // Now move and evict
   std::vector<msp_pointer_record*> pointersToEvict;
   {
      std::lock_guard<std::mutex> lock(m_mutex);
      for (const auto& entry : m_pointer_map) {
         // Get the pointer record
         auto record = *entry.second;

         // Move the data and register the touches
         move(record, destinationSpace);
         registerTouch(record, destinationSpace);

         // If the destinationSpace is ever allowed to be NONE, then we will need to
         // update the touch in the eviction space and make sure the last space is not
         // the eviction space.

         // Mark record for eviction later in this routine
         pointersToEvict.push_back(record);
      }
   }

   // This must be done in a second pass because free erases from m_pointer_map,
   // which would invalidate the iterator in the above loop
   for (const auto& entry : pointersToEvict) {
      free(entry, space);
   }
}


}  // end of namespace expt
}  // end of namespace chai
