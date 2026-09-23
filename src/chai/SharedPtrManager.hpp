//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_SharedPtrManager_HPP
#define CHAI_SharedPtrManager_HPP

#include "chai/ChaiMacros.hpp"
#include "chai/ExecutionContextManager.hpp"
#include "chai/ExecutionSpaces.hpp"
#include "chai/Types.hpp"

#include "chai/PointerRecord.hpp"
#include "chai/SharedPointerRecord.hpp"

#if defined(CHAI_ENABLE_RAJA_PLUGIN)
#include "chai/pluginLinker.hpp"
#endif

#include <unordered_map>

#include "umpire/Allocator.hpp"
#include "umpire/util/MemoryMap.hpp"


#include "chai/DeviceHelpers.hpp"


namespace chai
{
namespace expt
{

/*!
 * \brief Singleton that manages caching and movement of ManagedSharedPtr objects.
 *
 * The SharedPtrManager class co-ordinates the allocation and movement of
 * ManagedSharedPtr objects. These objects are cached, and data is only copied
 * between ExecutionSpaces when necessary. This functionality is typically
 * hidden behind a programming model layer, such as RAJA, or the exmaple
 * included in util/forall.hpp
 *
 * The SharedPtrManager is a singleton, so must always be accessed through the
 * static getInstance method. Here is an example using the SharedPtrManager:
 *
 * \code
 * const chai::SharedPtrManager* rm = chai::SharedPtrManager::getInstance();
 * rm->setExecutionSpace(chai::CPU);
 * // Do something with ManagedSharedPtr on the CPU... but they must be copied!
 * rm->setExecutionSpace(chai::NONE);
 * \endcode
 *
 * SharedPtrManager differs from ArrayManager such that it does not support
 * reallocation or callbacks (at this time).
 */
class SharedPtrManager
{
public:
  template <typename T>
  using T_non_const = typename std::remove_const<T>::type;

  using PointerMap = umpire::util::MemoryMap<msp_pointer_record*>;

  CHAISHAREDDLL_API static msp_pointer_record s_null_record;

  /*!
   * \brief Get the singleton instance.
   *
   * \return Pointer to the SharedPtrManager instance.
   *
   */
  CHAISHAREDDLL_API
  static SharedPtrManager* getInstance();

  /*!
   * \brief Set the current execution space.
   *
   * \param space The space to set as current.
   */
  CHAISHAREDDLL_API void setExecutionSpace(ExecutionSpace space);

  /*!
   * \brief Get the current execution space.
   *
   * \return The current execution space.jo
   */
  CHAISHAREDDLL_API ExecutionSpace getExecutionSpace();

  /*!
   * \brief Move data in pointer to the current execution space.
   *
   * \param pointer Pointer to data in any execution space.
   * \return Pointer to data in the current execution space.
   */
  CHAISHAREDDLL_API void* move(void* pointer,
                               msp_pointer_record* pointer_record,
                               ExecutionSpace = NONE, bool = false);

  /*!
   * \brief Register a touch of the pointer in the current execution space.
   *
   * \param pointer Raw pointer to register a touch of.
   */
  CHAISHAREDDLL_API void registerTouch(msp_pointer_record* pointer_record);

  /*!
   * \brief Register a touch of the pointer in the given execution space.
   *
   * The pointer doesn't need to exist in the space being touched.
   *
   * \param pointer Raw pointer to register a touch of.
   * \param space Space to register touch.
   */
  CHAISHAREDDLL_API void registerTouch(msp_pointer_record* pointer_record, ExecutionSpace space);

  /*!
   * \brief Make a new allocation of the data described by the msp_pointer_record in
   * the given space.
   *
   * \param pointer_record
   * \param space Space in which to make the allocation.
   */
  CHAISHAREDDLL_API void allocate(msp_pointer_record* pointer_record, ExecutionSpace space = CPU);

  /*!
   * \brief Set the default space for new ManagedArray allocations.
   *
   * ManagedArrays allocated without an explicit ExecutionSpace argument will
   * be allocated in space after this routine is called.
   *
   * \param space New space for default allocations.
   */
  CHAISHAREDDLL_API void setDefaultAllocationSpace(ExecutionSpace space);

  /*!
   * \brief Get the currently set default allocation space.
   *
   * See also setDefaultAllocationSpace.
   *
   * \return Current default space for allocations.
   */
  CHAISHAREDDLL_API ExecutionSpace getDefaultAllocationSpace();

  /*!
   * \brief Free allocation(s) associated with the given msp_pointer_record.
   *        Default (space == NONE) will free all allocations and delete
   *        the pointer record.
   */
  CHAISHAREDDLL_API void free(msp_pointer_record* pointer, ExecutionSpace space = NONE);

#if defined(CHAI_ENABLE_PICK)
  template <typename T>
   T_non_const<T> pick(T* src_ptr, size_t index);

  template <typename T>
   void set(T* dst_ptr, size_t index, const T& val);
#endif

  template<typename Ptr>
  msp_pointer_record* makeSharedPtrRecord(std::initializer_list<Ptr*> pointers,
                                                          std::initializer_list<chai::ExecutionSpace> spaces,
                                                          size_t size,
                                                          bool owned);

  CHAISHAREDDLL_API msp_pointer_record* makeSharedPtrRecord(void const* c_pointer, void const* c_d_pointer,
                                                            size_t size,
                                                            //ExecutionSpace space,
                                                            bool owned);

  /*!
   * \brief Assign a user-defined callback triggered upon memory operations.
   *        This callback applies to a single ManagedArray.
   */
  //CHAISHAREDDLL_API void setUserCallback(void* pointer, UserCallback const& f);

  /*!
   * \brief Assign a user-defined callback triggered upon memory operations.
   *        This callback applies to all ManagedArrays.
   */
  //CHAISHAREDDLL_API void setGlobalUserCallback(UserCallback const& f);

  /*!
   * \brief Set touched to false in all spaces for the given msp_pointer_record.
   *
   * \param pointer_record msp_pointer_record to reset.
   */
  CHAISHAREDDLL_API void resetTouch(msp_pointer_record* pointer_record);

  /*!
   * \brief Find the msp_pointer_record corresponding to the raw pointer.
   *
   * \param pointer Raw pointer to find the msp_pointer_record for.
   *
   * \return msp_pointer_record containing the raw pointer, or an empty
   *         msp_pointer_record if none found.
   */
  CHAISHAREDDLL_API msp_pointer_record* getPointerRecord(void* pointer);

  /*!
   * \brief Create a copy of the given msp_pointer_record with a new allocation
   *  in the active space.
   *
   * \param record The msp_pointer_record to copy.
   * \param poly true if the underlying type is polymorphic.
   *
   * \return A copy of the given msp_pointer_record, must be free'd with delete.
   */
  CHAISHAREDDLL_API msp_pointer_record* deepCopyRecord(msp_pointer_record const* record, bool poly);

  /*!
   * \brief Create a copy of the pointer map.
   *
   * \return A copy of the pointer map. Can be used to find memory leaks.
   */
  CHAISHAREDDLL_API std::unordered_map<void*, const msp_pointer_record*> getPointerMap() const;

  /*!
   * \brief Get the total number of arrays registered with the array manager.
   *
   * \return The total number of arrays registered with the array manager.
   */
  CHAISHAREDDLL_API size_t getTotalNumSharedPtrs() const;

  //TODO: define reportLeaks for ManagedSharedPtr.
  /*!
   * \brief Calls callbacks of pointers still in the map with ACTION_LEAKED.
   */
  //CHAISHAREDDLL_API void reportLeaks() const;

  /*!
   * \brief Get the allocator ID
   *
   * \return The allocator ID.
   */
  CHAISHAREDDLL_API int getAllocatorId(ExecutionSpace space) const;

  /*!
   * \brief Wraps our resource manager's copy.
   */
  CHAISHAREDDLL_API void copy(void * dst, void * src, size_t size); 
  
  /*!
   * \brief Registering an allocation with the SharedPtrManager
   *
   * \param record msp_pointer_record of this allocation.
   * \param space Space in which the pointer was allocated.
   * \param owned Should the allocation be free'd by CHAI?
   */
  CHAISHAREDDLL_API void registerPointer(msp_pointer_record* record,
                                         ExecutionSpace space,
                                         bool owned = true);

  /*!
   * \brief Deregister a msp_pointer_record from the SharedPtrManager.
   *
   * \param record msp_pointer_record of allocation to deregister.
   * \param deregisterFromUmpire If true, deregister from umpire as well.
   */
  CHAISHAREDDLL_API void deregisterPointer(msp_pointer_record* record, bool deregisterFromUmpire=false);

  /*!
   * \brief set the allocator for an execution space.
   *
   * \param space Execution space to set the default allocator for.
   * \param allocator The allocator to use for this space. Will be copied into chai.
   */
  void setAllocator(ExecutionSpace space, umpire::Allocator &allocator);

  /*!
   * \brief Get the allocator for an execution space.
   *
   * \param space Execution space of the allocator to get.
   *
   * \return The allocator for the given space.
   */
  umpire::Allocator getAllocator(ExecutionSpace space);
  
 /*!
   * \brief Turn callbacks on.
   */
  //void enableCallbacks() { m_callbacks_active = true; }

  /*!
   * \brief Turn callbacks off.
   */
  //void disableCallbacks() { m_callbacks_active = false; }

  /*!
   * \brief synchronize the device if there hasn't been a synchronize since the last kernel
   */
  CHAISHAREDDLL_API bool syncIfNeeded();

#if defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
  /*!
   * \brief Turn the GPU simulation mode on or off.
   */
  void setGPUSimMode(bool gpuSimMode)
  {
    ExecutionContextManager::getInstance().setGPUSimMode(gpuSimMode);
  }

  /*!
   * \brief Return true if GPU simulation mode is on, false otherwise.
   */
  bool isGPUSimMode()
  {
    return ExecutionContextManager::getInstance().isGPUSimMode();
  }
#endif

  /*!
   * \brief Evicts the data in the given space.
   *
   * \param space Execution space to evict.
   * \param destinationSpace The execution space to move the data to.
   *                            Must not equal space or NONE.
   */
  CHAISHAREDDLL_API void evict(ExecutionSpace space, ExecutionSpace destinationSpace);


protected:
  /*!
   * \brief Construct a new SharedPtrManager.
   *
   * The constructor is a protected member, ensuring that it can
   * only be called by the singleton getInstance method.
   */
  SharedPtrManager();



private:


  /*!
   * \brief Move data in msp_pointer_record to the corresponding ExecutionSpace.
   *
   * \param record
   * \param space
   */
  void move(msp_pointer_record* record, ExecutionSpace space, bool = false);
  
    /*!
   * \brief Execute a user callback if callbacks are active
   *
   * \param record The pointer record containing the callback
   * \param action The event that occurred
   * \param space The space in which the event occurred
   * \param size The number of bytes in the array associated with this pointer record
   */
//  inline void callback(const msp_pointer_record* record,
//                       Action action,
//                       ExecutionSpace space) const {
//     if (m_callbacks_active) {
//        // Callback for this ManagedArray only
//        if (record && record->m_user_callback) {
//           record->m_user_callback(record, action, space);
//        }
//
//        // Callback for all ManagedArrays
//        if (m_user_callback) {
//           m_user_callback(record, action, space);
//        }
//     }
//  }

  /**
   * Default space for new allocations.
   */
  ExecutionSpace m_default_allocation_space;

  /*!
   * Map of active ManagedArray pointers to their corresponding msp_pointer_record.
   */
  PointerMap m_pointer_map;

  /*!
   *
   * \brief Array of umpire::Allocators, indexed by ExecutionSpace.
   */
  umpire::Allocator* m_allocators[NUM_EXECUTION_SPACES];

  /*!
   * \brief The umpire resource manager.
   */
  umpire::ResourceManager& m_resource_manager;

  /*!
   * \brief Used for thread-safe operations.
   */
  mutable std::mutex m_mutex;

  /*!
   * \brief A callback triggered upon memory operations on all ManagedArrays.
   */
  //UserCallback m_user_callback;

  /*!
   * \brief Controls whether or not callbacks are called.
   */
  //bool m_callbacks_active;

};

}  // end of namespace expt
}  // end of namespace chai

#include "chai/SharedPtrManager.inl"

#endif  // CHAI_SharedPtrManager_HPP
