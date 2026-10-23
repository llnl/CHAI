//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CHAI_SHARED_ARRAY_MANAGER_HPP
#define CHAI_SHARED_ARRAY_MANAGER_HPP

#include "umpire/Allocator.hpp"
#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/config.hpp"
#include "umpire/util/MemoryResourceTraits.hpp"

#if !defined(UMPIRE_ENABLE_MPI)
#error "SharedArrayManager requires an MPI-enabled Umpire build."
#endif

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "mpi.h"

namespace chai::expt
{
namespace detail
{
inline std::atomic<unsigned long long>& sharedArrayAllocationCounter()
{
  static std::atomic<unsigned long long> counter{0};
  return counter;
}
}  // namespace detail

/*!
 * \brief Manages a host array in an Umpire shared-memory allocation.
 *
 * \tparam ElementType The type of elements contained in this array.
 *
 * \details Construction, copying, resizing, data access, and destruction of
 *          an allocated manager are collective over the communicator for the
 *          allocator's sharing scope. Every participating rank must perform
 *          those operations in the same order. Calls to data() synchronize
 *          the sharing communicator before returning the local mapping.
 *
 * \note Newly allocated elements are left uninitialized. Resizing preserves
 *       the prefix of the old allocation, but does not initialize any new
 *       elements.
 *
 * \note This class borrows the communicator returned by Umpire. Applications
 *       using POSIX shared memory must destroy all managers before calling
 *       umpire::cleanup_cached_communicators() and MPI_Finalize().
 */
template <typename ElementType>
class SharedArrayManager
{
  static_assert(std::is_trivially_copyable_v<ElementType>,
                "SharedArrayManager requires trivially copyable element "
                "types.");
  static_assert(std::is_default_constructible_v<ElementType>,
                "SharedArrayManager requires default-constructible element "
                "types.");
  static_assert(std::is_trivially_destructible_v<ElementType>,
                "SharedArrayManager requires trivially destructible element "
                "types.");

public:
  /*!
   * \brief Constructs an empty manager using Umpire's default SHARED
   *        allocator and MPI_COMM_WORLD.
   */
  SharedArrayManager()
      : SharedArrayManager(0, defaultAllocator(), MPI_COMM_WORLD)
  {
  }

  /*!
   * \brief Constructs a manager with \p size uninitialized elements using
   *        Umpire's default SHARED allocator and MPI_COMM_WORLD.
   */
  explicit SharedArrayManager(std::size_t size)
      : SharedArrayManager(size, defaultAllocator(), MPI_COMM_WORLD)
  {
  }

  /*!
   * \brief Constructs an empty manager using \p allocator.
   *
   * \param allocator Shared-memory allocator used for storage.
   * \param parent_comm Parent communicator used by Umpire when deriving a
   *                    POSIX shared-memory communicator. MPI3 resources use
   *                    their allocator-owned communicator.
   */
  explicit SharedArrayManager(const umpire::Allocator& allocator,
                              MPI_Comm parent_comm = MPI_COMM_WORLD)
      : SharedArrayManager(0, allocator, parent_comm)
  {
  }

  /*!
   * \brief Constructs a manager with \p size uninitialized elements using
   *        \p allocator.
   *
   * \param size Number of elements to allocate.
   * \param allocator Shared-memory allocator used for storage.
   * \param parent_comm Parent communicator used by Umpire when deriving a
   *                    POSIX shared-memory communicator. MPI3 resources use
   *                    their allocator-owned communicator.
   */
  SharedArrayManager(std::size_t size,
                     const umpire::Allocator& allocator,
                     MPI_Comm parent_comm = MPI_COMM_WORLD)
      : m_allocator{allocator}
  {
    initializeCommunicator(parent_comm);
    validateCollectiveSize(size);

    if (size != 0) {
      m_data = allocate(size);
      m_size = size;
      synchronize();
    }
  }

  /*!
   * \brief Deep-copy constructs a distinct shared allocation.
   */
  SharedArrayManager(const SharedArrayManager& other)
      : m_size{other.m_size},
        m_allocator{other.m_allocator},
        m_comm{other.m_comm},
        m_comm_rank{other.m_comm_rank}
  {
    if (m_size != 0) {
      other.synchronize();
      m_data = allocate(m_size);

      if (m_comm_rank == 0) {
        std::memcpy(m_data, other.m_data, sizeInBytes(m_size));
      }

      synchronize();
    }
  }

  /*!
   * \brief Deep-copy assigns into a distinct shared allocation.
   */
  SharedArrayManager& operator=(const SharedArrayManager& other)
  {
    if (this != &other) {
      SharedArrayManager copy{other};
      swap(copy);
    }

    return *this;
  }

  /*!
   * \brief Move-constructs by transferring allocation ownership.
   */
  SharedArrayManager(SharedArrayManager&& other) noexcept
      : m_data{other.m_data},
        m_size{other.m_size},
        m_allocator{other.m_allocator},
        m_comm{other.m_comm},
        m_comm_rank{other.m_comm_rank}
  {
    other.m_data = nullptr;
    other.m_size = 0;
    other.m_comm = MPI_COMM_NULL;
    other.m_comm_rank = 0;
  }

  /*!
   * \brief Move-assigns by releasing current storage and transferring
   *        allocation ownership.
   */
  SharedArrayManager& operator=(SharedArrayManager&& other)
  {
    if (this != &other) {
      clear();

      m_data = other.m_data;
      m_size = other.m_size;
      m_allocator = other.m_allocator;
      m_comm = other.m_comm;
      m_comm_rank = other.m_comm_rank;

      other.m_data = nullptr;
      other.m_size = 0;
      other.m_comm = MPI_COMM_NULL;
      other.m_comm_rank = 0;
    }

    return *this;
  }

  /*!
   * \brief Collectively releases the managed shared allocation.
   */
  ~SharedArrayManager() { clear(); }

  /*!
   * \brief Collectively resizes the managed allocation.
   *
   * Existing elements are preserved through min(size(), new_size). Newly
   * added elements are left uninitialized.
   */
  void resize(std::size_t new_size)
  {
    validateCollectiveSize(new_size);

    if (new_size == m_size) {
      synchronize();
      return;
    }

    synchronize();

    ElementType* new_data = nullptr;
    if (new_size != 0) {
      new_data = allocate(new_size);
    }

    if (m_comm_rank == 0 && m_data != nullptr && new_data != nullptr) {
      std::memcpy(new_data, m_data, sizeInBytes(std::min(m_size, new_size)));
    }

    synchronize();

    if (m_data != nullptr) {
      m_allocator.deallocate(m_data);
    }

    m_data = new_data;
    m_size = new_size;
    synchronize();
  }

  /*!
   * \brief Returns the number of managed elements without synchronizing.
   */
  std::size_t size() const noexcept { return m_size; }

  /*!
   * \brief Collectively synchronizes and returns the local shared mapping.
   */
  ElementType* data()
  {
    synchronize();
    return m_data;
  }

  /*!
   * \brief Collectively synchronizes and returns the local shared mapping.
   */
  const ElementType* data() const
  {
    synchronize();
    return m_data;
  }

  /*!
   * \brief Performs a barrier on the allocator's sharing communicator.
   */
  void synchronize() const
  {
    if (m_comm == MPI_COMM_NULL) {
      return;
    }

    ensureMpiActive();
    checkMpi(MPI_Barrier(m_comm), "MPI_Barrier");
  }

private:
  ElementType* m_data{nullptr};
  std::size_t m_size{0};
  umpire::Allocator m_allocator{};
  MPI_Comm m_comm{MPI_COMM_NULL};
  int m_comm_rank{0};

  static umpire::Allocator defaultAllocator()
  {
    ensureMpiActive();
    return ::umpire::ResourceManager::getInstance().getAllocator("SHARED");
  }

  static void ensureMpiActive()
  {
    int initialized = 0;
    checkMpi(MPI_Initialized(&initialized), "MPI_Initialized");
    if (!initialized) {
      throw std::runtime_error(
          "SharedArrayManager requires MPI_Init before construction or use.");
    }

    int finalized = 0;
    checkMpi(MPI_Finalized(&finalized), "MPI_Finalized");
    if (finalized) {
      throw std::runtime_error(
          "SharedArrayManager cannot be used after MPI_Finalize.");
    }
  }

  static void checkMpi(int status, const char* operation)
  {
    if (status != MPI_SUCCESS) {
      throw std::runtime_error(std::string{"SharedArrayManager: "} + operation +
                               " failed.");
    }
  }

  void initializeCommunicator(MPI_Comm parent_comm)
  {
    ensureMpiActive();

    if (m_allocator.getAllocationStrategy() == nullptr ||
        m_allocator.getAllocationStrategy()->getTraits().resource !=
            umpire::MemoryResourceTraits::resource_type::shared) {
      throw std::invalid_argument(
          "SharedArrayManager requires an Umpire shared-memory allocator.");
    }

    if (parent_comm == MPI_COMM_NULL) {
      throw std::invalid_argument(
          "SharedArrayManager requires a non-null parent communicator.");
    }

    m_comm = ::umpire::get_communicator_for_allocator(m_allocator, parent_comm);
    if (m_comm == MPI_COMM_NULL) {
      throw std::invalid_argument(
          "Umpire could not derive a communicator for the shared-memory "
          "allocator scope.");
    }

    checkMpi(MPI_Comm_rank(m_comm, &m_comm_rank), "MPI_Comm_rank");
  }

  void validateCollectiveSize(std::size_t size) const
  {
    static_assert(sizeof(std::size_t) <= sizeof(unsigned long long));

    const unsigned long long local_size = static_cast<unsigned long long>(size);
    unsigned long long minimum_size = 0;
    unsigned long long maximum_size = 0;

    checkMpi(MPI_Allreduce(&local_size,
                           &minimum_size,
                           1,
                           MPI_UNSIGNED_LONG_LONG,
                           MPI_MIN,
                           m_comm),
             "MPI_Allreduce");
    checkMpi(MPI_Allreduce(&local_size,
                           &maximum_size,
                           1,
                           MPI_UNSIGNED_LONG_LONG,
                           MPI_MAX,
                           m_comm),
             "MPI_Allreduce");

    if (minimum_size != maximum_size) {
      throw std::invalid_argument(
          "SharedArrayManager requires the same size on every participating "
          "rank.");
    }

    if (size > std::numeric_limits<std::size_t>::max() / sizeof(ElementType)) {
      throw std::length_error(
          "SharedArrayManager allocation size overflows std::size_t.");
    }
  }

  static std::size_t sizeInBytes(std::size_t size) noexcept
  {
    return size * sizeof(ElementType);
  }

  std::string nextAllocationName() const
  {
    unsigned long long sequence = 0;
    int leader_world_rank = 0;

    if (m_comm_rank == 0) {
      sequence = detail::sharedArrayAllocationCounter().fetch_add(
          1, std::memory_order_relaxed);
      checkMpi(MPI_Comm_rank(MPI_COMM_WORLD, &leader_world_rank),
               "MPI_Comm_rank");
    }

    checkMpi(MPI_Bcast(&sequence, 1, MPI_UNSIGNED_LONG_LONG, 0, m_comm),
             "MPI_Bcast");
    checkMpi(MPI_Bcast(&leader_world_rank, 1, MPI_INT, 0, m_comm), "MPI_Bcast");

    return "chai_shared_array_" + std::to_string(leader_world_rank) + "_" +
           std::to_string(sequence);
  }

  ElementType* allocate(std::size_t size)
  {
    return static_cast<ElementType*>(
        m_allocator.allocate(nextAllocationName(), sizeInBytes(size)));
  }

  void clear()
  {
    if (m_data == nullptr) {
      m_size = 0;
      return;
    }

    synchronize();
    m_allocator.deallocate(m_data);
    m_data = nullptr;
    m_size = 0;
    synchronize();
  }

  void swap(SharedArrayManager& other) noexcept
  {
    using std::swap;
    swap(m_data, other.m_data);
    swap(m_size, other.m_size);
    swap(m_allocator, other.m_allocator);
    swap(m_comm, other.m_comm);
    swap(m_comm_rank, other.m_comm_rank);
  }
};  // class SharedArrayManager
}  // namespace chai::expt

#endif  // CHAI_SHARED_ARRAY_MANAGER_HPP
