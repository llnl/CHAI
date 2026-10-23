//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include <cstddef>
#include <string>
#include <vector>

#include "chai/expt/ManagedArrayPointer.hpp"
#include "chai/expt/ManagedArraySharedPointer.hpp"
#include "chai/expt/ManagedArrayView.hpp"
#include "chai/expt/SharedArrayManager.hpp"
#include "gtest/gtest.h"
#include "mpi.h"
#include "umpire/ResourceManager.hpp"
#include "umpire/Umpire.hpp"
#include "umpire/config.hpp"

namespace
{
template <typename T>
using Manager = chai::expt::SharedArrayManager<T>;

template <typename T>
using Pointer = chai::expt::ManagedArrayPointer<T, Manager<T>>;

template <typename T>
using SharedPointer = chai::expt::ManagedArraySharedPointer<T, Manager<T>>;

template <typename T>
using View = chai::expt::ManagedArrayView<T, Manager<T>>;

std::vector<umpire::Allocator> sharedAllocators()
{
  auto& resource_manager = umpire::ResourceManager::getInstance();
  std::vector<umpire::Allocator> allocators;

#if defined(UMPIRE_ENABLE_IPC_SHARED_MEMORY)
  allocators.push_back(resource_manager.getAllocator("SHARED::POSIX"));
#endif
#if defined(UMPIRE_ENABLE_MPI3_SHARED_MEMORY)
  allocators.push_back(resource_manager.getAllocator("SHARED::MPI3"));
#endif

  return allocators;
}

bool hasTwoSharingRanks(const umpire::Allocator& allocator)
{
  MPI_Comm comm =
      umpire::get_communicator_for_allocator(allocator, MPI_COMM_WORLD);
  int size = 0;
  MPI_Comm_size(comm, &size);
  return size == 2;
}

int sharingRank(const umpire::Allocator& allocator)
{
  MPI_Comm comm =
      umpire::get_communicator_for_allocator(allocator, MPI_COMM_WORLD);
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  return rank;
}
}  // namespace

TEST(SharedArrayManagerMPI, VisibilityResizeAndDeepCopy)
{
  for (const umpire::Allocator& allocator : sharedAllocators()) {
    if (!hasTwoSharingRanks(allocator)) {
      GTEST_SKIP() << "This test requires two ranks in one shared-memory "
                      "scope.";
    }

    const int rank = sharingRank(allocator);
    Manager<int> values{5, allocator};
    int* data = values.data();

    if (rank == 0) {
      for (std::size_t i = 0; i < values.size(); ++i) {
        data[i] = static_cast<int>(20 + i);
      }
    }

    data = values.data();
    for (std::size_t i = 0; i < values.size(); ++i) {
      EXPECT_EQ(data[i], static_cast<int>(20 + i));
    }

    values.resize(9);
    data = values.data();
    for (std::size_t i = 0; i < 5; ++i) {
      EXPECT_EQ(data[i], static_cast<int>(20 + i));
    }

    Manager<int> copy{values};
    int* copy_data = copy.data();
    data = values.data();
    EXPECT_NE(copy_data, data);

    if (rank == 0) {
      copy_data[0] = -20;
    }

    copy_data = copy.data();
    data = values.data();
    EXPECT_EQ(copy_data[0], -20);
    EXPECT_EQ(data[0], 20);
  }
}

TEST(SharedArrayManagerMPI, RejectsMismatchedResize)
{
  for (const umpire::Allocator& allocator : sharedAllocators()) {
    if (!hasTwoSharingRanks(allocator)) {
      GTEST_SKIP() << "This test requires two ranks in one shared-memory "
                      "scope.";
    }

    const int rank = sharingRank(allocator);
    Manager<int> values{4, allocator};

    EXPECT_THROW(values.resize(static_cast<std::size_t>(5 + rank)),
                 std::invalid_argument);
    EXPECT_EQ(values.size(), 4);
    values.synchronize();
  }
}

TEST(SharedArrayManagerMPI, WrapperSynchronization)
{
  for (const umpire::Allocator& allocator : sharedAllocators()) {
    if (!hasTwoSharingRanks(allocator)) {
      GTEST_SKIP() << "This test requires two ranks in one shared-memory "
                      "scope.";
    }

    const int rank = sharingRank(allocator);

    Pointer<int> pointer = Pointer<int>::make(4, allocator);
    if (rank == 0) {
      pointer[0] = 31;
    }
    pointer.update();
    EXPECT_EQ(pointer[0], 31);

    Pointer<int> pointer_copy{pointer};
    EXPECT_EQ(pointer_copy.data(), pointer.data());
    pointer.free();

    {
      SharedPointer<int> shared = SharedPointer<int>::make(4, allocator);
      if (rank == 0) {
        shared[0] = 32;
      }
      shared.update();
      EXPECT_EQ(shared[0], 32);

      SharedPointer<int> shared_copy{shared};
      EXPECT_EQ(shared_copy.data(), shared.data());
    }

    {
      Manager<int> manager{4, allocator};
      View<int> view{manager};
      if (rank == 0) {
        view[0] = 33;
      }
      view.update();
      EXPECT_EQ(view[0], 33);

      View<int> view_copy{view};
      EXPECT_EQ(view_copy.data(), view.data());
    }
  }
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);

  int world_size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  if (world_size != 2) {
    MPI_Finalize();
    return 1;
  }

  const int result = RUN_ALL_TESTS();

  umpire::cleanup_cached_communicators();
  MPI_Finalize();
  return result;
}
