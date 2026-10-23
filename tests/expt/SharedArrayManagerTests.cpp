//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include <cstddef>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
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

struct CountedDefaultConstruction {
  CountedDefaultConstruction() { ++constructions; }

  int value;
  static int constructions;
};

int CountedDefaultConstruction::constructions = 0;

static_assert(std::is_trivially_copyable_v<CountedDefaultConstruction>);
static_assert(std::is_trivially_destructible_v<CountedDefaultConstruction>);

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
}  // namespace

TEST(SharedArrayManager, DefaultAndSizeConstruction)
{
  Manager<int> empty;
  EXPECT_EQ(empty.size(), 0);
  EXPECT_EQ(empty.data(), nullptr);

  Manager<int> values{8};
  EXPECT_EQ(values.size(), 8);
  int* data = values.data();
  ASSERT_NE(data, nullptr);

  for (std::size_t i = 0; i < values.size(); ++i) {
    data[i] = static_cast<int>(i);
  }
}

TEST(SharedArrayManager, InjectedSharedAllocators)
{
  for (const umpire::Allocator& allocator : sharedAllocators()) {
    Manager<int> values{4, allocator};
    EXPECT_EQ(values.size(), 4);
    EXPECT_NE(values.data(), nullptr);
  }
}

TEST(SharedArrayManager, DoesNotDefaultConstructElements)
{
  CountedDefaultConstruction::constructions = 0;

  Manager<CountedDefaultConstruction> values{6};
  EXPECT_EQ(values.size(), 6);
  EXPECT_EQ(CountedDefaultConstruction::constructions, 0);
}

TEST(SharedArrayManager, ResizePreservesPrefix)
{
  Manager<int> values{5};
  int* data = values.data();
  for (std::size_t i = 0; i < values.size(); ++i) {
    data[i] = static_cast<int>(10 + i);
  }

  values.resize(9);
  EXPECT_EQ(values.size(), 9);
  data = values.data();
  for (std::size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(data[i], static_cast<int>(10 + i));
  }

  values.resize(3);
  EXPECT_EQ(values.size(), 3);
  data = values.data();
  for (std::size_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(data[i], static_cast<int>(10 + i));
  }

  values.resize(0);
  EXPECT_EQ(values.size(), 0);
  EXPECT_EQ(values.data(), nullptr);
}

TEST(SharedArrayManager, DeepCopyAndMove)
{
  Manager<int> source{4};
  int* source_data = source.data();
  for (std::size_t i = 0; i < source.size(); ++i) {
    source_data[i] = static_cast<int>(i + 1);
  }

  Manager<int> copy{source};
  int* copy_data = copy.data();
  source_data = source.data();
  EXPECT_NE(copy_data, source_data);
  for (std::size_t i = 0; i < source.size(); ++i) {
    EXPECT_EQ(copy_data[i], source_data[i]);
  }

  copy_data[0] = -1;
  EXPECT_EQ(source_data[0], 1);

  Manager<int> assigned;
  assigned = source;
  EXPECT_NE(assigned.data(), source.data());
  EXPECT_EQ(assigned.data()[2], 3);

  Manager<int> moved{std::move(copy)};
  EXPECT_EQ(copy.size(), 0);
  EXPECT_EQ(moved.size(), 4);
  EXPECT_EQ(moved.data()[0], -1);

  Manager<int> move_assigned;
  move_assigned = std::move(moved);
  EXPECT_EQ(moved.size(), 0);
  EXPECT_EQ(move_assigned.size(), 4);
  EXPECT_EQ(move_assigned.data()[0], -1);
}

TEST(SharedArrayManager, RejectsInvalidAllocatorAndOverflow)
{
  auto host = umpire::ResourceManager::getInstance().getAllocator("HOST");
  EXPECT_THROW((Manager<int>{host}), std::invalid_argument);

  constexpr std::size_t too_many =
      std::numeric_limits<std::size_t>::max() / sizeof(int) + 1;
  EXPECT_THROW((Manager<int>{too_many}), std::length_error);
}

TEST(SharedArrayManager, ExperimentalWrapperIntegration)
{
  Pointer<int> pointer = Pointer<int>::make(4);
  int* pointer_data = pointer.data();
  pointer_data[0] = 11;

  Pointer<int> pointer_copy{pointer};
  EXPECT_EQ(pointer_copy.data(), pointer.data());
  EXPECT_EQ(pointer_copy[0], 11);
  pointer.free();

  SharedPointer<int> shared = SharedPointer<int>::make(4);
  shared.data()[0] = 12;
  SharedPointer<int> shared_copy{shared};
  EXPECT_EQ(shared_copy.data(), shared.data());
  EXPECT_EQ(shared_copy[0], 12);

  Manager<int> manager{4};
  View<int> view{manager};
  view.data()[0] = 13;
  View<int> view_copy{view};
  EXPECT_EQ(view_copy.data(), view.data());
  EXPECT_EQ(view_copy[0], 13);
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);

  const int result = RUN_ALL_TESTS();

  umpire::cleanup_cached_communicators();
  MPI_Finalize();
  return result;
}
