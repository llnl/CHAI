[comment]: # (#################################################################)
[comment]: # (Copyright Lawrence Livermore National Security, LLC and other CHAI)
[comment]: # (contributors. See the CHAI LICENSE and COPYRIGHT files for details.)
[comment]: # 
[comment]: # (# SPDX-License-Identifier: BSD-3-Clause)
[comment]: # (#################################################################)

# CHAI Software Release Notes

Notes describing significant changes in each CHAI release are documented
in this file.

The format of this file is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [Unreleased] - Release date YYYY-MM-DD

### Added
- Added `chai::unpack_pointer_table` for creating persistent host/device pointer tables from a `ManagedArray<managed_ptr<T>>`.
- Promoted execution-context support into the core `chai` namespace as `ExecutionContext`, `ExecutionContextManager`, and `ExecutionContextGuard`.
- Added the automatically registered `ExecutionContextRAJAPlugin` when `CHAI_ENABLE_RAJA_PLUGIN` is enabled.

### Changed
- The `isSlice` function can now be called on `const chai::ManagedArray<T>`.
- `ArrayManager` now uses `ExecutionContextManager` as the source of execution-space and device-synchronization state. Its existing execution-space methods remain available as compatibility wrappers.
- The RAJA integration now uses `ExecutionContextRAJAPlugin` for both the core and experimental CHAI data structures.

### Removed
- Removed the experimental `chai::expt::Context*` APIs, the legacy `RajaExecutionSpacePlugin`, and the `CHAI_ENABLE_EXPERIMENTAL_RAJA_PLUGIN` option.

## [Version 2026.07.0] - Release date 2025-08-11

### Changed
- CHAI now requires C++20 as the minimum C++ standard.
- CHAI now requires ROCm 6 as the minimum ROCm version when ROCm is enabled.
- CHAI now requires CUDA 12 as the minimum CUDA version when CUDA is enabled.
- Updated BLT to v0.7.2.
- Updated RAJA to v2026.07.0.
- Updated Umpire to v2026.07.1.
- Made a ManagedArray constructor explicit to prevent unwanted implicit conversions.
- Added size\_type, value\_type, reference, const\_reference, pointer, and const\_pointer typenames to ManagedArray.
- In CUDA builds, prevent resource manager from being created during construction of an empty ManagedArray.
- Added experimental re-implementation of CHAI. See docs/sphinx/expt/design.rst for details. It is still subject to change and has some limitations, but is anticipated to be more performant and flexible.

## [Version 2025.12.0] - Release date 2025-12-22

### Fixed
- Fixed compiler error related to using a lambda as a default function argument.
- Fixed thin version of a ManagedArray constructor.

### Changed
- Updated Umpire to v2025.12.0.
- Updated RAJA to v2025.12.0.

### Added
- Added `CHAI_VERSION_*` macros to chai/config.hpp.
- Added overload of `chai::unpack` for `ManagedArray<managed_ptr<T>>` for use in calls to `chai::make_managed`.

## [Version 2025.09.1] - Release date 2025-09-15

### Fixed
- Fixed build error when `CHAI_ENABLE_EXPERIMENTAL` was set to ON.

### Changed
- Turned off `CHAI_ENABLE_EXPERIMENTAL` by default.

## [Version 2025.09.0] - Release date 2025-09-12

### Fixed
- Improved support for non-trivial element types in `ManagedArray`.

### Changed
- CHAI now requires C++17 as the minimum C++ standard.
- CHAI now requires CUDA 11 as the minimum CUDA version.
- Updated BLT to v0.7.1.
- Updated Umpire to v2025.09.0.
- Updated RAJA to v2025.09.0.

### Experimental
- Added a `ManagedSharedPtr` class similar to `managed_ptr` that acts like `std::shared_ptr` on the host and a view on the device.
- Added `CHAI_ENABLE_EXPERIMENTAL` CMake option for enabling experimental features. Defaults to ON.

## [Version 2025.03.1] - Release date 2025-06-23

### Changed
- Updated to RAJA v2025.03.2
- Use memcpy instead of umpire copy for CPU-only thin managed array realloc (allows tracking to be disabled).

### Fixed
- Fixed ManagedArray::set when CHAI\_DISABLE\_RM=OFF and the initial space is not CPU.
- Fixed memory leaks in ArrayManager.

## [Version 2025.03.0] - Release date 2025-03-19

### Added
- Added a ManagedArray::clone function and deprecated chai::deepCopy.

### Fixed
- Fixes reallocate when using pinned or unified memory.
- Adds missing synchronize when using pinned memory.
- Fixes possible hangs when evicting data.
- Now respects allocators passed to ManagedArray constructors when CHAI\_DISABLE\_RM=TRUE.

### Removed
- Removes deprecated ManagedArray::getPointer method. Use ManagedArray::data instead.
- Removes ManagedArray::incr and ManagedArray::decr methods. Use ManagedArray::pick and ManagedArray::set instead.
- Removes optional support for implicitly casting between raw pointers and ManagedArrays (CHAI\_ENABLE\_IMPLICIT\_CONVERSIONS). Use makeManagedArray and ManagedArray::data to perform explicit conversions instead.
- Removes equality and inequality comparison operators between ManagedArrays and raw pointers.
- Removes make\_managed\_from\_factory function for creating managed\_ptr objects from factory functions. This change will lead to safer adoption of allocators during construction and destruction of managed\_ptr objects.
- Removes CHAI\_ENABLE\_PICK CMake option. ManagedArray::pick and ManagedArray::set will always be available.

## [Version 2024.07.0] - Release date 2024-07-26

### Changed
- Updated to Umpire v2024.07.0 and RAJA v2024.07.0
- Increased minimum CMake version required to 3.23
- Turns off benchmarks and examples by default
- Enable RAJA plugin by default

## [Version 2024.02.2] - Release date 2024-06-26

### Changed
- Updated to RAJA v2024.02.2
- Improved debugging with GPU simulation mode

## [Version 2024.02.1] - Release date 2024-04-19

### Changed
- Updated to BLT v0.6.2, Umpire v2024.02.1, and RAJA v2024.02.1

## [Version 2024.02.0] - Release date 2024-03-04

### Added
- Support for APUs with a single memory space. To use, configure with -DCHAI\_DISABLE\_RM=ON -DCHAI\_THIN\_GPU\_ALLOCATE=ON.

### Changed
- Moved installed CMake targets from share/chai/cmake to lib/cmake/chai to be consistent with other libraries in the RAJA Portability Suite
- Improved dependency handling during the build of CHAI and when it is imported into another library/application
- Removed ArrayManager::enableDeviceSynchronization and ArrayManager::disableDeviceSynchronization. Instead, use the environment variables for device synchronization after all kernels (e.g. CUDA\_LAUNCH\_BLOCKING or HIP\_LAUNCH\_BLOCKING)

### Fixed
- Use free instead of realloc when the size is 0 (fixes a warning from valgrind)
