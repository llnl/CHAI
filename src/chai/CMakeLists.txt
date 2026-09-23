##############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
# contributors. See the CHAI LICENSE and COPYRIGHT files for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

include(CMakePackageConfigHelpers)

configure_file(
  ${PROJECT_SOURCE_DIR}/src/chai/config.hpp.in
  ${PROJECT_BINARY_DIR}/include/chai/config.hpp)

set (chai_headers
  ArrayManager.hpp
  ArrayManager.inl
  ChaiMacros.hpp
  DeviceHelpers.hpp
  ExecutionContext.hpp
  ExecutionContextGuard.hpp
  ExecutionContextManager.hpp
  ExecutionSpaces.hpp
  ManagedArray.hpp
  ManagedArray.inl
  managed_ptr.hpp
  PointerRecord.hpp
  Types.hpp)

set (chai_sources
  ArrayManager.cpp)

if(CHAI_DISABLE_RM)
  set(chai_headers
    ${chai_headers}
    ManagedArray_thin.inl)
endif ()

if(CHAI_ENABLE_EXPERIMENTAL)
  set(chai_headers
    ${chai_headers}
    expt/DualArrayManager.hpp
    expt/HostArrayManager.hpp
    expt/HostSharedPointer.hpp
    expt/ManagedArrayPointer.hpp
    expt/ManagedArraySharedPointer.hpp
    expt/ManagedArrayView.hpp
    expt/UnifiedArrayManager.hpp
    ManagedSharedPtr.hpp
    SharedPtrCounter.hpp
    SharedPtrManager.hpp
    SharedPtrManager.inl
    SharedPointerRecord.hpp)

  set (chai_sources
    ${chai_sources}
    SharedPtrManager.cpp)
endif()

set (chai_depends
  ${CHAI_UMPIRE_TARGET})

if (CHAI_ENABLE_CUDA)
  set (chai_depends
    ${chai_depends}
    cuda_runtime)
endif ()
if (CHAI_ENABLE_HIP)
  set (chai_depends
    ${chai_depends}
    blt::hip_runtime)
endif ()

if (CHAI_ENABLE_RAJA_PLUGIN)
  set (chai_headers
    ${chai_headers}
    pluginLinker.hpp
    ManagedArrayView.hpp
    ExecutionContextRAJAPlugin.hpp)

  set (chai_sources
    ${chai_sources}
    ExecutionContextRAJAPlugin.cpp
    ExecutionContextRAJAPluginRegistration.cpp)

  set (chai_depends
    ${chai_depends}
    RAJA)

  if (CHAI_ENABLE_CUDA)
    set (chai_depends
      ${chai_depends}
      cuda)
  endif ()
endif ()

blt_add_library(
  NAME chai
  SOURCES ${chai_sources}
  HEADERS ${chai_headers}
  DEPENDS_ON ${chai_depends})

install(FILES ${chai_headers} DESTINATION include/chai/)

install(FILES ../util/forall.hpp DESTINATION include/chai/util/)

target_include_directories(
  chai
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:include>)

configure_package_config_file(
  ${PROJECT_SOURCE_DIR}/cmake/chai-config.cmake.in
  ${PROJECT_BINARY_DIR}/chai-config.cmake
  INSTALL_DESTINATION lib/cmake/chai)

install(
  FILES ${PROJECT_BINARY_DIR}/chai-config.cmake
  DESTINATION lib/cmake/chai)

write_basic_package_version_file(
  ${PROJECT_BINARY_DIR}/chai-config-version.cmake
  COMPATIBILITY SameMajorVersion)

install(FILES
  ${PROJECT_BINARY_DIR}/chai-config-version.cmake
  DESTINATION lib/cmake/chai)

install(
  FILES ${PROJECT_BINARY_DIR}/include/chai/config.hpp
  DESTINATION include/chai)

install(
  FILES ${chai_headers}
  DESTINATION include/chai)

install(
  TARGETS chai
  EXPORT chai-targets
  RUNTIME DESTINATION lib
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

install(EXPORT chai-targets DESTINATION lib/cmake/chai)

blt_install_tpl_setups(DESTINATION lib/cmake/chai)
