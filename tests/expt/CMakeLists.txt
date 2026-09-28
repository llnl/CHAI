##############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
# contributors. See the CHAI LICENSE and COPYRIGHT files for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

set(chai_expt_test_headers
    TestHelpers.hpp)

set(chai_expt_test_depends
    chai
    gtest)

if(ENABLE_CUDA)
  set(chai_expt_test_depends
      ${chai_expt_test_depends}
      cuda)
endif()

if(ENABLE_HIP)
  set(chai_expt_test_depends
      ${chai_expt_test_depends}
      blt::hip)
endif()

blt_add_executable(
  NAME ManagedArrayPointerTests
  SOURCES ManagedArrayPointerTests.cpp
  HEADERS ${chai_expt_test_headers}
  INCLUDES ${PROJECT_BINARY_DIR}/include
  DEPENDS_ON ${chai_expt_test_depends})

blt_add_test(
  NAME ManagedArrayPointerTests
  COMMAND ManagedArrayPointerTests)

blt_add_executable(
  NAME ManagedArrayViewTests
  SOURCES ManagedArrayViewTests.cpp
  HEADERS ${chai_expt_test_headers}
  INCLUDES ${PROJECT_BINARY_DIR}/include
  DEPENDS_ON ${chai_expt_test_depends})

blt_add_test(
  NAME ManagedArrayViewTests
  COMMAND ManagedArrayViewTests)

blt_add_executable(
  NAME HostArrayPointerTests
  SOURCES HostArrayPointerTests.cpp
  HEADERS ${chai_expt_test_headers}
  INCLUDES ${PROJECT_BINARY_DIR}/include
  DEPENDS_ON ${chai_expt_test_depends})

blt_add_test(
  NAME HostArrayPointerTests
  COMMAND HostArrayPointerTests)

blt_add_executable(
  NAME HostArrayViewTests
  SOURCES HostArrayViewTests.cpp
  HEADERS ${chai_expt_test_headers}
  INCLUDES ${PROJECT_BINARY_DIR}/include
  DEPENDS_ON ${chai_expt_test_depends})

blt_add_test(
  NAME HostArrayViewTests
  COMMAND HostArrayViewTests)

blt_add_executable(
  NAME HostArraySharedPointerTests
  SOURCES HostArraySharedPointerTests.cpp
  HEADERS ${chai_expt_test_headers}
  INCLUDES ${PROJECT_BINARY_DIR}/include
  DEPENDS_ON ${chai_expt_test_depends})

blt_add_test(
  NAME HostArraySharedPointerTests
  COMMAND HostArraySharedPointerTests)

if(ENABLE_CUDA OR ENABLE_HIP)
  blt_add_executable(
    NAME DualArrayManagerTests
    SOURCES DualArrayManagerTests.cpp
    HEADERS ${chai_expt_test_headers}
    INCLUDES ${PROJECT_BINARY_DIR}/include
    DEPENDS_ON ${chai_expt_test_depends})

  blt_add_test(
    NAME DualArrayManagerTests
    COMMAND DualArrayManagerTests)

  blt_add_executable(
    NAME UnifiedArrayManagerTests
    SOURCES UnifiedArrayManagerTests.cpp
    HEADERS ${chai_expt_test_headers}
    INCLUDES ${PROJECT_BINARY_DIR}/include
    DEPENDS_ON ${chai_expt_test_depends})

  blt_add_test(
    NAME UnifiedArrayManagerTests
    COMMAND UnifiedArrayManagerTests)
endif()
