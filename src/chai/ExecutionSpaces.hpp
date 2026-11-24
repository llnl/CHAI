//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#ifndef CHAI_ExecutionSpaces_HPP
#define CHAI_ExecutionSpaces_HPP

#include "chai/config.hpp"

namespace chai
{

/*!
 * \brief Enum listing possible execution spaces.
 */
enum ExecutionSpace {
  /*! Default, no execution space. */
  NONE = 0,
  /*! Executing in CPU space */
  CPU,
#if defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)
  /*! Execution in GPU space */
  GPU,
#endif
#if defined(CHAI_ENABLE_UM)
  UM,
#endif
#if defined(CHAI_ENABLE_PINNED)
  PINNED,
#endif
  // NUM_EXECUTION_SPACES should always be last!
  /*! Used to count total number of spaces */
  NUM_EXECUTION_SPACES
};

}  // end of namespace chai

#endif  // CHAI_ExecutionSpaces_HPP
