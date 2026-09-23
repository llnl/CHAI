//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CHAI_EXECUTION_CONTEXT_HPP
#define CHAI_EXECUTION_CONTEXT_HPP

namespace chai
{
  /*!
   * \brief Execution context identifier.
   */
  enum class ExecutionContext
  {
    NONE = 0,   /*!< No context. */
    HOST = 1,   /*!< Host (CPU) context. */
    DEVICE = 2  /*!< Device (GPU/accelerator) context. */
  };  // enum class ExecutionContext
}  // namespace chai

#endif  // CHAI_EXECUTION_CONTEXT_HPP
