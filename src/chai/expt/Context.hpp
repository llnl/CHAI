//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CHAI_CONTEXT_HPP
#define CHAI_CONTEXT_HPP

namespace chai::expt
{
  /*!
   * \brief Execution context identifier.
   */
  enum class Context
  {
    HOST = 0,   /*!< Host (CPU) context. */
    DEVICE = 1  /*!< Device (GPU/accelerator) context. */
  };  // enum class Context
}  // namespace chai::expt

#endif  // CHAI_CONTEXT_HPP
