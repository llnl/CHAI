//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "chai/ExecutionContextRAJAPlugin.hpp"

#include "chai/Types.hpp"

RAJA_INSTANTIATE_REGISTRY(RAJA::util::PluginRegistry);

// This is needed to link a dynamic lib as RAJA does not provide an exported
// definition of this symbol.
#if defined(_WIN32) && !defined(CHAISTATICLIB)
#ifdef CHAISHAREDDLL_EXPORTS
namespace RAJA
{
namespace util
{
PluginStrategy::PluginStrategy() = default;
}  // namespace util
}  // namespace RAJA
#endif
#endif

static RAJA::util::PluginRegistry::add<chai::ExecutionContextRAJAPlugin> P(
    "ExecutionContextRAJAPlugin",
    "Plugin to set the CHAI execution context from the RAJA platform");

namespace chai
{
CHAISHAREDDLL_API void linkExecutionContextRAJAPlugin()
{
}
}  // namespace chai
