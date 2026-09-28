..
    # Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
    # contributors. See the CHAI LICENSE and COPYRIGHT files for details.
    #
    # SPDX-License-Identifier: BSD-3-Clause

.. _execution_contexts:

==================
Execution Contexts
==================

CHAI uses an execution context to identify where work is about to execute.
``chai::ExecutionContext`` provides ``HOST``, ``DEVICE``, and ``NONE`` values.
The ``chai::ExecutionContextManager`` singleton stores the current context and
tracks whether asynchronous device work needs to be synchronized.

Applications can set the context directly:

.. code-block:: cpp

  #include "chai/ExecutionContextManager.hpp"

  auto& manager = chai::ExecutionContextManager::getInstance();
  manager.setContext(chai::ExecutionContext::HOST);
  // Use CHAI data structures on the host...
  manager.setContext(chai::ExecutionContext::NONE);

For scoped changes, ``chai::ExecutionContextGuard`` is preferred. It restores
the preceding context when the guard is destroyed, including when guards are
nested:

.. code-block:: cpp

  #include "chai/ExecutionContextGuard.hpp"

  {
    chai::ExecutionContextGuard guard{chai::ExecutionContext::DEVICE};
    // Launch work that uses CHAI data structures on the device...
  }

``ArrayManager::setExecutionSpace``, ``ArrayManager::getExecutionSpace``, and
``ArrayManager::syncIfNeeded`` remain available. They use the same state as
``ExecutionContextManager``. The ``CPU`` and ``GPU`` execution spaces correspond
to the ``HOST`` and ``DEVICE`` contexts, respectively. Execution spaces without
a direct context equivalent are reported as ``ExecutionContext::NONE`` by
``getContext()``.

RAJA integration
----------------

When CHAI is configured with ``CHAI_ENABLE_RAJA_PLUGIN=ON``, the
``chai::ExecutionContextRAJAPlugin`` is built and registered automatically. It
sets the execution context before RAJA captures a kernel body and resets it
after capture. Applications do not need to include the plugin header or register
the plugin themselves:

.. code-block:: cpp

  #include "RAJA/RAJA.hpp"

  RAJA::forall<RAJA::seq_exec>(RAJA::TypedRangeSegment<int>(0, N),
    [=](int i) {
      // CHAI objects captured here observe ExecutionContext::HOST.
    });

For CUDA and HIP execution policies, captured CHAI objects similarly observe
``ExecutionContext::DEVICE``.
