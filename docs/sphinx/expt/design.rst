..
    # Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
    # contributors. See the CHAI LICENSE and COPYRIGHT files for details.
    #
    # SPDX-License-Identifier: BSD-3-Clause

.. _experimental_design:

*******************
Experimental Design
*******************

CHAI provides data structures that implicitly manage coherence across multiple execution contexts.

------------------
Execution contexts
------------------

The experimental array managers use CHAI's core ``ExecutionContext``,
``ExecutionContextManager``, and ``ExecutionContextGuard`` APIs. See
:ref:`execution_contexts` for usage and RAJA integration. When
``CHAI_ENABLE_RAJA_PLUGIN`` is enabled, CHAI registers its execution-context
plugin automatically.

-------------------
ManagedArrayPointer
-------------------

This class provides a uniform interface for working with different types of memory
across multiple backends. It has pointer semantics, meaning that copies are shallow,
which allows this object to be passed by value to a CUDA or HIP kernel. When copy
constructed, it queries the array manager to update the cached size and pointer
from the array manager. The array must be explicitly freed from only one of the
shallow copies.

.. code-block:: cpp

  #include "chai/expt/ManagedArrayPointer.hpp"
  #include "RAJA/RAJA.hpp"

  // It's recommended to use an alias so that it is easy to swap out the array manager.
  template <typename T>
  using ManagedArrayPointer = ::chai::expt::ManagedArrayPointer<T, UnifiedArrayManager>;

  const std::size_t N = 1000000;
  ManagedArrayPointer<int> a;
  a.resize(N);

  ::RAJA::forall<::RAJA::seq_exec>(::RAJA::TypedRangeSegment<int>(0, N), [=] (int i) {
    a[i] = i;  // Use CHAI data structures in the HOST context...
  });

  constexpr int BLOCK_SIZE = 256;

  ::RAJA::forall<::RAJA::cuda_exec_async<BLOCK_SIZE>>(::RAJA::TypedRangeSegment<int>(0, N), [=] __device__ (int i) {
    a[i] -= 1;  // Use CHAI data structures in the DEVICE context...
  });

  a.free();

-------------------------
ManagedArraySharedPointer
-------------------------

This class provides a uniform interface for working with different types of memory
across multiple backends. It has shared pointer semantics, meaning that copies are
shallow, which allows this object to be passed by value to a CUDA or HIP kernel.
When copy constructed, it queries the array manager to update the cached size and
pointer from the array manager. Similar to std::shared_ptr, all host copies have
shared ownership such that when the last host copy is destroyed, it will trigger
clean up of the underlying resources. Note that device copies are not reference
counted since clean up cannot be triggered from the device.

.. code-block:: cpp

  #include "chai/expt/ManagedArraySharedPointer.hpp"
  #include "RAJA/RAJA.hpp"

  // It's recommended to use an alias so that it is easy to swap out the array manager.
  template <typename T>
  using ManagedArraySharedPointer = ::chai::expt::ManagedArraySharedPointer<T, UnifiedArrayManager>;

  const std::size_t N = 1000000;
  ManagedArraySharedPointer<int> a;
  a.resize(N);

  ::RAJA::forall<::RAJA::seq_exec>(::RAJA::TypedRangeSegment<int>(0, N), [=] (int i) {
    a[i] = i;  // Use CHAI data structures in the HOST context...
  });

  constexpr int BLOCK_SIZE = 256;

  ::RAJA::forall<::RAJA::cuda_exec_async<BLOCK_SIZE>>(::RAJA::TypedRangeSegment<int>(0, N), [=] __device__ (int i) {
    a[i] -= 1;  // Use CHAI data structures in the DEVICE context...
  });

----------------
ManagedArrayView
----------------

This class provides a uniform interface for viewing different types of memory
across multiple backends. It has shallow-copy semantics, which allows it to be
passed by value to a CUDA or HIP kernel. When copy constructed, it queries the
array manager to update the cached size and pointer from the array manager.
Unlike ``ManagedArrayPointer``, this class is non-owning: it cannot allocate,
resize, or free the underlying data, and the referenced manager must outlive
the view.

.. code-block:: cpp

  #include "chai/expt/ManagedArrayView.hpp"
  #include "chai/expt/UnifiedArrayManager.hpp"
  #include "RAJA/RAJA.hpp"

  // It's recommended to use an alias so that it is easy to swap out the array manager.
  template <typename T>
  using UnifiedArrayManager = ::chai::expt::UnifiedArrayManager<T>;

  template <typename T>
  using ManagedArrayView = ::chai::expt::ManagedArrayView<T, UnifiedArrayManager<T>>;

  const std::size_t N = 1000000;
  UnifiedArrayManager<int> manager{N};
  ManagedArrayView<int> a{manager};

  ::RAJA::forall<::RAJA::seq_exec>(::RAJA::TypedRangeSegment<int>(0, N), [=] (int i) {
    a[i] = i;  // Use CHAI data structures in the HOST context...
  });

  constexpr int BLOCK_SIZE = 256;

  ::RAJA::forall<::RAJA::cuda_exec_async<BLOCK_SIZE>>(::RAJA::TypedRangeSegment<int>(0, N), [=] __device__ (int i) {
    a[i] -= 1;  // Use CHAI data structures in the DEVICE context...
  });

----------------
HostArrayManager
----------------

This class manages a host array. It is designed for use with ManagedArrayPointer.

HostArrayManager performs value initialization of each array element.
That is to say, numeric types will be initialized to zero and nontrivial
types will be default constructed. In the future, this behavior may
change to default initialization for performance reasons, such that
numeric types will be left in an indeterminate state and nontrivial
types will be default constructed.

HostArrayManager does not rely on ExecutionContextManager, so it will behave
differently than other array managers. The major difference is that
when used with ManagedArrayPointer, the ManagedArrayPointer does not
need the update method called or to be copy constructed before it can
be used on the host. Since it will not respect the current execution context,
be extra careful to avoid using it on the device.

.. code-block:: cpp

  #include "chai/expt/HostArrayManager.hpp"
  #include "chai/expt/ManagedArrayPointer.hpp"

  // It's recommended to use an alias so that it is easy to swap out the array manager.
  template <typename T>
  using HostArrayManager = ::chai::expt::HostArrayManager<T>;

  template <typename T>
  using ManagedArrayPointer = ::chai::expt::ManagedArrayPointer<T, HostArrayManager<T>>;

  const std::size_t N = 1000000;
  ManagedArrayPointer<int> a{HostArrayManager<int>(N)};

  for (std::size_t i = 0; i < N; ++i)
  {
    a[i] = i;
  }

------------------
UnifiedArrayManager
------------------

``UnifiedArrayManager`` manages a single contiguous array allocated from Umpire's
unified (managed) memory allocator (``UM``). Since unified memory is accessible
from both CPU and GPU, CHAI can provide a single pointer value that is valid in
both the ``HOST`` and ``DEVICE`` contexts.

``UnifiedArrayManager`` relies on :ref:`ExecutionContextManager <execution_contexts>` to
avoid unnecessary full device synchronizations and to ensure correctness when
switching between contexts:

- ``data(touch=false)`` returns a pointer suitable for read access in the current
  context and synchronizes with the most recent modifying context (if needed).
- ``data(touch=true)`` indicates the caller will modify the array in the current
  context; it performs any required synchronization first, then records the
  current context as the most recently modified context.

Like ``HostArrayManager``, ``UnifiedArrayManager`` performs value initialization
of each element on the host (numeric types are initialized to zero).

.. code-block:: cpp

  #include "chai/ExecutionContextGuard.hpp"
  #include "chai/expt/UnifiedArrayManager.hpp"

  const std::size_t N = 1000000;
  ::chai::expt::UnifiedArrayManager<int> a{N};

  {
    ::chai::ExecutionContextGuard guard{::chai::ExecutionContext::HOST};
    int* p = a.data(true);
    for (std::size_t i = 0; i < N; ++i) {
      p[i] = static_cast<int>(i);
    }
  }

  {
    ::chai::ExecutionContextGuard guard{::chai::ExecutionContext::DEVICE};
    int* p = a.data(true);
    // Launch a CUDA/HIP kernel that writes through p...
  }

  {
    ::chai::ExecutionContextGuard guard{::chai::ExecutionContext::HOST};
    // If the most recent modification was on DEVICE, this call synchronizes first.
    const int* p = a.data(false);
    // Read through p...
  }

----------------
DualArrayManager
----------------

``DualArrayManager`` manages separate host and device allocations and keeps them
coherent explicitly. Unlike ``UnifiedArrayManager``, it does not rely on a
single unified-memory pointer that is valid in both contexts. Instead, it
allocates storage lazily in the requested context and copies data between the
two allocations when the authoritative copy lives in the other context.

``DualArrayManager`` relies on :ref:`ExecutionContextManager <execution_contexts>` in
the same way as ``UnifiedArrayManager``:

- ``data(touch=false)`` returns a pointer suitable for read access in the
  current context and synchronizes with the most recent modifying context when
  needed.
- ``data(touch=true)`` indicates the caller will modify the array in the current
  context; it performs any required synchronization first, then records the
  current context as authoritative.

``DualArrayManager`` also performs value initialization of each element when a
new allocation is created. Since it uses raw host/device allocations and
byte-wise copies between them, it is intended for trivially copyable element
types.

This manager is useful when an application wants explicit mirrored allocations
in host and device memory rather than unified memory.

.. code-block:: cpp

  #include "chai/ExecutionContextGuard.hpp"
  #include "chai/expt/DualArrayManager.hpp"

  const std::size_t N = 1000000;
  ::chai::expt::DualArrayManager<int> a{N};

  {
    ::chai::ExecutionContextGuard guard{::chai::ExecutionContext::HOST};
    int* p = a.data(true);
    for (std::size_t i = 0; i < N; ++i) {
      p[i] = static_cast<int>(i);
    }
  }

  {
    ::chai::ExecutionContextGuard guard{::chai::ExecutionContext::DEVICE};
    int* p = a.data(true);
    // If the most recent modification was on HOST, this call copies to DEVICE first.
    // Launch a CUDA/HIP kernel that writes through p...
  }

  {
    ::chai::ExecutionContextGuard guard{::chai::ExecutionContext::HOST};
    // If the most recent modification was on DEVICE, this call synchronizes and copies first.
    const int* p = a.data(false);
    // Read through p...
  }
