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

-------
Context
-------

Currently, there are two execution contexts that are handled by CHAI. These are represented in the `Context` enum class.
The `HOST` enum value represents synchronous execution on a CPU. The `DEVICE` enum value represents asynchronous execution on a GPU.
Both NVIDIA and AMD GPUs are supported.

--------------
ContextManager
--------------

Implicitly managing data coherence requires managing some global state. This is handled by a singleton called `ContextManager`.
When an application enters an execution context, it uses `ContextManager` to set the current context. `ContextManager` also
tracks which contexts may need synchronization. CHAI data structures can query `ContextManager` to update data coherence and
inform `ContextManager` of needed synchronization or synchronization that has been performed.

Note: It is much faster for `ContextManager` to track synchronization than to repeatedly call `cudaDeviceSynchronize()` or `hipDeviceSynchronize()`.

.. code-block:: cpp

  #include "chai/expt/ContextManager.hpp"
   
  ::chai::expt::ContextManager& contextManager = ::chai::expt::ContextManager::getInstance();

  contextManager.setContext(::chai::expt::Context::HOST);
  // Use CHAI data structures in the HOST context...
  contextManager.setContext(::chai::expt::Context::NONE);

  contextManager.setContext(::chai::expt::Context::DEVICE);
  // Use CHAI data structures in the DEVICE context...
  contextManager.setContext(::chai::expt::Context::NONE);
   
------------
ContextGuard
------------

It is easy to forget to reset the current context or even to forget the current context
when writing code. Similar to `std::lock_guard`, CHAI provides `ContextGuard` that sets
the active context and then resets it upon destruction. This is the recommended approach.

.. code-block:: cpp

  #include "chai/expt/ContextGuard.hpp"

  {
    ::chai::expt::ContextGuard contextGuard{::chai::expt::Context::HOST};
    // Use CHAI data structures in the HOST context...
  }

  {
    ::chai::expt::ContextGuard contextGuard{::chai::expt::Context::DEVICE};
    // Use CHAI data structures in the DEVICE context...
  }

-----------------
ContextRAJAPlugin
-----------------

In an application that also uses RAJA, CHAI provides a RAJA plugin, `ContextRAJAPlugin`,
that implicitly manages the context in calls to RAJA. To enable this plugin, configure with
`-DCHAI_ENABLE_EXPERIMENTAL_RAJA_PLUGIN=ON` and register the plugin. In the future, registration
may be handled by CHAI.

.. code-block:: cpp

  #include "chai/expt/ContextRAJAPlugin.hpp"
  #include "RAJA/RAJA.hpp"

  static ::RAJA::util::PluginRegistry::add<chai::expt::ContextRAJAPlugin> P(
    "CHAIContextPlugin",
    "Plugin that integrates CHAI context management with RAJA.");
   
  ::RAJA::forall<::RAJA::seq_exec>(::RAJA::TypedRangeSegment<int>(0, N), [=] (int i) {
    // Use CHAI data structures in the HOST context...
  });

  constexpr int BLOCK_SIZE = 256;

  ::RAJA::forall<::RAJA::cuda_exec_async<BLOCK_SIZE>>(::RAJA::TypedRangeSegment<int>(0, N), [=] __device__ (int i) {
    // Use CHAI data structures in the DEVICE context...
  });

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

  #include "chai/expt/ContextRAJAPlugin.hpp"
  #include "chai/expt/ManagedArrayPointer.hpp"
  #include "RAJA/RAJA.hpp"

  static ::RAJA::util::PluginRegistry::add<chai::expt::ContextRAJAPlugin> P(
    "CHAIContextPlugin",
    "Plugin that integrates CHAI context management with RAJA.");

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

  #include "chai/expt/ContextRAJAPlugin.hpp"
  #include "chai/expt/ManagedArraySharedPointer.hpp"
  #include "RAJA/RAJA.hpp"

  static ::RAJA::util::PluginRegistry::add<chai::expt::ContextRAJAPlugin> P(
    "CHAIContextPlugin",
    "Plugin that integrates CHAI context management with RAJA.");

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

  #include "chai/expt/ContextRAJAPlugin.hpp"
  #include "chai/expt/ManagedArrayView.hpp"
  #include "chai/expt/UnifiedArrayManager.hpp"
  #include "RAJA/RAJA.hpp"

  static ::RAJA::util::PluginRegistry::add<chai::expt::ContextRAJAPlugin> P(
    "CHAIContextPlugin",
    "Plugin that integrates CHAI context management with RAJA.");

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

HostArrayManager does not rely on ContextManager, so it will behave
differently than other array managers. The major difference is that
when used with ManagedArrayPointer, the ManagedArrayPointer does not
need the update method called or to be copy constructed before it can
be used on the host. Since it will not respect the current Context,
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
SharedArrayManager
------------------

``SharedArrayManager`` manages a host array in memory shared by processes in
an MPI sharing scope. This is separate from ``ManagedArraySharedPointer``:
``SharedArrayManager`` selects where the elements are allocated, while
``ManagedArraySharedPointer`` provides shared ownership and automatic cleanup
within each process.

The manager uses Umpire's ``SHARED`` resources and supports both POSIX and
MPI3 shared-memory allocators. By default it uses the global ``SHARED``
allocator and ``MPI_COMM_WORLD``. An allocator can also be supplied explicitly;
for POSIX resources, an optional parent communicator controls how Umpire
derives the node-local sharing communicator. MPI3 resources use the
communicator owned by the allocator.

.. code-block:: cpp

  #include "chai/expt/ManagedArrayPointer.hpp"
  #include "chai/expt/SharedArrayManager.hpp"

  template <typename T>
  using SharedArrayManager = ::chai::expt::SharedArrayManager<T>;

  template <typename T>
  using SharedArrayPointer =
    ::chai::expt::ManagedArrayPointer<T, SharedArrayManager<T>>;

  // Every rank in the sharing communicator performs these calls in the same
  // order. The returned pointer is the rank's local mapping of shared storage.
  SharedArrayPointer<int> a = SharedArrayPointer<int>::make(1024);
  int* data = a.data();

  int shared_rank = 0;
  MPI_Comm shared_comm = ::umpire::get_communicator_for_allocator(
    ::umpire::ResourceManager::getInstance().getAllocator("SHARED"),
    MPI_COMM_WORLD);
  MPI_Comm_rank(shared_comm, &shared_rank);

  if (shared_rank == 0) {
    data[0] = 42;
  }

  // data() and update() synchronize before returning on the host.
  a.update();
  // Every rank can now read a[0].

  a.free();
  ::umpire::cleanup_cached_communicators();

Construction, manager copying, resizing, ``data()``, and destruction of an
allocated manager are collective. Wrapper copy constructors that call
``update()`` are therefore also collective. All participating ranks must make
these calls in the same order; calling them on only some ranks can deadlock.
``size()`` and access through a cached pointer or ``operator[]`` do not
synchronize. Direct manager users can call ``synchronize()`` explicitly.

Manager copies are deep: each copy receives a distinct shared allocation.
Copies of ``ManagedArrayPointer``, ``ManagedArraySharedPointer``, and
``ManagedArrayView`` retain their documented shallow semantics. Resizing
preserves the existing prefix, but newly allocated and newly grown elements
are left uninitialized. Element types must be trivially copyable, default
constructible, and trivially destructible.

The default Umpire shared resource has a capacity of 16 MiB. Applications that
need more capacity should create a ``SHARED::POSIX`` or ``SHARED::MPI3``
resource with larger traits and pass that allocator to the manager. The
manager supplies allocation names automatically, including for raw POSIX
allocators.

``SharedArrayManager`` borrows the communicator returned by Umpire. Destroy all
managers before calling ``umpire::cleanup_cached_communicators()``, and perform
that cleanup before ``MPI_Finalize()``. Umpire caches POSIX communicators by
allocator ID, so use a consistent parent communicator for a given POSIX
allocator.

------------------
UnifiedArrayManager
------------------

``UnifiedArrayManager`` manages a single contiguous array allocated from Umpire's
unified (managed) memory allocator (``UM``). Since unified memory is accessible
from both CPU and GPU, CHAI can provide a single pointer value that is valid in
both the ``HOST`` and ``DEVICE`` contexts.

``UnifiedArrayManager`` relies on :ref:`ContextManager <experimental_design>` to
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

  #include "chai/expt/Context.hpp"
  #include "chai/expt/ContextGuard.hpp"
  #include "chai/expt/UnifiedArrayManager.hpp"

  const std::size_t N = 1000000;
  ::chai::expt::UnifiedArrayManager<int> a{N};

  {
    ::chai::expt::ContextGuard guard{::chai::expt::Context::HOST};
    int* p = a.data(true);
    for (std::size_t i = 0; i < N; ++i) {
      p[i] = static_cast<int>(i);
    }
  }

  {
    ::chai::expt::ContextGuard guard{::chai::expt::Context::DEVICE};
    int* p = a.data(true);
    // Launch a CUDA/HIP kernel that writes through p...
  }

  {
    ::chai::expt::ContextGuard guard{::chai::expt::Context::HOST};
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

``DualArrayManager`` relies on :ref:`ContextManager <experimental_design>` in
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

  #include "chai/expt/Context.hpp"
  #include "chai/expt/ContextGuard.hpp"
  #include "chai/expt/DualArrayManager.hpp"

  const std::size_t N = 1000000;
  ::chai::expt::DualArrayManager<int> a{N};

  {
    ::chai::expt::ContextGuard guard{::chai::expt::Context::HOST};
    int* p = a.data(true);
    for (std::size_t i = 0; i < N; ++i) {
      p[i] = static_cast<int>(i);
    }
  }

  {
    ::chai::expt::ContextGuard guard{::chai::expt::Context::DEVICE};
    int* p = a.data(true);
    // If the most recent modification was on HOST, this call copies to DEVICE first.
    // Launch a CUDA/HIP kernel that writes through p...
  }

  {
    ::chai::expt::ContextGuard guard{::chai::expt::Context::HOST};
    // If the most recent modification was on DEVICE, this call synchronizes and copies first.
    const int* p = a.data(false);
    // Read through p...
  }
