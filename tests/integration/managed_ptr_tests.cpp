//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CHAI
// contributors. See the CHAI LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
#include "gtest/gtest.h"

#define GPU_TEST(X, Y)              \
  static void gpu_test_##X##Y();    \
  TEST(X, Y) { gpu_test_##X##Y(); } \
  static void gpu_test_##X##Y()

#include "chai/config.hpp"
#include "chai/ArrayManager.hpp"
#include "chai/ManagedArray.hpp"
#include "chai/managed_ptr.hpp"

#include "../src/util/forall.hpp"

// Standard library headers
#include <cstdlib>

class Base1 {
   public:
      CHAI_HOST_DEVICE Base1() {}
      CHAI_HOST_DEVICE virtual ~Base1() {}

      CHAI_HOST_DEVICE virtual bool isBase1() { return true; }
};

class Base2 {
   public:
      CHAI_HOST_DEVICE Base2() {}
      CHAI_HOST_DEVICE virtual ~Base2() {}

      CHAI_HOST_DEVICE virtual bool isBase2() { return true; }
};

class ClassWithMultipleInheritance : public Base1, public Base2 {
   public:
      CHAI_HOST_DEVICE ClassWithMultipleInheritance() : Base1(), Base2() {}
      CHAI_HOST_DEVICE virtual ~ClassWithMultipleInheritance() {}
};

class RawArrayClass {
   public:
      CHAI_HOST_DEVICE RawArrayClass() : m_values(nullptr) {}
      CHAI_HOST_DEVICE RawArrayClass(int* values) : m_values(values) {}

      CHAI_HOST_DEVICE ~RawArrayClass() {}

      CHAI_HOST_DEVICE int getValue(const int i) const { return m_values[i]; }

   private:
      int* m_values;
};

class RawPointerClass {
   public:
      CHAI_HOST_DEVICE RawPointerClass() : m_innerClass(nullptr) {}
      CHAI_HOST_DEVICE RawPointerClass(RawArrayClass* innerClass) : m_innerClass(innerClass) {}

      CHAI_HOST_DEVICE ~RawPointerClass() {}

      CHAI_HOST_DEVICE int getValue(const int i) const { return m_innerClass->getValue(i); }

   private:
      RawArrayClass* m_innerClass;
};

class TestBase {
   public:
      CHAI_HOST_DEVICE TestBase() {}
      CHAI_HOST_DEVICE virtual ~TestBase() {}

      CHAI_HOST_DEVICE virtual int getValue(const int i) const = 0;
};

class TestDerived : public TestBase {
   public:
      CHAI_HOST_DEVICE TestDerived() : TestBase(), m_values(nullptr) {}
      CHAI_HOST_DEVICE TestDerived(chai::ManagedArray<int> values) : TestBase(), m_values(values) {}
      CHAI_HOST_DEVICE virtual ~TestDerived() {}

      CHAI_HOST_DEVICE virtual int getValue(const int i) const { return m_values[i]; }

   private:
      chai::ManagedArray<int> m_values;
};

class TestInnerBase {
   public:
      CHAI_HOST_DEVICE TestInnerBase() {}
      CHAI_HOST_DEVICE virtual ~TestInnerBase() {}

      CHAI_HOST_DEVICE virtual int getValue() = 0;
};

class TestInner : public TestInnerBase {
   public:
      CHAI_HOST_DEVICE TestInner() : TestInnerBase(), m_value(0) {}
      CHAI_HOST_DEVICE TestInner(int value) : TestInnerBase(), m_value(value) {}
      CHAI_HOST_DEVICE virtual ~TestInner() {}

      CHAI_HOST_DEVICE virtual int getValue() { return m_value; }

   private:
      int m_value;
};

class TestContainer {
   public:
      CHAI_HOST_DEVICE TestContainer() : m_innerType(nullptr) {}
      CHAI_HOST_DEVICE TestContainer(chai::managed_ptr<TestInner> innerType) : m_innerType(innerType) {}

      CHAI_HOST_DEVICE ~TestContainer() {}

      CHAI_HOST_DEVICE int getValue() const {
         return m_innerType->getValue();
      }

   private:
      chai::managed_ptr<TestInner> m_innerType;
};

class MultipleRawArrayClass {
   public:
      CHAI_HOST_DEVICE MultipleRawArrayClass() : m_values1(nullptr), m_values2(nullptr) {}
      CHAI_HOST_DEVICE MultipleRawArrayClass(int* values1, int* values2) :
         m_values1(values1),
         m_values2(values2)
      {}

      CHAI_HOST_DEVICE ~MultipleRawArrayClass() {}

      CHAI_HOST_DEVICE int getValue(const int i, const int j) const {
         if (i == 0) {
            return m_values1[j];
         }
         else if (i == 1) {
            return m_values2[j];
         }
         else {
            return -1;
         }
      }

   private:
      int* m_values1;
      int* m_values2;
};

TEST(managed_ptr, class_with_raw_array)
{
  const int expectedValue = rand();

  chai::ManagedArray<int> array(1, chai::CPU);

  forall(sequential(), 0, 1, [=] (int i) {
    array[i] = expectedValue;
  });

  auto rawArrayClass = chai::make_managed<RawArrayClass>(chai::unpack(array));

  ASSERT_EQ(rawArrayClass->getValue(0), expectedValue);

  rawArrayClass.free();
  array.free();
}

TEST(managed_ptr, class_with_multiple_raw_arrays)
{
  const int expectedValue1 = rand();
  const int expectedValue2 = rand();

  chai::ManagedArray<int> array1(1, chai::CPU);
  chai::ManagedArray<int> array2(1, chai::CPU);

  forall(sequential(), 0, 1, [=] (int i) {
     array1[i] = expectedValue1;
     array2[i] = expectedValue2;
  });

  auto multipleRawArrayClass = chai::make_managed<MultipleRawArrayClass>(
                                 chai::unpack(array1),
                                 chai::unpack(array2));

  ASSERT_EQ(multipleRawArrayClass->getValue(0, 0), expectedValue1);
  ASSERT_EQ(multipleRawArrayClass->getValue(1, 0), expectedValue2);

  multipleRawArrayClass.free();
  array2.free();
  array1.free();
}

TEST(managed_ptr, class_with_managed_array)
{
  const int expectedValue = rand();

  chai::ManagedArray<int> array(1, chai::CPU);

  forall(sequential(), 0, 1, [=] (int i) {
     array[i] = expectedValue;
  });

  auto derived = chai::make_managed<TestDerived>(array);

  ASSERT_EQ(derived->getValue(0), expectedValue);

  derived.free();
  array.free();
}

TEST(managed_ptr, class_with_raw_ptr)
{
  const int expectedValue = rand();

  chai::ManagedArray<int> array(1, chai::CPU);

  forall(sequential(), 0, 1, [=] (int i) {
     array[i] = expectedValue;
  });

  auto rawArrayClass = chai::make_managed<RawArrayClass>(chai::unpack(array));
  auto rawPointerClass = chai::make_managed<RawPointerClass>(
                           chai::unpack(rawArrayClass));

  ASSERT_EQ((*rawPointerClass).getValue(0), expectedValue);

  rawPointerClass.free();
  rawArrayClass.free();
  array.free();
}

TEST(managed_ptr, class_with_managed_ptr)
{
  const int expectedValue = rand();

  auto derived = chai::make_managed<TestInner>(expectedValue);
  TestContainer container(derived);

  ASSERT_EQ(container.getValue(), expectedValue);

  derived.free();
}

TEST(managed_ptr, nested_managed_ptr)
{
  const int expectedValue = rand();

  auto derived = chai::make_managed<TestInner>(expectedValue);
  auto container = chai::make_managed<TestContainer>(derived);

  ASSERT_EQ(container->getValue(), expectedValue);

  container.free();
  derived.free();
}

TEST(managed_ptr, array_of_managed_ptr)
{
  int numManagedPointers = 10;

  int* expectedValues = new int[numManagedPointers];

  chai::managed_ptr<TestInner>* managedPointers = new chai::managed_ptr<TestInner>[numManagedPointers];

  for (int i = 0; i < numManagedPointers; ++i) {
     const int expectedValue = rand();
     expectedValues[i] = expectedValue;
     managedPointers[i] = chai::make_managed<TestInner>(expectedValue);
  }

  for (int i = 0; i < numManagedPointers; ++i) {
     ASSERT_EQ(managedPointers[i]->getValue(), expectedValues[i]);
     managedPointers[i].free();
  }

  delete[] managedPointers;
  delete[] expectedValues;
}

TEST(managed_ptr, c_array_of_managed_ptr)
{
  int numManagedPointers = 10;

  int* expectedValues = new int[numManagedPointers];

  chai::managed_ptr<TestInner>* managedPointers = (chai::managed_ptr<TestInner>*) malloc(numManagedPointers*sizeof(chai::managed_ptr<TestInner>));

  for (int i = 0; i < numManagedPointers; ++i) {
     const int expectedValue = rand();
     expectedValues[i] = expectedValue;
     managedPointers[i] = chai::make_managed<TestInner>(expectedValue);
  }

  for (int i = 0; i < numManagedPointers; ++i) {
     ASSERT_EQ(managedPointers[i]->getValue(), expectedValues[i]);
     managedPointers[i].free();
  }

  free(managedPointers);
  delete[] expectedValues;
}

TEST(managed_ptr, managed_array_of_managed_ptr)
{
  int numManagedPointers = 10;

  int* expectedValues = new int[numManagedPointers];

  chai::ManagedArray<chai::managed_ptr<TestInner>> managedPointers(numManagedPointers, chai::CPU);

  forall(sequential(), 0, numManagedPointers, [=] (int i) {
     const int expectedValue = rand();
     expectedValues[i] = expectedValue;
     managedPointers[i] = chai::make_managed<TestInner>(expectedValue);
  });

  forall(sequential(), 0, numManagedPointers, [=] (int i) {
     ASSERT_EQ(managedPointers[i]->getValue(), expectedValues[i]);
     managedPointers[i].free();
  });

  managedPointers.free();
  delete[] expectedValues;
}

#if defined(CHAI_GPUCC) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE)

template <typename T>
CHAI_GLOBAL void deviceNew(T** arr) {
   *arr = new T[5];
}

template <typename T>
CHAI_GLOBAL void deviceDelete(T** arr) {
   delete[] *arr;
}

CHAI_GLOBAL void passObjectToKernel(const chai::ManagedArray<int> arr) {
   arr[0] = -1;
}

GPU_TEST(managed_ptr, make_on_device)
{
  int** hostArray = (int**) malloc(sizeof(int*));
  hostArray[0] = nullptr;

  int** deviceArray = nullptr;
  chai::gpuMalloc((void**)(&deviceArray), sizeof(int*));

  int** deviceArray2 = nullptr;
  chai::gpuMalloc((void**)(&deviceArray2), sizeof(int*));

  deviceNew<<<1, 1>>>(deviceArray);

  chai::gpuMemcpy(hostArray, deviceArray, sizeof(int*), gpuMemcpyDeviceToHost);
  chai::gpuMemcpy(deviceArray2, hostArray, sizeof(int*), gpuMemcpyHostToDevice);
  ASSERT_NE(hostArray[0], nullptr);

  deviceDelete<<<1, 1>>>(deviceArray2);
  free(hostArray);
  chai::gpuFree(deviceArray);
  chai::gpuFree(deviceArray2);
}

GPU_TEST(managed_ptr, gpu_new_and_delete_on_device)
{
  // Initialize host side memory to hold a pointer
  RawArrayClass** cpuPointerHolder = (RawArrayClass**) malloc(sizeof(RawArrayClass*));
  cpuPointerHolder[0] = nullptr;

  // Initialize device side memory to hold a pointer
  RawArrayClass** gpuPointerHolder = nullptr;
  chai::gpuMalloc((void**)(&gpuPointerHolder), sizeof(RawArrayClass*));

  // Create on the device
  chai::detail::make_on_device<<<1, 1>>>(gpuPointerHolder);

  // Copy to the host side memory
  chai::gpuMemcpy(cpuPointerHolder, gpuPointerHolder, sizeof(RawArrayClass*), gpuMemcpyDeviceToHost);

  // Free device side memory
  chai::gpuFree(gpuPointerHolder);

  // Save the pointer
  ASSERT_NE(cpuPointerHolder[0], nullptr);
  RawArrayClass* gpuPointer = cpuPointerHolder[0];

  // Free host side memory
  free(cpuPointerHolder);

  chai::detail::destroy_on_device<<<1, 1>>>(gpuPointer);
}

GPU_TEST(managed_ptr, gpu_build_managed_ptr)
{
  // Initialize host side memory to hold a pointer
  RawArrayClass** cpuPointerHolder = (RawArrayClass**) malloc(sizeof(RawArrayClass*));
  cpuPointerHolder[0] = nullptr;

  // Initialize device side memory to hold a pointer
  RawArrayClass** gpuPointerHolder = nullptr;
  chai::gpuMalloc((void**)(&gpuPointerHolder), sizeof(RawArrayClass*));

  // Create on the device
  chai::detail::make_on_device<<<1, 1>>>(gpuPointerHolder);

  // Copy to the host side memory
  chai::gpuMemcpy(cpuPointerHolder, gpuPointerHolder, sizeof(RawArrayClass*), gpuMemcpyDeviceToHost);

  // Free device side memory
  chai::gpuFree(gpuPointerHolder);

  // Save the pointer
  ASSERT_NE(cpuPointerHolder[0], nullptr);
  RawArrayClass* gpuPointer = cpuPointerHolder[0];

  // Free host side memory
  free(cpuPointerHolder);

  chai::managed_ptr<RawArrayClass> managedPtr(
      {chai::GPU}, {gpuPointer}, {chai::delete_on_device_deleter<RawArrayClass>()});

  managedPtr.free();
}


GPU_TEST(managed_ptr, pass_object_to_kernel)
{
  const int expectedValue = rand();

  chai::ManagedArray<int> array(1, chai::CPU);

  forall(sequential(), 0, 1, [=] (int i) {
     array[i] = expectedValue;
  });

  chai::ArrayManager* manager = chai::ArrayManager::getInstance();
  manager->setExecutionSpace(chai::GPU);
  passObjectToKernel<<<1, 1>>>(array);
  manager->setExecutionSpace(chai::CPU);
  manager->syncIfNeeded();
  array.move(chai::CPU);
  ASSERT_EQ(array[0], -1);

  array.free();
}

GPU_TEST(managed_ptr, gpu_class_with_raw_array)
{
  const int expectedValue = rand();

  chai::ManagedArray<int> array(1, chai::CPU);

  forall(sequential(), 0, 1, [=] (int i) {
     array[i] = expectedValue;
  });

  auto rawArrayClass = chai::make_managed<RawArrayClass>(chai::unpack(array));
  chai::ManagedArray<int> results(1, chai::GPU);

  forall(gpu(), 0, 1, [=] __device__ (int i) {
    results[i] = rawArrayClass->getValue(i);
  });

  results.move(chai::CPU);
  ASSERT_EQ(results[0], expectedValue);

  results.free();
  rawArrayClass.free();
  array.free();
}

GPU_TEST(managed_ptr, gpu_class_with_raw_array_and_callback)
{
  const int expectedValue = rand();

  chai::ManagedArray<int> array(1, chai::CPU);

  forall(sequential(), 0, 1, [=] (int i) {
     array[i] = expectedValue;
  });

  auto cpuPointer = new RawArrayClass(array.data());
  auto gpuPointer = chai::make_on_device<RawArrayClass>(chai::unpack(array));

  auto callback = [=] (chai::Action action, chai::ExecutionSpace space, void*) mutable -> bool {
     switch (action) {
        case chai::ACTION_FREE:
           switch (space) {
              case chai::NONE:
                 array.free();
                 return true;
              default:
                 return false;
           }
        default:
           return false;
     }
  };

  auto managedPointer = chai::managed_ptr<RawArrayClass>(
      {chai::CPU, chai::GPU},
      {cpuPointer, gpuPointer},
      {
          chai::delete_on_host_deleter<RawArrayClass>(),
          chai::destroy_on_device_deleter<RawArrayClass>()
      },
      callback);

  chai::ManagedArray<int> results(1, chai::GPU);

  forall(gpu(), 0, 1, [=] __device__ (int i) {
    results[i] = managedPointer->getValue(i);
  });

  results.move(chai::CPU);
  ASSERT_EQ(results[0], expectedValue);

  results.free();
  managedPointer.free();
}

GPU_TEST(managed_ptr, gpu_class_with_managed_array)
{
  const int expectedValue = rand();

  chai::ManagedArray<int> array(1, chai::CPU);

  forall(sequential(), 0, 1, [=] (int) {
     array[0] = expectedValue;
  });

  chai::managed_ptr<TestBase> derived = chai::make_managed<TestDerived>(array);

  chai::ManagedArray<int> results(1, chai::GPU);
  
  forall(gpu(), 0, 1, [=] __device__ (int i) {
    results[i] = derived->getValue(i);
  });

  results.move(chai::CPU);

  ASSERT_EQ(results[0], expectedValue);

  results.free();
  derived.free();
  array.free();
}

GPU_TEST(managed_ptr, gpu_class_with_raw_ptr)
{
  const int expectedValue = rand();

  chai::ManagedArray<int> array(1, chai::CPU);

  forall(sequential(), 0, 1, [=] (int) {
     array[0] = expectedValue;
  });

  auto rawArrayClass = chai::make_managed<RawArrayClass>(chai::unpack(array));
  auto rawPointerClass = chai::make_managed<RawPointerClass>(
                           chai::unpack(rawArrayClass));

  chai::ManagedArray<int> results(1, chai::GPU);

  forall(gpu(), 0, 1, [=] __device__ (int i) {
    results[i] = (*rawPointerClass).getValue(i);
  });

  results.move(chai::CPU);
  ASSERT_EQ(results[0], expectedValue);

  results.free();
  rawPointerClass.free();
  rawArrayClass.free();
  array.free();
}

GPU_TEST(managed_ptr, gpu_class_with_managed_ptr)
{
  const int expectedValue = rand();

  auto derived = chai::make_managed<TestInner>(expectedValue);
  TestContainer container(derived);

  chai::ManagedArray<int> results(1, chai::GPU);
  
  forall(gpu(), 0, 1, [=] __device__ (int i) {
    results[i] = container.getValue();
  });

  results.move(chai::CPU);
  ASSERT_EQ(results[0], expectedValue);

  results.free();
  derived.free();
}

GPU_TEST(managed_ptr, gpu_nested_managed_ptr)
{
  const int expectedValue = rand();

  auto derived = chai::make_managed<TestInner>(expectedValue);
  auto container = chai::make_managed<TestContainer>(derived);

  chai::ManagedArray<int> results(1, chai::GPU);

  forall(gpu(), 0, 1, [=] __device__ (int i) {
    results[i] = container->getValue();
  });

  results.move(chai::CPU);
  ASSERT_EQ(results[0], expectedValue);

  results.free();
  container.free();
  derived.free();
}

GPU_TEST(managed_ptr, gpu_multiple_inheritance)
{
  auto derived = chai::make_managed<ClassWithMultipleInheritance>();

  chai::managed_ptr<Base1> base1 = derived;
  chai::managed_ptr<Base2> base2 = derived;

  chai::ManagedArray<bool> results(2, chai::GPU);

  forall(gpu(), 0, 1, [=] __device__ (int i) {
    results[i] = base1->isBase1();
    results[1] = base2->isBase2();
  });

  results.move(chai::CPU);

  ASSERT_EQ(results[0], true);
  ASSERT_EQ(results[1], true);

  results.free();
  base2.free();
}

GPU_TEST(managed_ptr, static_pointer_cast)
{
  const int expectedValue = rand();

  chai::ManagedArray<int> array(1, chai::CPU);

  forall(sequential(), 0, 1, [=] (int) {
     array[0] = expectedValue;
  });

  auto derived = chai::make_managed<TestDerived>(array);
  auto base = chai::static_pointer_cast<TestBase>(derived);
  auto derivedFromBase = chai::static_pointer_cast<TestDerived>(base);

  chai::ManagedArray<int> results(3, chai::GPU);

  forall(gpu(), 0, 1, [=] __device__ (int i) {
    results[i] = derived->getValue(i);
    results[1] = base->getValue(i);
    results[2] = derivedFromBase->getValue(i);
  });

  results.move(chai::CPU);

  ASSERT_EQ(results[0], expectedValue);
  ASSERT_EQ(results[1], expectedValue);
  ASSERT_EQ(results[2], expectedValue);

  results.free();
  derivedFromBase.free();
  array.free();
}

GPU_TEST(managed_ptr, dynamic_pointer_cast)
{
  const int expectedValue = rand();

  chai::ManagedArray<int> array(1, chai::CPU);

  forall(sequential(), 0, 1, [=] (int) {
     array[0] = expectedValue;
  });

  auto derived = chai::make_managed<TestDerived>(array);
  auto base = chai::dynamic_pointer_cast<TestBase>(derived);
  auto derivedFromBase = chai::dynamic_pointer_cast<TestDerived>(base);

  chai::ManagedArray<int> results(3, chai::GPU);

  forall(gpu(), 0, 1, [=] __device__ (int i) {
    results[i] = derived->getValue(i);
    results[1] = base->getValue(i);
    results[2] = derivedFromBase->getValue(i);
  });

  results.move(chai::CPU);

  ASSERT_EQ(results[0], expectedValue);
  ASSERT_EQ(results[1], expectedValue);
  ASSERT_EQ(results[2], expectedValue);

  results.free();
  derivedFromBase.free();
  array.free();
}

GPU_TEST(managed_ptr, const_pointer_cast)
{
  const int expectedValue = rand();

  chai::ManagedArray<int> array(1, chai::CPU);

  forall(sequential(), 0, 1, [=] (int) {
     array[0] = expectedValue;
  });

  auto derived = chai::make_managed<TestDerived>(array);
  auto constDerived = chai::const_pointer_cast<const TestDerived>(derived);
  auto derivedFromConst = chai::const_pointer_cast<TestDerived>(constDerived);

  chai::ManagedArray<int> results(3, chai::GPU);

  forall(gpu(), 0, 1, [=] __device__ (int i) {
    results[i] = derived->getValue(i);
    results[1] = constDerived->getValue(i);
    results[2] = derivedFromConst->getValue(i);
  });

  results.move(chai::CPU);

  ASSERT_EQ(results[0], expectedValue);
  ASSERT_EQ(results[1], expectedValue);
  ASSERT_EQ(results[2], expectedValue);

  results.free();
  constDerived.free();
  array.free();
}

GPU_TEST(managed_ptr, reinterpret_pointer_cast)
{
  const int expectedValue = rand();

  chai::ManagedArray<int> array(1, chai::CPU);

  forall(sequential(), 0, 1, [=] (int) {
     array[0] = expectedValue;
  });

  auto derived = chai::make_managed<TestDerived>(array);
  auto base = chai::reinterpret_pointer_cast<TestBase>(derived);
  auto derivedFromBase = chai::reinterpret_pointer_cast<TestDerived>(base);

  chai::ManagedArray<int> results(3, chai::GPU);

  forall(gpu(), 0, 1, [=] __device__ (int i) {
    results[i] = derived->getValue(i);
    results[1] = base->getValue(i);
    results[2] = derivedFromBase->getValue(i);
  });

  results.move(chai::CPU);

  ASSERT_EQ(results[0], expectedValue);
  ASSERT_EQ(results[1], expectedValue);
  ASSERT_EQ(results[2], expectedValue);

  results.free();
  derivedFromBase.free();
  array.free();
}

#endif

class TestArrayObject {
public:
   CHAI_HOST_DEVICE TestArrayObject() : m_value(-1) {}
   CHAI_HOST_DEVICE TestArrayObject(int value) : m_value(value) {}
   CHAI_HOST_DEVICE virtual ~TestArrayObject() {}
   
   CHAI_HOST_DEVICE int getValue() const { return m_value; }
   CHAI_HOST_DEVICE void setValue(int value) { m_value = value; }

private:
   int m_value;
};

TEST(managed_ptr, ManagedArray_of_managed_ptr_unpacker)
{
  const int size = 5;
  chai::ManagedArray<chai::managed_ptr<TestArrayObject>> array_of_ptrs(size);
  
  // Fill with some test values
  forall(sequential(), 0, size, [=] (int i) {
    array_of_ptrs[i] = chai::make_managed<TestArrayObject>(i * 10);
  });
  
  // Test unpack operation
  auto unpacker = chai::unpack(array_of_ptrs);
  TestArrayObject** raw_ptrs = unpacker.data();
  
  // Verify values can be accessed through the raw pointers
  forall(sequential(), 0, size, [=] (int i) {
    EXPECT_EQ(raw_ptrs[i]->getValue(), i * 10);
    
    // Test modification through raw pointers
    raw_ptrs[i]->setValue(i * 20);
    EXPECT_EQ(array_of_ptrs[i]->getValue(), i * 20);
  });
  
  // Clean up
  forall(sequential(), 0, size, [=] (int i) {
    array_of_ptrs[i].free();
  });
  array_of_ptrs.free();
}

#ifdef CHAI_GPUCC

GPU_TEST(managed_ptr, gpu_ManagedArray_of_managed_ptr_unpacker)
{
  const int size = 5;
  chai::ManagedArray<chai::managed_ptr<TestArrayObject>> array_of_ptrs(size);
  
  // Fill with some test values
  forall(sequential(), 0, size, [=] (int i) {
    array_of_ptrs[i] = chai::make_managed<TestArrayObject>(i * 10);
  });
  
  // Create results array to verify GPU access
  chai::ManagedArray<int> results(size, chai::GPU);
  // Create results array to verify GPU access
  chai::ManagedArray<int> results2(size, chai::GPU);
  
  // Test on GPU
  auto unpacker = chai::unpack(array_of_ptrs);
  
  forall(gpu(), 0, size, [=] __device__ (int i) {
    TestArrayObject** raw_ptrs = unpacker.data();
    results[i] = raw_ptrs[i]->getValue();
    
    // Modify through raw pointers on device
    raw_ptrs[i]->setValue(i * 20);
    results2[i] = array_of_ptrs[i]->getValue();
  });
  
  // Verify results
  forall(sequential(), 0, size, [=] (int i) {
    EXPECT_EQ(results[i], i * 10);
    
    // After GPU execution, check that values were modified
    EXPECT_EQ(results2[i], i * 20);
  });
  
  // Clean up
  results.free();
  results2.free();
  forall(sequential(), 0, size, [=] (int i) {
    array_of_ptrs[i].free();
  });
  array_of_ptrs.free();
}

#endif

// Class hierarchy for testing polymorphic behavior
class ABase {
public:
   CHAI_HOST_DEVICE ABase() {}
   CHAI_HOST_DEVICE virtual ~ABase() {}
   
   // Virtual function to manipulate an array of objects
   CHAI_HOST_DEVICE virtual void setArrayValues(int /* size */, int /* value */) {
      // Base implementation does nothing
   }
   
   CHAI_HOST_DEVICE virtual int getTypeID() const { return 0; }
};

class BDerived : public ABase {
public:
   // Constructor that takes a TestArrayObject**
   CHAI_HOST_DEVICE BDerived(TestArrayObject** objects) : ABase(), m_objects(objects) {}
   CHAI_HOST_DEVICE virtual ~BDerived() {}
   
   // Override to implement array manipulation on member array
   CHAI_HOST_DEVICE virtual void setArrayValues(int size, int value) override {
      for (int i = 0; i < size; i++) {
         m_objects[i]->setValue(value * (i + 1));
      }
   }
   
   CHAI_HOST_DEVICE virtual int getTypeID() const override { return 1; }
   
private:
   TestArrayObject** m_objects; // Member array of TestArrayObject pointers
};

TEST(managed_ptr, polymorphic_with_ManagedArray_unpacker)
{
   // Create array of managed pointers
   const int size = 5;
   chai::ManagedArray<chai::managed_ptr<TestArrayObject>> array_of_objects(size);
   
   // Initialize array
   forall( sequential(), 0, size ,[=](int i ) {
      array_of_objects[i] = chai::make_managed<TestArrayObject>(0);
   });
   
   // Create unpacker and keep it alive for the duration of the test
   auto unpacker = chai::unpack(array_of_objects);
   
   // Create derived object with the unpacker
   chai::managed_ptr<ABase> poly_ptr = chai::make_managed<BDerived>(unpacker.data());
   
   // Verify correct polymorphic behavior
   EXPECT_EQ(poly_ptr->getTypeID(), 1);
   
   // Use virtual function to modify array through the member pointer
   const int base_value = 10;
   forall( sequential(), 0, 1, [=] (int /* i */) {
      poly_ptr->setArrayValues(size, base_value);
   });

   // Verify values set by polymorphic method
   forall( sequential(), 0, size ,[=](int i ) {
      EXPECT_EQ(array_of_objects[i]->getValue(), base_value * (i + 1));
   });
   
   // Cleanup
   forall( sequential(), 0, size ,[=](int i ) {
      array_of_objects[i].free();
   });
   array_of_objects.free();
   poly_ptr.free();
}

#ifdef CHAI_GPUCC

GPU_TEST(managed_ptr, gpu_polymorphic_with_ManagedArray_unpacker)
{
   // Create array of managed pointers
   const int size = 5;
   chai::ManagedArray<chai::managed_ptr<TestArrayObject>> array_of_objects(size);
   
   // Initialize array
   forall( sequential(), 0, size ,[=](int i ) {
      array_of_objects[i] = chai::make_managed<TestArrayObject>(0);
   });
   
   // Create unpacker and keep it alive for the duration of the test
   auto unpacker = chai::unpack(array_of_objects);
   
   // Create derived object with the unpacker
   chai::managed_ptr<ABase> poly_ptr = chai::make_managed<BDerived>(unpacker);
   
   // Use virtual function on device
   const int base_value = 10;
   
   forall(gpu(), 0, 1, [=] __device__ (int) {
      // Call polymorphic method on device to modify the array through member pointer
      poly_ptr->setArrayValues(size, base_value);
   });
   
   // Test results on device
   auto results_unpacker = chai::unpack(array_of_objects);
   chai::ManagedArray<int> results(size, chai::GPU);
   
   forall(gpu(), 0, size, [=] __device__ (int i) {
      TestArrayObject** raw_ptrs = results_unpacker.data();
      results[i] = raw_ptrs[i]->getValue();
   });
   
   // Verify results
   forall( sequential(), 0, size ,[=](int i ) {
      EXPECT_EQ(results[i], base_value * (i + 1));
   });
   
   // Cleanup
   results.free();
   forall( sequential(), 0, size ,[=](int i ) {
      array_of_objects[i].free();
   });
   array_of_objects.free();
   poly_ptr.free();
}

#endif
#if 0 // TODO: Enable if/when ManagedArrays of managed_ptrs can be handled correctly.

class RawArrayOfPointersClass {
   public:
      CHAI_HOST_DEVICE RawArrayOfPointersClass() = delete;
      CHAI_HOST_DEVICE RawArrayOfPointersClass(RawArrayClass** arrayOfPointers) :
         m_arrayOfPointers(arrayOfPointers)
      {}

      CHAI_HOST_DEVICE int getValue(const int i, const int j) const {
         return m_arrayOfPointers[i]->getValue(j);
      }

   private:
      RawArrayClass** m_arrayOfPointers = nullptr;
};

TEST(managed_ptr, class_with_raw_array_of_pointers)
{
  const int expectedValue = rand();

  chai::ManagedArray<int> array(1, chai::CPU);
  array[0] = expectedValue;

  auto rawArrayClass = chai::make_managed<RawArrayClass>(chai::unpack(array));
  chai::managed_ptr<RawArrayClass> arrayOfPointers[1] = {rawArrayClass};

  auto rawArrayOfPointersClass = chai::make_managed<RawArrayOfPointersClass>(arrayOfPointers);
  ASSERT_EQ(rawArrayOfPointersClass->getValue(0, 0), expectedValue);
}

#endif
