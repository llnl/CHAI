#ifndef CHAI_HOST_MEMORY_RESOURCE_HPP
#define CHAI_HOST_MEMORY_RESOURCE_HPP

#include "chai/MemoryType.hpp"

namespace chai {
   template <class AllocatorType>
   class HostMemoryResource {
      public:
         ~HostMemoryResource() {
            deallocate();
         }

         void* allocate(size_t size) {
            m_data = m_allocator.allocate(MemoryType::Host, size);
            m_size = size;
            return m_data;
         }

         void* reallocate(size_t size) {
            void* newData = m_allocator.allocate(MemoryType::Host, size);
            memcpy(newData, m_data, size < m_size ? size : m_size);
            m_allocator.deallocate(MemoryType::Host, m_data);
            m_data = newData;
            m_size = size;
            return m_data;
         }

         void deallocate() {
            m_allocator.deallocate(MemoryType::Host, m_data);
            m_data = nullptr;
            m_size = 0;
         }

         void* data(bool touch) {
            return data(MemoryResourcePlugin::getExecutionSpace(), touch);
         }

         void* data(ExecutionSpace /* executionSpace */, bool /* touch */) {
            return m_data;
         }

      private:
         void* m_data = nullptr;
         size_t m_size = 0;
         AllocatorType m_allocator;
   };  // class HostMemoryResource

}  // namespace chai

#endif  // CHAI_HOST_MEMORY_RESOURCE_HPP
