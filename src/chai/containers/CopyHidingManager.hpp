#ifndef CHAI_COPY_HIDING_MANAGER_HPP
#define CHAI_COPY_HIDING_MANAGER_HPP

namespace chai {
   template <class T, class Allocator>
   class CopyHidingManager {
      public:
         ~CopyHidingManager()
         {
            m_allocator.deallocate(chai::GPU, m_gpu_data);
            m_allocator.deallocate(chai::CPU, m_cpu_data);
         }

         T* allocate(size_t size)
         {
            m_gpu_data = m_allocator.allocate(chai::GPU, size);
            m_cpu_data = m_allocator.allocate(chai::CPU, size);
         }

         T* reallocate(size_t size)
         {
            m_gpu_data = m_allocator.reallocate(chai::GPU, size);
            m_cpu_data = m_allocator.reallocate(chai::CPU, size);
         }

         T* data(ExecutionSpace executionSpace,
                 bool update = true,
                 bool touch = true)
         {
            if (update && executionSpace != m_space)
            {

            }

            if (touch)
            {
               m_space = executionSpace;
            }
            else
            {
               m_space = NONE;
            }

            if (executionSpace == CPU)
            {
               return m_cpu_data;
            }
            else if (executionSpace == GPU)
            {
               return m_gpu_data;
            }
            else
            {
               return nullptr;
            }
         }

      private:
         T* m_cpu_data = nullptr;
         T* m_gpu_data = nullptr;
         size_t m_size = 0;
         chai::ExecutionSpace m_space = NONE;
         Allocator m_allocator;

   };
}  // namespace chai

#endif  // CHAI_COPY_HIDING_MANAGER_HPP
