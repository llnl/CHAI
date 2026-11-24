#ifndef CHAI_PARRAY_HPP
#define CHAI_PARRAY_HPP

#include "chai/config.hpp"

namespace chai {
   template <class T, class Manager>
   class PArray
   {
      public:
         constexpr PArray() = default;

         PArray(size_t size)
         {
            if (size > 0)
            {
               allocate(size);
            }
         }

         CHAI_HOST_DEVICE PArray(const PArray& other)
            : m_data{other.m_data},
              m_size{other.m_size},
              m_manager{other.m_manager}
         {
#if !defined(CHAI_DEVICE_COMPILE)
            if (m_manager)
            {
               m_data = m_manager->data();
            }
#endif
         }

         CHAI_HOST_DEVICE PArray& operator=(const PArray& other) = default;

         void reallocate(size_t size)
         {
            if (size == 0)
            {
               deallocate();
            }
            else if (m_size == 0)
            {
               allocate();
            }
            else
            {
               m_data = m_manager->reallocate(size);
               m_size = size;
            }
         }

         void deallocate()
         {
            if (m_manager) {
               delete m_manager;
               m_manager = nullptr;
               m_data = nullptr;
               m_size = 0;
            }
         }

         CHAI_HOST_DEVICE size_t size() const
         {
            return m_size;
         }

         CHAI_HOST_DEVICE T& operator[](size_t i) const
         {
            return m_data[i];
         }

         CHAI_HOST_DEVICE T* data() const
         {
#if !defined(CHAI_DEVICE_COMPILE)
            if (m_manager)
            {
               m_data = m_manager->data();
            }
#endif
            return m_data;
         }

         CHAI_HOST_DEVICE const T* cdata() const
         {
#if !defined(CHAI_DEVICE_COMPILE)
            if (m_manager)
            {
               m_data = m_manager->cdata();
            }
#endif
            return m_data;
         }

      private:
         T* m_data = nullptr;
         size_t m_size = 0;
         Manager* m_manager = nullptr;

         void allocate(size_t size)
         {
            m_manager = new Manager();
            m_data = m_manager->allocate(size);
            m_size = size;
         }
   };  // class PArray
}  // namespace chai

#endif  // CHAI_PARRAY_HPP
