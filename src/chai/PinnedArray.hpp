#if !defined(CARE_PINNED_ARRAY_HPP)
#define CARE_PINNED_ARRAY_HPP

namespace care {

   template <class T>
   class PinnedArray {
      public:
         PinnedArray() = default;

         PinnedArray(const Pinned& other) :
            m_size{other.m_size},
            m_data{other.m_data},
            m_control{other.m_control}
         {
#if !defined(__HIP_DEVICE_COMPILE__)
            // TODO: Implement callback
            PinnedPlugin::registerCallback();
#endif
         }

         void resize(size_t count) {
            if (count == 0) {
               free();
            }
            else if (m_size == 0) {
               cuda::resources::Hip resource = cuda::resources::Hip::get_default();
               hipMallocAsync(&m_data, count * sizeof(T), resource.get_stream());
               m_control = new ControlBlock();
               m_control->m_last_resource = resource;
               m_control->m_last_event = resource.get_event();
            }
            else if (m_size != count) {
               T* newData = nullptr;
               hipMallocAsync(&newData, count * sizeof(T), m_control->m_last_resource.get_stream());
               const size_t min = m_size < count ? m_size : count;
               hipMemcpyAsync(newData, m_data, min * sizeof(T), hipMemcpyDeviceToDevice, m_control->m_last_resource);
               m_control->m_last_event = m_control->m_last_resource.get_event();
               hipFreeAsync(m_data, m_control->m_last_resource);
               m_data = newData;
            }

            m_size = count;
         }

         void free() {
            if (m_control) {
               hipFreeAsync(m_data, m_control->m_last_resource.get_stream());
               delete m_control;
               m_control = nullptr;
               m_data = nullptr;
               m_size = 0;
            }

         T& operator[](size_t i) const {
            return m_data[i];
         }

      private:
         struct ControlBlock {
            camp::resources::Resource m_last_resource;
            camp::resources::Event m_last_event;
         };

         size_t m_size = 0;
         T* m_data = nullptr;
         ControlBlock* m_control = nullptr;
   };

}  // namespace care

#endif  // CARE_PINNED_ARRAY_HPP
