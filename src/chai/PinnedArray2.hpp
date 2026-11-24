#ifndef CHAI_PINNED_ARRAY_HPP
#define CHAI_PINNED_ARRAY_HPP

namespace chai {
   class MemoryManager {
      virtual void update(camp::resources::Resource& resource) = 0;

      virtual void update(camp::resources::Resource& resource,
                          camp::resources::Event& event) = 0;
   };

   class MemoryManagerPlugin : RAJA::util::PluginStrategy {
      public:
         void preCapture(camp::resources::Resource& resource) override {
            m_isCapturing = true;
         }

         void postCapture(camp::resources::Resource& resource) override {
            m_isCapturing = false;

            for (MemoryManager* manager : m_managers) {
               manager->update(resource);
            }
         }

         void postLaunch(camp::resources::Resource& resource) override {
            camp::resources::Event event = resource.get_event();

            for (MemoryManager* manager : m_managers) {
               manager->update(resource, event);
            }

            m_managers.clear();
         }

         void registerMemoryManager(MemoryManager* manager) {
            if (m_isCapturing) {
               m_managers.push_back(manager);
            }
         }

      private:
         bool m_isCapturing{false};
         std::vector<MemoryManager*> m_managers;
   };

   class HipMallocMemoryManager : public MemoryManager
   {
      public:
         HipMallocMemoryManager(size_t size, camp::resources::Hip& resource)
           : MemoryManager(),
             m_size(size),
             m_last_resource(resource)
         {
            hipMallocAsync(&m_data, size, resource.get_stream());
            m_event = resource.get_event();
         }

         ~HipMallocMemoryManager()
         {
            hipFreeAsync(m_data, m_last_resource.get_stream());
         }

         size_t size() const {
            return m_size;
         }

         void update(camp::resources::Resource& resource) override {
            if (resource != m_last_resource) {
               m_event.wait();
               m_event = camp::resources::Event();
            }
         }

         void update(camp::resources::Resource& resource,
                     camp::resources::Event& event) override {
            m_last_resource = resource;
            m_last_event = event;
         }

         void* data(camp::resources::Resource resource) {
            if (resource != m_last_resource) {
               m_event.wait();
            }

            return m_data;
         }

         void reset(camp::resources::Resource resource) {
            m_last_resource = resource;
            m_event = resource.get_event();
         }

      private:
         size_t m_size = 0;
         void* m_data = nullptr;
         camp::resources::Resource m_last_resource;
         camp::resources::Event m_event;
   };

   template <class T>
   class PinnedArray {
      public:
         static PinnedArray allocate(size_t count,
                                     camp::resources::Hip resource) {
            if (count == 0) {
               return PinnedArray();
            }
            else {
               T* data;
               hipMallocAsync(&data, count * sizeof(T), resource.get_stream());
               ControlBlock* control = new ControlBlock(count, data, resource);
               return PinnedArray(count, data, control);
            }
         }

         static PinnedArray allocate(size_t count) {
            return allocate(count, camp::resources::Hip::get_default());
         }

         static void deallocate(PinnedArray& ptr,
                                camp::resources::Hip resource) {
            ControlBlock* control = ptr.get_control();

            if (control) {
               if (resource != control->m_last_resource) {
                  control->m_last_event.wait();
               }

               hipFreeAsync(control->m_data, resource.get_stream());
               delete m_control;
               ptr.reset();
            }
         }

         PinnedArray() = default;

         PinnedArray(size_t count,
                     T* data,
                     ControlBlock* control) :
            m_size{count},
            m_data{data},
            m_control{control}
         {
         }

         CHAI_HOST_DEVICE PinnedArray(const Pinned& other) :
            m_size{other.m_size},
            m_data{other.m_data},
            m_control{other.m_control}
         {
#if !defined(__HIP_DEVICE_COMPILE__)
            // TODO: Implement callback
            PinnedPlugin::registerCallback();
            m_data = static_cast<T*>(m_manager->data<T>());
#endif
         }

         ControlBlock* get_control() {
            return m_control;
         }

         void reset() {
            m_size = 0;
            m_data = nullptr;
            m_control = nullptr;
         }

         void resize(size_t count) {
            if (count == 0) {
               free();
            }
            else if (m_size == 0) {
               camp::resources::Hip resource = camp::resources::Hip::get_default();
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
            ControlBlock(size_t count,
                         T* data,
                         camp::resources::Resource resource) :
               m_count{count},
               m_data{data},
               m_resource{resource},
               m_event{resource.get_event()}
            {}

            size_t m_count = 0;
            T* m_data = nullptr;
            camp::resources::Resource m_last_resource;
            camp::resources::Event m_last_event;
         };

         size_t m_size = 0;
         T* m_data = nullptr;
         ControlBlock* m_control = nullptr;
   };

}  // namespace care

#endif  // CARE_PINNED_ARRAY_HPP
