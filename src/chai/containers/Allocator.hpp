#ifndef CHAI_ALLOCATOR_HPP
#define CHAI_ALLOCATOR_HPP

#include "chai/MemoryType.hpp"

namespace chai {
   class Allocator {
      public:
         static void setAllocator(MemoryType type, int id) {
            const umpire::Allocator& allocator =
               umpire::ResourceManager::getInstance().getAllocator(id);

            switch (type) {
               case MemoryType::Host:
                  getHostAllocator() = allocator;
                  break;
            }
         }

         void* allocate(MemoryType type, size_t size) {
            switch (type) {
               case MemoryType::Host:
                  return getHostAllocator().allocate(size);
            }
         }

         void deallocate(MemoryType type, void* data) {
            switch (type) {
               case MemoryType::Host:
                  getHostAllocator().deallocate(data);
            }
         }

      private:
         static umpire::Allocator& getHostAllocator() {
            static umpire::Allocator s_hostAllocator =
               umpire::ResourceManager::getInstance().getAllocator("HOST");

            return s_hostAllocator;
         }
   };  // namespace Allocator
}  // namespace chai

#endif  // CHAI_ALLOCATOR_HPP
