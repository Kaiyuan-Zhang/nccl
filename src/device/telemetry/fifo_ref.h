#ifndef TELEMETRY_DEVICE_INCLUDE_FIFO_REF_H_
#define TELEMETRY_DEVICE_INCLUDE_FIFO_REF_H_

#include "cuda_compat.h"

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y)-1) / (y))
#endif

#ifndef ROUNDUP
#define ROUNDUP(x, y) (DIVUP((x), (y)) * (y))
#endif

#define FIFO_ALIGNMENT (128)

namespace telemetry {
#if __NVCC__
__device__ __forceinline__ uint64_t ld_volatile_global(uint64_t* ptr) {
  uint64_t ans;
  asm volatile("ld.volatile.global.u64 %0, [%1];"
               : "=l"(ans)
               : "l"(__cvta_generic_to_global(ptr))
               : "memory");
  return ans;
}
__device__ __forceinline__ void st_volatile_global(uint64_t* ptr,
                                                   uint64_t val) {
  asm volatile(
      "st.volatile.global.u64 [%0], %1;" ::"l"(__cvta_generic_to_global(ptr)),
      "l"(val)
      : "memory");
}
#endif

// Reference to the spsc FIFO object
//
// This object is created for operating on the FIFO,
// the underlying buffer should live longer than the object.
template <typename T>
class SpscFifoRef {
 public:
  __device__ __host__ SpscFifoRef() : head_(nullptr) {}

  __device__ __host__ SpscFifoRef(void* buffer, size_t size_log2) {
    // FIFO layout:
    // head and tail followed by element array
    // head and tail are 64-bit integer but are FIFO_ALIGNMENT aligned
    // each element contains a T value and is also FIFO_ALIGNMENT aligned
    head_ = (uint64_t*)buffer;
    tail_ = (uint64_t*)((uint8_t*)buffer + FIFO_ALIGNMENT);
    data_ = (uint8_t*)buffer + 2 * FIFO_ALIGNMENT;
    size_ = 1UL << size_log2;
    size_mask_ = size_ - 1;
#ifdef __CUDA_ARCH__
    head_cache_ = ld_volatile_global(head_);
    tail_cache_ = ld_volatile_global(tail_);
#else
    head_cache_ = __atomic_load_n(head_, __ATOMIC_ACQUIRE);
    tail_cache_ = __atomic_load_n(tail_, __ATOMIC_ACQUIRE);
#endif
  }

  __device__ __host__ bool IsActive() { return head_ != nullptr; }

  // Get entry by idx
  __device__ __host__ T* GetEntry(size_t idx) {
    return (T*)(data_ + ROUNDUP(sizeof(T), FIFO_ALIGNMENT) * idx);
  }

  __device__ __host__ T* NextWrite() {
    if (tail_cache_ - head_cache_ >= size_) {
#ifdef __CUDA_ARCH__
      head_cache_ = ld_volatile_global(head_);
#else
      head_cache_ = __atomic_load_n(head_, __ATOMIC_ACQUIRE);
#endif
      if (tail_cache_ - head_cache_ >= size_) {
        return nullptr; // Queue is full! Lossy drop.
      }
    }
    return GetEntry(tail_cache_ & size_mask_);
  }

  __device__ __host__ void CommitWrite() {
    tail_cache_++;
#ifdef __CUDA_ARCH__
    st_volatile_global(tail_, tail_cache_);
#else
    __atomic_store_n(tail_, tail_cache_, __ATOMIC_RELEASE);
#endif
  }

  // max_retry == 0 means block indefinitely
  __device__ __host__ T* NextRead(int max_retry = 0) {
    while (head_cache_ >= tail_cache_) {
#ifdef __CUDA_ARCH__
      tail_cache_ = ld_volatile_global(tail_);
#else
      tail_cache_ = __atomic_load_n(tail_, __ATOMIC_ACQUIRE);
#endif
      if (max_retry > 0) {
        max_retry--;
        if (max_retry == 0) {
          return nullptr;
        }
      }
    }
    return GetEntry(head_cache_ & size_mask_);
  }

  __device__ __host__ void FinishRead() {
    head_cache_++;
#ifdef __CUDA_ARCH__
    st_volatile_global(head_, head_cache_);
#else
    __atomic_store_n(head_, head_cache_, __ATOMIC_RELEASE);
#endif
  }

 private:
  uint8_t* data_;
  uint64_t* head_;
  uint64_t* tail_;
  uint64_t head_cache_;
  uint64_t tail_cache_;
  uint64_t size_;
  uint64_t size_mask_;
};

// Reference to the mpsc FIFO object
// Note that this ref object should not be shared among producers
//
// This object is created for operating on the FIFO,
// the underlying buffer should live longer than the object.
template <typename T>
class MpscFifoRef {
 public:
  // Entry of FIFO
  // for MPSC FIFO, we need to use an extra field, "version",
  // to mark the readiness of data
  struct Entry {
    T data;
    uint64_t version;
  };

  __device__ __host__ MpscFifoRef() : head_(nullptr) {}

  __device__ __host__ MpscFifoRef(void* buffer, size_t size_log2) {
    // FIFO layout:
    // head and tail followed by element array
    // head and tail are 64-bit integer but are FIFO_ALIGNMENT aligned
    // each element contains a T value and is also FIFO_ALIGNMENT aligned
    head_ = (uint64_t*)buffer;
    tail_ = (uint64_t*)((uint8_t*)buffer + FIFO_ALIGNMENT);
    data_ = (uint8_t*)buffer + 2 * FIFO_ALIGNMENT;
    size_ = 1UL << size_log2;
    size_mask_ = size_ - 1;
#ifdef __CUDA_ARCH__
    head_cache_ = ld_volatile_global(head_);
    tail_cache_ = ld_volatile_global(tail_);
#else
    head_cache_ = __atomic_load_n(head_, __ATOMIC_ACQUIRE);
    tail_cache_ = __atomic_load_n(tail_, __ATOMIC_ACQUIRE);
#endif
  }

  __device__ __host__ bool IsActive() { return head_ != nullptr; }

  // Get entry by idx
  __device__ __host__ Entry* GetEntry(size_t idx) {
    return (Entry*)(data_ + ROUNDUP(sizeof(Entry), FIFO_ALIGNMENT) * idx);
  }

  __device__ __host__ T* NextWrite() {
#ifdef __CUDA_ARCH__
    uint64_t tail = ld_volatile_global(tail_);
#else
    uint64_t tail = __atomic_load_n(tail_, __ATOMIC_ACQUIRE);
#endif
    while (true) {
      if (tail - head_cache_ >= size_) {
#ifdef __CUDA_ARCH__
        head_cache_ = ld_volatile_global(head_);
#else
        head_cache_ = __atomic_load_n(head_, __ATOMIC_ACQUIRE);
#endif
        if (tail - head_cache_ >= size_) {
          return nullptr; // Queue is full! Lossy drop.
        }
      }

#ifdef __CUDA_ARCH__
      uint64_t old_tail = atomicCAS((unsigned long long *)tail_, tail, tail + 1);
#else
      uint64_t old_tail = tail;
      __atomic_compare_exchange_n(tail_, &old_tail, tail + 1, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
#endif
      if (old_tail == tail) {
        tail_cache_ = tail;
        break;
      }
      tail = old_tail;
    }
    return &GetEntry(tail_cache_ & size_mask_)->data;
  }

  __device__ __host__ void CommitWrite() {
    Entry* e = GetEntry(tail_cache_ & size_mask_);
#ifdef __CUDA_ARCH__
    st_volatile_global(&e->version, tail_cache_ + 1);
#else
    __atomic_store_n(&e->version, tail_cache_ + 1, __ATOMIC_RELEASE);
#endif
  }

  // max_retry == 0 means block indefinitely
  __device__ __host__ T* NextRead(int max_retry = 0) {
    while (head_cache_ >= tail_cache_) {
#ifdef __CUDA_ARCH__
      tail_cache_ = ld_volatile_global(tail_);
#else
      tail_cache_ = __atomic_load_n(tail_, __ATOMIC_ACQUIRE);
#endif
      if (max_retry > 0) {
        max_retry--;
        if (max_retry == 0) {
          return nullptr;
        }
      }
    }
    // now we know that a producer has allocated a slot and is writing data
    // we can poll the version
    Entry* e = GetEntry(head_cache_ & size_mask_);
    uint64_t version;
    do {
#ifdef __CUDA_ARCH__
      version = ld_volatile_global(&e->version);
#else
      version = __atomic_load_n(&e->version, __ATOMIC_ACQUIRE);
#endif
    } while (version != head_cache_ + 1);
    return &e->data;
  }

  __device__ __host__ void FinishRead() {
    head_cache_++;
#ifdef __CUDA_ARCH__
    st_volatile_global(head_, head_cache_);
#else
    __atomic_store_n(head_, head_cache_, __ATOMIC_RELEASE);
#endif
  }

 private:
  uint8_t* data_;
  uint64_t* head_;
  uint64_t* tail_;
  uint64_t head_cache_;
  uint64_t tail_cache_;
  uint64_t size_;
  uint64_t size_mask_;
};

}  // namespace telemetry

#endif /* TELEMETRY_DEVICE_INCLUDE_FIFO_REF_H_ */
