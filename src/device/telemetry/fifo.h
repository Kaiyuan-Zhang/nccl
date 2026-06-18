#ifndef TELEMETRY_DEVICE_INCLUDE_FIFO_H_
#define TELEMETRY_DEVICE_INCLUDE_FIFO_H_

#include <cstdint>
#include <type_traits>

#include "cuda_compat.h"
#include "internal/gpu_mem.h"
#include "fifo_ref.h"

namespace telemetry {

template <typename T>
class SpscFifo {
 public:
  SpscFifo(int device, size_t size_log2)
      : host_mem_(GetBufferSize(size_log2), internal::GetGpuNumaId(device)),
        size_log2_(size_log2) {
    uint64_t* head = (uint64_t*)host_mem_.HostPtr();
    uint64_t* tail = (uint64_t*)((uint8_t*)head + FIFO_ALIGNMENT);
    __atomic_store_n(head, 0, __ATOMIC_RELEASE);
    __atomic_store_n(tail, 0, __ATOMIC_RELEASE);
  }

  // Spsc is neither copyable nor movable
  SpscFifo(const SpscFifo&) = delete;
  SpscFifo& operator=(const SpscFifo&) = delete;

  size_t SizeLog2() const { return size_log2_; }

  void* HostPtr() { return host_mem_.HostPtr(); }
  CUdeviceptr DevPtr(int device) { return host_mem_.MapOnGpu(device); }

  SpscFifoRef<T> FifoRef() {
    return { HostPtr(), SizeLog2() };
  }

 private:
  static size_t GetBufferSize(size_t size_log2) {
    size_t num_entry = 1UL << size_log2;
    size_t entry_size = ROUNDUP(sizeof(T), FIFO_ALIGNMENT);
    return num_entry * entry_size + FIFO_ALIGNMENT * 2;
  }

  internal::MappableHostMem host_mem_;
  size_t size_log2_;
};

template <typename T>
class MpscFifo {
 public:
  MpscFifo(int device, size_t size_log2)
      : host_mem_(GetBufferSize(size_log2), internal::GetGpuNumaId(device)),
        size_log2_(size_log2) {
    uint64_t* head = (uint64_t*)host_mem_.HostPtr();
    uint64_t* tail = (uint64_t*)((uint8_t*)head + FIFO_ALIGNMENT);
    __atomic_store_n(head, 0, __ATOMIC_RELEASE);
    __atomic_store_n(tail, 0, __ATOMIC_RELEASE);

    size_t entry_size = ROUNDUP(sizeof(typename MpscFifoRef<T>::Entry), FIFO_ALIGNMENT);
    uint8_t* ptr = &((uint8_t *)host_mem_.HostPtr())[FIFO_ALIGNMENT * 2];
    for (size_t i = 0; i < (1UL << size_log2_); ++i) {
      typename MpscFifoRef<T>::Entry* e = (typename MpscFifoRef<T>::Entry *)&ptr[entry_size * i];
      __atomic_store_n(&e->version, 0, __ATOMIC_RELEASE);
    }
  }

  // Spsc is neither copyable nor movable
  MpscFifo(const MpscFifo&) = delete;
  MpscFifo& operator=(const MpscFifo&) = delete;

  size_t SizeLog2() const { return size_log2_; }

  void* HostPtr() { return host_mem_.HostPtr(); }
  CUdeviceptr DevPtr(int device) { return host_mem_.MapOnGpu(device); }

  MpscFifoRef<T> FifoRef() {
    return { HostPtr(), SizeLog2() };
  }

 private:
  static size_t GetBufferSize(size_t size_log2) {
    size_t num_entry = 1UL << size_log2;
    size_t entry_size = ROUNDUP(sizeof(typename MpscFifoRef<T>::Entry), FIFO_ALIGNMENT);
    return num_entry * entry_size + FIFO_ALIGNMENT * 2;
  }

  internal::MappableHostMem host_mem_;
  size_t size_log2_;
};

}  // namespace telemetry

#endif /* TELEMETRY_DEVICE_INCLUDE_FIFO_H_ */
