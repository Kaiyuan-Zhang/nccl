#ifndef TELEMETRY_DEVICE_SRC_GPU_MEM_H_
#define TELEMETRY_DEVICE_SRC_GPU_MEM_H_

#include <cuda.h>

#include <optional>

namespace telemetry {
namespace internal {

class GpuMem {
 public:
  GpuMem(int device, size_t size);

  // GpuMem is neither copyable nor movable
  GpuMem(const GpuMem&) = delete;
  GpuMem& operator=(const GpuMem&) = delete;

  void* DevPtr() { return reinterpret_cast<void*>(device_ptr_); }

  ~GpuMem();

 private:
  CUmemGenericAllocationHandle handle_;
  size_t actual_size_;
  CUdeviceptr device_ptr_;
};

class MappableHostMem {
 public:
  explicit MappableHostMem(size_t size);
  MappableHostMem(size_t size, int numa_id);

  // MappableHostMem is neither copyable nor movable
  MappableHostMem(const MappableHostMem&) = delete;
  MappableHostMem& operator=(const MappableHostMem&) = delete;

  void* HostPtr() { return reinterpret_cast<void*>(host_ptr_); }
  CUdeviceptr MapOnGpu(int device);

  ~MappableHostMem();

 private:
  MappableHostMem(size_t size, std::optional<int> numa_id);

  CUmemGenericAllocationHandle handle_;
  std::optional<int> numa_id_;
  size_t actual_size_;
  CUdeviceptr host_ptr_;
};

int GetGpuNumaId(int device);

}  // namespace internal
}  // namespace telemetry

#endif /* TELEMETRY_DEVICE_SRC_GPU_MEM_H_ */
